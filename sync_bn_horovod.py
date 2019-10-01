# MIT License

# Copyright (c) 2019 kaiJIN

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import torch
import time
from torch import nn
from torchvision import models
from torch.nn import functional as F
import horovod.torch as hvd
from sync_bn import SynchronizedBatchNorm1d
from sync_bn import SynchronizedBatchNorm2d
from sync_bn import SynchronizedBatchNorm3d
import functools


def view(name, tensor):
  info = '[NAME]: {:25s}'.format(name)
  info += ' [SHAPE]: {:15s}'.format(str(list(tensor.shape)))
  info += ' [SUM]: {:.7f}, [MEAN]: {:.7f}, [MAX]: {:.7f}, [MIN]: {:.7f}, [VAR]: {:.7f}'
  info = info.format(tensor.sum(), tensor.mean(),
                     tensor.max(), tensor.min(), tensor.var())
  print(info)


def convert_model(module):
  mod = module
  for pth_module, sync_module in zip([nn.BatchNorm1d,
                                      nn.BatchNorm2d,
                                      nn.BatchNorm3d],
                                     [SynchronizedBatchNorm1d,
                                      SynchronizedBatchNorm2d,
                                      SynchronizedBatchNorm3d]):
    if isinstance(module, pth_module):
      mod = sync_module(module.num_features, module.eps,
                        module.momentum, module.affine)
      mod.running_mean = module.running_mean
      mod.running_var = module.running_var
      if module.affine:
        mod.weight.data = module.weight.data.clone().detach()
        mod.bias.data = module.bias.data.clone().detach()
  for name, child in module.named_children():
    mod.add_module(name, convert_model(child))
  return mod


class LeNet(nn.Module):
  def __init__(self, norm_layer):
    super(LeNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 200, 5)
    self.bn = norm_layer(200)
    self.conv2 = nn.Conv2d(200, 16, 5)
    self.bn1 = norm_layer(16)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    out = F.relu(self.bn(self.conv1(x)))
    out = F.max_pool2d(out, 2)
    out = F.relu(self.bn1(self.conv2(out)))
    out = F.max_pool2d(out, 2)
    out = out.view(out.size(0), -1)
    out = F.relu(self.fc1(out))
    out = F.relu(self.fc2(out))
    out = self.fc3(out)
    return out


def test_resnet(affine, track_running_stats):
  torch.manual_seed(0)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  if torch.cuda.is_available():
    device = torch.device('cuda:%d' % hvd.rank())
  else:
    device = torch.device('cpu')

  if hvd.rank() == 0:
    print('affine:', affine, 'track_running_stats', track_running_stats)

  bn = functools.partial(
      nn.BatchNorm2d,
      track_running_stats=track_running_stats,
      affine=affine)
  sync_bn = functools.partial(
      SynchronizedBatchNorm2d,
      track_running_stats=track_running_stats,
      affine=affine)

  # prepare model
  model = models.resnet18(norm_layer=bn).to(device)
  sync_model = models.resnet18(norm_layer=sync_bn).to(device)
  sync_model.load_state_dict(model.state_dict())
  # print(sync_model)

  # prepare inputs
  num_samples = 8
  num_steps = 3
  inputs = torch.rand(num_steps, num_samples, 3, 32, 32).float().to(device)
  start_idx = hvd.rank() * int(num_samples / hvd.size())
  end_idx = (hvd.rank() + 1) * int(num_samples / hvd.size())

  # test training
  if hvd.rank() == 0:
    print('[TRAINING PHASE]')

  # using pytorch-official version
  model.train()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
  for i in range(num_steps):
    outputs = model(inputs[i])
    loss = outputs.mean()
    optimizer.zero_grad()
    loss.backward()
    if hvd.rank() == 0:
      view('model.outputs.%d' % i, outputs)
    optimizer.step()

  # using sync-version
  sync_model.train()
  optimizer = torch.optim.SGD(sync_model.parameters(), lr=0.1)
  optimizer = hvd.DistributedOptimizer(
      optimizer, named_parameters=sync_model.named_parameters())
  hvd.broadcast_parameters(sync_model.state_dict(), root_rank=0)

  for i in range(num_steps):
    outputs = sync_model(inputs[i, start_idx: end_idx])
    loss = outputs.mean()
    optimizer.zero_grad()
    loss.backward()
    outputs = hvd.allgather(outputs)
    if hvd.rank() == 0:
      view('sync_model.outputs.%d' % i, outputs)
    optimizer.step()

  # test inference
  if hvd.rank() == 0:
    print('[INFERENCE PHASE-1]')

  model.eval()
  with torch.no_grad():
    for i in range(num_steps):
      outputs = model(inputs[i])
      if hvd.rank() == 0:
        view('model.outputs.%d' % i, outputs)

  sync_model.eval()
  with torch.no_grad():
    for i in range(num_steps):
      outputs = sync_model(inputs[i, start_idx: end_idx])
      outputs = hvd.allgather(outputs)
      if hvd.rank() == 0:
        view('sync_model.outputs.%d' % i, outputs)

  # test inference
  if hvd.rank() == 0:
    print('[INFERENCE PHASE-2]')

  model.eval()
  with torch.no_grad():
    for i in range(num_steps):
      outputs = model(inputs[i])
      if hvd.rank() == 0:
        view('model.outputs.%d' % i, outputs)

  sync_model.eval()
  with torch.no_grad():
    for i in range(num_steps):
      outputs = sync_model(inputs[i, start_idx: end_idx])
      outputs = hvd.allgather(outputs)
      if hvd.rank() == 0:
        view('sync_model.outputs.%d' % i, outputs)

  if hvd.rank() == 0:
    # for key, value in sync_model.state_dict().items():
    #   print(key, value.shape)
    print('\n')


if __name__ == "__main__":
  hvd.init()
  test_resnet(True, False)
  test_resnet(True, True)

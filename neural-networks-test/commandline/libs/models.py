import random
import torch
import torchvision.models as models
from torch import optim as Optim
from torch import nn as Nn

# a = lambda: random.randint(0, 100)
RANDOM_CHANNELS = lambda: random.sample([7, 42, 32, 64, 128, 101, 256], 1)[0]

KERNEL_SIZE = (3, 3)
STRIDE = (1, 1)
PADDING = (1, 1)

def get_model(arch, class_count, learning_rate, hidden_units, momentum, gpu = False):
  model = None

  if gpu and not torch.cuda.is_available():
    raise Exception("GPU is not available")
  else:
    torch.device("cuda:0" if gpu else "cpu")

  if arch == "squeezenet_1_1":
    model = models.squeezenet1_1( pretrained=True)
    # Finetune Squeezenet
    model.classifier = Nn.Sequential(
        Nn.Dropout(p=0.2),
        Nn.Conv2d(512, class_count, kernel_size=1),
        Nn.ReLU(inplace=True),
        Nn.AvgPool2d(13)
    )
    model.forward = lambda x: model.classifier(model.features(x)).view(x.size(0), class_count)
  elif arch == "Densenet_161":
    model = models.densenet161(pretrained=True)
    model.classifier = Nn.Linear(in_features=2208, out_features=class_count, bias=True)
  else:
    #model = models.vgg16(pretrained=True)
    #model = models.alexnet(pretrained=True)
    raise NotImplementedError("Architecture not supported")

  if hidden_units is not None:
    model = change_hidden_units(arch, model, hidden_units)

  criterion = Nn.CrossEntropyLoss()
  optimizer = Optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

  return model, criterion, optimizer

def change_hidden_units(arch, model, hidden_units):
  features = list(model.features)
  current_units = len(features)
  if hidden_units == current_units:
    return model

  if hidden_units > current_units:
    features = __add_hidden_units(
      features, hidden_units, current_units, (512 if arch == "squeezenet_1_1" else 2208) # By Definition
    )
  elif (arch == "squeezenet_1_1" and hidden_units <= 10) \
  or (arch == "Densenet_161" and hidden_units <= 8):
    max_h = 10 if arch == "squeezenet_1_1" else 8
    raise ValueError(f"Known Bug: Hidden units are expected to be greater than {max_h}. [HELP NEEDED]")
  elif arch == "squeezenet_1_1":
    features = __squeeze_remove_units(features, current_units, hidden_units)
  else:
    features = __densenet_remove_units(features, current_units, hidden_units)

  model.features = torch.nn.Sequential(*features)
  return model

def __add_hidden_units(features, hidden_units, current_units, out_features):
  add_n_layers = hidden_units - current_units

  in_channels = out_features
  out_channels = RANDOM_CHANNELS()
  for _ in range(add_n_layers - 1):
    features.append(Nn.Conv2d(in_channels, out_channels, KERNEL_SIZE, STRIDE, PADDING))
    in_channels = out_channels
    out_channels = RANDOM_CHANNELS()

  features.append(Nn.Conv2d(in_channels, out_features, KERNEL_SIZE, STRIDE, PADDING))
  return features

def __densenet_remove_units(features, current_units, hidden_units):
    remove_n_layers = current_units - hidden_units
    in_features = 3
    out_features = [None, 96, 96, 96, 96][remove_n_layers] # By Definition
    return [Nn.Conv2d(in_features, out_features, KERNEL_SIZE, (2, 2), (1, 1))] + features[remove_n_layers:]

def __squeeze_remove_units(features, current_units, hidden_units):
    remove_n_layers = current_units - hidden_units
    in_features = 3
    out_features = [None, 65, 64, 64, 128, 128, 128, 256, 256, 256][remove_n_layers] # By Definition
    return [Nn.Conv2d(in_features, out_features, KERNEL_SIZE, (2, 2), (1, 1))] + features[remove_n_layers:]


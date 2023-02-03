import torch
import torchvision.models as models
from torch import optim as Optim
from torch import nn as Nn

def get_model(arch, class_count, learning_rate, momentum, gpu = False):
  model = None
  torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")

  if arch == "squeezenet_1_1":
    model = models.squeezenet1_0(pretrained=True)
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
    # Finetune Densenet_161
    model.classifier = Nn.Linear(in_features=2208, out_features=class_count, bias=True)
  else:
    #model = models.vgg16(pretrained=True)
    #model = models.alexnet(pretrained=True)
    raise NotImplementedError("Architecture not supported")


  criterion = Nn.CrossEntropyLoss()
  optimizer = Optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

  return model, criterion, optimizer

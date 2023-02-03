from torchvision import transforms as Transforms

data_transforms =  {
  "train": Transforms.Compose([
    Transforms.ColorJitter(),
    Transforms.RandomAffine(42),
    Transforms.RandomHorizontalFlip(),
    Transforms.RandomRotation(32),
    Transforms.RandomVerticalFlip(),
    Transforms.RandomRotation(21),
    # Transforms.RandomPerspective(),

    Transforms.Resize(256),
    Transforms.CenterCrop(224),
    Transforms.ToTensor(),
    Transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]),
  "validate": Transforms.Compose([
    Transforms.Resize(256),
    Transforms.CenterCrop(224),
    Transforms.ToTensor(),
    Transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
}

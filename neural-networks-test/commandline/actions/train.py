import torch
import time

from libs.print_steps  import print_step

def train_step(loading_str, tr_loss, images, labels, model, optimizer, criterion):
  optimizer.zero_grad()

  print(loading_str + "|", end='\r')
  output = model(images)

  print(loading_str + "/", end='\r')
  loss = criterion(output, labels)

  print(loading_str + "-", end='\r')
  loss.backward()

  print(loading_str + "\\", end='\r')
  optimizer.step()

  print(loading_str + "|", end='\r')
  tr_loss += loss.item()

  print(loading_str + "/", end='\r')
  loading_str += '⋅'

  return tr_loss, loading_str

def validation_step(dataloaders, model, criterion):
  vd_loss = 0.0
  vd_correct = 0

  with torch.no_grad():
    model.eval()
    for (images, labels) in iter(dataloaders["validate"]):
      log_ps = model(images)
      loss = criterion(log_ps, labels)
      vd_loss += loss.item()

      ps = torch.exp(log_ps)
      _, top_class = ps.topk(1, dim=1)
      equals = top_class == labels.view(*top_class.shape)
      vd_correct += equals.sum().item()

  model.train()

  return vd_loss, vd_correct

def do_train(dataloaders, epochs, model, criterion, optimizer):
  validate_loader_count = len(dataloaders['validate'].dataset)
  train_loader_count = len(dataloaders['train'].dataset)

  # Freeze parameters
  for param in model.features.parameters():
      param.requires_grad = False

  print(f"Training Count: {train_loader_count}, Validation Count: {validate_loader_count}, Epochs: {epochs}")
  print()

  time_count = 0
  start_time = time.time()

  for epoch in range(epochs):
      tr_loss = 0.0

      loading_str = "⋅"
      for (images, labels) in iter(dataloaders["train"]):
        tr_loss, loading_str = train_step(loading_str, tr_loss, images, labels, model, optimizer, criterion)
      else:
        vd_loss, vd_correct = validation_step(dataloaders, model, criterion)

        time_now = print_step(
          epoch, epochs, start_time, time_count,
          tr_loss, vd_loss, vd_correct,
          validate_loader_count, train_loader_count
        )
        start_time = time_now

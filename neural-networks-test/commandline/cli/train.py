#!/usr/bin/env python

import pathlib
import time
import torch
import json

from libs.models import get_model
from actions.train import do_train
from libs.dataloaders import get_dataloaders

BATCH_SIZE = 128

import click

@click.command()
@click.argument("data_directory", type=click.Path(exists=True, dir_okay=True, readable=True), required=1)
@click.option('-c', '--category-name',type=click.Path(exists=True, file_okay=True, readable=True), required=1)
@click.option('-a', '--arch', type=click.Choice(['squeezenet_1_1', 'Densenet_161']), default='squeezenet_1_1')
@click.option('-l', '--learning-rate', type=click.FLOAT, default=0.1)
@click.option('-m', '--momentum', type=click.FLOAT, default=0.9)
@click.option('-e', '--epochs', type=click.INT, default=3)
@click.option('-h', '--hidden_units', type=click.INT, default=None)
@click.option('-g', '--gpu', type=click.BOOL, is_flag=True, default=False)
@click.option('-s', '--save-dir', type=click.Path(exists=True, dir_okay=True, readable=True))

def train(data_directory, category_name, arch, learning_rate, momentum, hidden_units, epochs, gpu, save_dir):
  """
    Train a model on the given data directory. Currently supporting the following architectures:\n
      * squeezenet_1_1\n
      * Densenet_161\n
    Data directory must contain a `/train` and `/valid` subfolders.
  """

  if not pathlib.Path(f"{data_directory}/train").is_dir():
    raise NotADirectoryError(f"Data directory {data_directory}/train does not exist")
  if not pathlib.Path(f"{data_directory}/valid").is_dir():
    raise NotADirectoryError(f"Data directory {data_directory}/valid does not exist")

  dataloaders, image_datasets  = get_dataloaders(data_directory, BATCH_SIZE)

  with open(category_name, 'r') as f:
    cat_to_name = json.load(f)

  class_count = len(cat_to_name)

  model, criterion, optimizer = get_model(arch, class_count, learning_rate, hidden_units, momentum, gpu=gpu)
  do_train(dataloaders, epochs, model, criterion, optimizer)

  if save_dir is not None:
    chk = f"{save_dir}/model_{str(time.time()).replace('.', '-')}.pth"
    torch.save({
      'arch_name': arch,
      'class_count': class_count,
      'learning_rate': learning_rate,
      'momentum': momentum,
      'state_dict': model.state_dict(),
      'class_map': image_datasets['train'].class_to_idx,
      'epochs': epochs,
      'optmizer_state_dict': optimizer.state_dict
    }, chk)
    print(f"  Saved checkpoint: {chk}")


if __name__ == '__main__':
  train()

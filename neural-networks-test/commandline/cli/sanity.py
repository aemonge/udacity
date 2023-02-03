#!/usr/bin/env python

import pathlib
import torch
import json

from libs.models import get_model
from actions.test import do_sanity
from libs.dataloaders import get_dataloaders

BATCH_SIZE = 128

import click

@click.command()
@click.argument("data_directory", type=click.Path(exists=True, dir_okay=True, readable=True), required=1)
@click.argument("checkpoint-path", type=click.Path(exists=True, file_okay=True, readable=True), required=1)
@click.option('-c', '--category-name',type=click.Path(exists=True, file_okay=True, readable=True), required=1)
@click.option('-n', '--n_count', type=click.INT, default=3)
@click.option('-g', '--gpu', type=click.BOOL, is_flag=True, default=False)
@click.option('-t', '--tilted_title', type=click.BOOL, is_flag=True, default=False)

def sanity(data_directory, category_name, checkpoint_path, n_count, gpu, tilted_title):
  """
    Sanity check in the /test folder of the data_directory for N amount of images,
    displaying their tag, image and class prediction.
  """

  if not pathlib.Path(f"{data_directory}/test").is_dir():
    raise NotADirectoryError(f"Data directory {data_directory}/test does not exist")

  with open(category_name, 'r') as f:
    cat_to_name = json.load(f)

  data = torch.load(checkpoint_path)
  model, _, optimizer = get_model(
    data['arch_name'], data['class_count'], data['learning_rate'], data['momentum'], gpu=gpu
  )
  print("\033c", end='') # Clear the deprecation warning, about pre-trained. I'm following instructions
  print()
  model.load_state_dict(data['state_dict'])
  optimizer.state_dict = data['optmizer_state_dict']

  do_sanity(f"{data_directory}/test", n_count, cat_to_name, data['class_map'], model, tilted_title)

if __name__ == '__main__':
  sanity()

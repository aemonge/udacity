#!/usr/bin/env python

import pathlib
import torch
import json

from libs.models import get_model
from actions.test import do_test
from libs.dataloaders import get_dataloaders

BATCH_SIZE = 128

import click

@click.command()
@click.argument("data_directory", type=click.Path(exists=True, dir_okay=True, readable=True), required=1)
@click.argument("checkpoint-path", type=click.Path(exists=True, file_okay=True, readable=True), required=1)
@click.option('-g', '--gpu', type=click.BOOL, is_flag=True, default=False)

def test(data_directory, checkpoint_path, gpu):
  """
    Test a pre trained model. looking for the ./test in the data data_directory
  """

  if not pathlib.Path(f"{data_directory}/test").is_dir():
    raise NotADirectoryError(f"Data directory {data_directory}/test does not exist")

  data = torch.load(checkpoint_path)
  model, _, optimizer = get_model(
    data['arch_name'], data['class_count'], data['learning_rate'], data['momentum'], gpu=gpu
  )
  model.load_state_dict(data['state_dict'])
  optimizer.state_dict = data['optmizer_state_dict']
  dataloaders, _  = get_dataloaders(data_directory, BATCH_SIZE)

  do_test(dataloaders, model)

if __name__ == '__main__':
  test()

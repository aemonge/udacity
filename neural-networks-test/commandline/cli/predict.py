#!/usr/bin/env python

import torch
import json
from libs.models import get_model
from actions.predict import do_predict

T_BATCH_SIZE = 12

import click

@click.command()
@click.argument("path-to-image", type=click.Path(exists=True, dir_okay=True, readable=True), required=1)
@click.argument("checkpoint-path", type=click.Path(exists=True, file_okay=True, readable=True), required=1)
@click.option('-c', '--category-name',type=click.Path(exists=True, file_okay=True, readable=True), required=1)
@click.option('-t', '--top_k', type=click.INT, default=1)
@click.option('-g', '--gpu', type=click.BOOL, is_flag=True, default=False)

def predict(path_to_image, checkpoint_path, category_name, top_k, gpu):
  """
    Predict the class (or classes) of an image using a pre-trained deep learning model.
  """

  data = torch.load(checkpoint_path)
  model, _, optimizer = get_model(
    data['arch_name'], data['class_count'], data['learning_rate'], data['momentum'], gpu=gpu
  )
  print("\033c", end='') # Clear the deprecation warning, about pre-trained. I'm following instructions
  model.load_state_dict(data['state_dict'])
  optimizer.state_dict = data['optmizer_state_dict']

  with open(category_name, 'r') as f:
    cat_to_name = json.load(f)

  prediction = do_predict(path_to_image, model, cat_to_name, data['class_map'], top_k)
  print(prediction)

if __name__ == '__main__':
  predict()

import torch
import numpy as np
from matplotlib import pyplot as plt
from libs.image_tools import process_image
from PIL import Image


def print_pbs(class_map, cat_to_name, probs, idxs, ax=None):
    inv_map = { v:k for k,v in class_map.items() }
    classes = [cat_to_name[inv_map[i]] for i in idxs]

    probs = np.round(probs, 2)
    if ax is None:
        plt.subplot(1, 1)
        plt.barh(classes, probs)
    else:
        ax.barh(classes, probs)

def predict(image, model, topk=5, use_log=False):
    '''
      Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with torch.no_grad():
        model.eval()
        tensor_img = process_image(image)
        logps = model(tensor_img.unsqueeze(0)) # from 3d to 4d [ introducion a batch dimension ]
        ps = logps[0] # Return to 3D [ no batches again ]
        ps_val, ps_idx = ps.topk(topk)

        return ps_val.numpy(), ps_idx.numpy()

def do_predict(img_path, model, cat_to_name, class_map, top_k):
  inv_map = { v:k for k,v in class_map.items() }

  probs, idxs = predict(Image.open(img_path), model, topk=top_k)
  predicitons = [cat_to_name[inv_map[i]] for i in idxs]
  if top_k > 1:
    return list(zip(predicitons, probs))
  # Else
  return predicitons[0]


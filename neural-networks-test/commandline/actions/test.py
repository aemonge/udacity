import torch
from wcmatch import glob

from random import randrange
from matplotlib import pyplot as plt
from libs.image_tools import process_image, imshow
from actions.predict import predict, print_pbs
from PIL import Image

def do_test(dataloaders, model):
  # TODO: Shouldn't I be loading the dataloaders['test'] instead ?
  test_correct = 0
  test_loader_count = len(dataloaders['test'].dataset)

  with torch.no_grad():
      model.eval()
      for (images, labels) in iter(dataloaders["test"]):
          log_ps = model(images)

          ps = torch.exp(log_ps)
          _, top_class = ps.topk(1, dim=1)
          equals = top_class == labels.view(*top_class.shape)
          test_correct += equals.sum().item()
          print('.', end='')
      else:
          print('\033[F\033[K', end='\r')  # back prev line and clear
          print(f" Test Accuracy: {(test_correct * 100 / test_loader_count):.2f}%")

def do_sanity(path, count, cat_to_name, class_map, model, tilted):
  all_imgs = [
    file for file in glob.glob(f"{path}/**/*.@(jpg|jpeg|png)", flags=glob.EXTGLOB)
  ]

  fig, ax = plt.subplots(count, 2, width_ratios=[4,1])
  if tilted:
    plt.subplots_adjust(left=0.22, bottom=0.08, top=0.98, hspace=0.6, wspace=0)
  else:
    plt.subplots_adjust(left=0.22, bottom=0.08, top=0.90, hspace=0.6, wspace=0.35, right=0.86)

  for c_idx in range(count):
    ix = randrange(len(all_imgs))
    img_path = all_imgs[ix]
    real_class_id = str.split(img_path, '/')[2]
    real_class = cat_to_name[real_class_id]
    image = Image.open(img_path)

    tensor_img = process_image(image)
    probs, idxs = predict(image, model, topk=3)

    print_pbs(class_map, cat_to_name, probs, idxs, ax=ax[c_idx][0])
    imshow(tensor_img, title=real_class, ax=ax[c_idx][1], tilted=tilted)

  fig.align_ylabels(ax[:, 1])
  plt.show()

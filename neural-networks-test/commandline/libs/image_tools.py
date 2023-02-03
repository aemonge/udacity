import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms as Transforms

def imshow(image, ax=None, title=None, tilted=False):
    """Imshow for Tensor."""
    if ax is None:
        _, ax = plt.subplots()

    if title is not None:
      if tilted:
        ax.set_ylabel(title.replace(' ', '\n'), fontsize=10)
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis='y')
        ax.set_xticks([])
        ax.set_yticks([])
      else:
        ax.set_title(title)
        ax.axis("off")

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image, interpolation='nearest')

    return ax

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array

        Parameters:
        -----------
            image : PIL Image

        Returns:
        --------
            Numpy Array
    '''
    t_transforms = Transforms.Compose([
        Transforms.Resize(256),
        Transforms.CenterCrop(224),
        Transforms.ToTensor(),
        Transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = t_transforms(image)
    return image

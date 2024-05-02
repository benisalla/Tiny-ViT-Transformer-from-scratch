import math
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def disp_img(img):
    """
    Displays a single image with its dimensions as the title.
    
    Args:
        img (array): The image to display.
    """
    plt.imshow(img)
    plt.title(img.shape)
    plt.axis('off')
    plt.show()

def disp_patches(img, P, C):
    """
    Displays an image and its patches given the patch size.
    
    Args:
        img (array): The image to be patched.
        P (int): Patch size (assumed square).
        C (int): Number of channels in the image.
    """
    Hn, Wn = img.shape[0] // P, img.shape[1] // P
    patches = img.reshape(Hn, P, Wn, P, C).swapaxes(1, 2)

    fig = plt.figure(figsize=(16, 8))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 2])

    ax1 = fig.add_subplot(grid[0])
    ax1.imshow(img)
    ax1.set(title=f'Image ==> {P}x{P} patches', xticks=[], yticks=[])

    subgrid = grid[1].subgridspec(Hn, Wn, hspace=-0.5, wspace=0)
    for i in range(Hn):
        for j in range(Wn):
            ax = fig.add_subplot(subgrid[i, j])
            ax.imshow(patches[i][j], aspect='equal')
            ax.set(xticks=[], yticks=[])
    plt.show()

def imshow(img, title=None, ax=None):
    """
    Displays a single tensor image with normalization reversed.
    
    Args:
        img (Tensor): Image tensor in CxHxW format.
        title (str, optional): Title for the plot.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis object to draw the image on.
    """
    if ax is None:
        _, ax = plt.subplots()
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    if title:
        ax.set_title(title, fontdict={'fontsize': 8, 'color': 'black'})
    ax.axis('off')

def show_batch_images(dataloader, classes):
    """
    Displays a batch of images from a dataloader.
    
    Args:
        dataloader (DataLoader): The dataloader to fetch data from.
        classes (list): List of class names for labeling images.
    """
    in_batch, out_batch = next(iter(dataloader))
    batch_size = in_batch.size(0)
    n_rows, n_cols = int(math.sqrt(batch_size)), math.ceil(batch_size / int(math.sqrt(batch_size)))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, ax in enumerate(axes[:batch_size]):
        img = in_batch[i]
        title = classes[out_batch[i].item()]
        imshow(img, title=title, ax=ax)
    
    for j in range(batch_size, n_rows * n_cols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def show_one_image(dataloader, classes):
    """
    Displays a single image from a dataloader.
    
    Args:
        dataloader (DataLoader): The dataloader to fetch data from.
        classes (list): List of class names for labeling the image.
    """
    in_batch, out_batch = next(iter(dataloader))
    img = in_batch[0]
    img_class = out_batch[0].item()
    imshow(img=img, title=classes[img_class])

def filter_images_per_class(data_dir, classes, max_img_cls=None):
    """
    Filters images per class from a specified directory.
    
    Args:
        data_dir (str): The directory where images are stored.
        classes (list): List of classes to filter images for.
        max_img_cls (int, optional): Maximum number of images to fetch per class.
    
    Returns:
        list of tuples: List containing tuples of (image path, class index).
    """
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    filtered_data = []
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            images = os.listdir(class_dir)[:max_img_cls]
            filtered_data.extend((os.path.join(class_dir, img_name), class_to_idx[class_name]) for img_name in images)
    return filtered_data

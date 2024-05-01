import math
import os
from matplotlib import pyplot as plt
import numpy as np


def disp_img(img):
    plt.imshow(img)
    plt.title(img.shape)
    plt.axis('off')
    plt.show()

def disp_patches(img, P, C):
    Hn = img.shape[0] // P
    Wn = img.shape[1] // P
    patches = img.reshape(Hn, P, Wn, P, C).swapaxes(1, 2)

    fig = plt.figure(figsize=(16, 8))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 2])

    # image
    ax1 = fig.add_subplot(grid[0])
    ax1.set(title=f'Image ==> {P}x{P} patches')
    ax1.set(xticks=[], yticks=[])
    ax1.imshow(img)

    # patches
    subgrid = grid[1].subgridspec(Hn, Wn, hspace=-0.5, wspace=0)
    for i in range(Hn):
        for j in range(Wn):
            ax = fig.add_subplot(subgrid[i, j])
            ax.set(xticks=[], yticks=[])
            ax.imshow(patches[i][j], aspect='equal')


# display the image
def imshow(img, title=None, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    if title is not None:
        ax.set_title(title, fontdict={'fontsize': 8, 'color': 'black'})
    ax.axis('off')


# Display a batch of images
def show_batch_images(dataloader, classes):
    in_batch, out_batch = next(iter(dataloader))
    batch_size = in_batch.size(0)  

    n_rows = int(math.sqrt(batch_size))
    n_cols = math.ceil(batch_size / n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    else:
        axes = axes.flatten()  
    
    for i in range(batch_size):
        ax = axes[i]
        title = classes[out_batch[i].item()]
        img = in_batch[i]
        imshow(img, title=title, ax=ax)
    
    for j in range(i + 1, n_rows*n_cols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# Display a batch of images
def show_one_image(dataloader, classes):
    in_batch, out_batch = next(iter(dataloader))
    img = in_batch[0]
    img_class = out_batch[0].item()
    class_name = classes[img_class]
    imshow(img=img, title=class_name)
    

def filter_images_per_class(data_dir, classes, max_img_cls=None):
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    filtered_data = []
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            images = os.listdir(class_dir)[:max_img_cls]
            for img_name in images:
                img_path = os.path.join(class_dir, img_name)
                item = (img_path, class_to_idx[class_name])
                filtered_data.append(item)
    return filtered_data
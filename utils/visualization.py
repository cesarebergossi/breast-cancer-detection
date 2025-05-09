import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import config

def plot_example(example, image_title='Image', mask_title='Mask'):
    image = example[0].squeeze()
    mask = example[1].squeeze()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(image_title)
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(mask_title)
    axes[1].axis('off')
    
    plt.show()

def plot_prediction(example, pred_mask,image_title='Image', mask_title='Mask',pred_title = 'predicted'):

    image = example[0].squeeze()
    mask = example[1].squeeze()
    prediction = pred_mask.squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(image_title)
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(mask_title)
    axes[1].axis('off')

    axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title(pred_title)
    axes[2].axis('off')

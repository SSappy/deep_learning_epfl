"""
File containing the helper function plot_history to visualize the learning curves.
"""


import numpy as np

import matplotlib.pyplot as plt


def plot_history(history):
    """
    Plots a training history.
    :param history: Training history returned by the fit function.
    :return: Nothing.
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    epochs = np.arange(len(history['loss']))

    ax[0].plot(epochs, history['loss'], label='loss')

    if 'val_loss' in history:
        ax[0].plot(epochs, history['val_loss'], label='val_loss')
        ax[0].set_title('Loss and validation loss as a function of the number of epochs')
    else:
        ax[0].set_title('Loss as a function of the number of epochs')

    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')

    ax[0].legend()

    ax[1].plot(epochs, history['acc'], label='acc')

    if 'val_acc' in history:
        ax[1].plot(epochs, history['val_acc'], label='val_acc')
        ax[1].set_title('Accuracy and validation accuracy as a function of the number of epochs')
    else:
        ax[1].set_title('Accuracy as a function of the number of epochs')

    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')

    ax[1].legend()

    plt.tight_layout()

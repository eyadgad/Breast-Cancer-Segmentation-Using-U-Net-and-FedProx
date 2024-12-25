import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# === Loss and Metrics === #
def dice_loss(y_true, y_pred):
    """
    Computes the Dice loss for binary segmentation tasks.
    """
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
    return 1 - (2 * intersection / union)


def iou(y_true, y_pred):
    """
    Computes Intersection over Union (IoU).
    """
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / union


def sen(y_true, y_pred):
    """
    Computes Sensitivity (Recall).
    """
    true_positive = tf.reduce_sum(y_true * y_pred)
    false_negative = tf.reduce_sum(y_true * (1 - y_pred))
    return true_positive / (true_positive + false_negative)


def spe(y_true, y_pred):
    """
    Computes Specificity.
    """
    true_negative = tf.reduce_sum((1 - y_true) * (1 - y_pred))
    false_positive = tf.reduce_sum((1 - y_true) * y_pred)
    return true_negative / (true_negative + false_positive)


def f1(y_true, y_pred):
    """
    Computes F1 Score.
    """
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def load_checkpoint(CHECKPOINT_FILE):
    """
    Load the last completed round from the checkpoint file.

    Returns:
        The last completed round number.
    """
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as file:
            return int(file.read().strip())
    return -1  # No checkpoint found


def save_checkpoint(round_num, CHECKPOINT_FILE):
    """
    Save the current round number to the checkpoint file.

    Args:
        round_num: The round number to save.
    """
    with open(CHECKPOINT_FILE, "w") as file:
        file.write(str(round_num))

def plot_history(history, round_num, PLOTS_DIR, save_to_disk=True):
    """
    Plot training history and save plots to files.

    Args:
        history: Training history object.
        round_num: Current round number for naming plots.
        save_to_disk (bool): If True, saves plots to disk.
    """
    figs = []

    # Plot training & validation accuracy
    fig = plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Round {round_num} - Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    if save_to_disk:
        file_path = os.path.join(PLOTS_DIR, f'accuracy_round_{round_num}.png')
        plt.savefig(file_path)
        figs.append(file_path)
    plt.close(fig)

    # Plot training & validation loss
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Round {round_num} - Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    if save_to_disk:
        file_path = os.path.join(PLOTS_DIR, f'loss_round_{round_num}.png')
        plt.savefig(file_path)
        figs.append(file_path)
    plt.close(fig)

    # Plot custom metrics
    for metric in ['iou', 'sen', 'spe', 'f1']:
        if metric in history.history:
            fig = plt.figure()
            plt.plot(history.history[metric])
            plt.plot(history.history['val_' + metric])
            plt.title(f'Round {round_num} - Model {metric.upper()}')
            plt.ylabel(metric.upper())
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            if save_to_disk:
                file_path = os.path.join(PLOTS_DIR, f'{metric}_round_{round_num}.png')
                plt.savefig(file_path)
                figs.append(file_path)
            plt.close(fig)

    print(f"Plots saved for Round {round_num} in {PLOTS_DIR}")
    return figs

def save_training_history(history, round_num, HISTORY_FILE):
    """
    Save the training history for a given round.

    Args:
        history: Training history object.
        round_num: Current round number.
    """
    history_data = {
        'round': round_num,
        'history': history.history
    }
    if os.path.exists(HISTORY_FILE):
        previous_history = np.load(HISTORY_FILE, allow_pickle=True).tolist()
    else:
        previous_history = []
    previous_history.append(history_data)
    np.save(HISTORY_FILE, previous_history)
    print(f"Training history saved for Round {round_num}.")
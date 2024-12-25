import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from preprocess import prepare_federated_data
from train import dice_loss, iou, sen, spe, f1, GLOBAL_WEIGHTS_FILE, OUTPUT_DIR, PLOTS_DIR, dataset_paths, image_size
from tensorflow.keras.models import load_model
import os
import numpy as np


# Function to evaluate the model and save sample predictions
def evaluate_and_visualize(model, test_images, test_masks, OUTPUT_DIR=OUTPUT_DIR, PLOTS_DIR=PLOTS_DIR, threshold=0.5, accuracy_threshold=0.99):
    """
    Evaluate the model and visualize predictions with high accuracy.

    Args:
        model: Trained model to evaluate.
        test_images: Array of test images.
        test_masks: Array of ground truth masks.
        threshold (float): Threshold for binary classification.
        accuracy_threshold (float): Accuracy threshold to select high-performing predictions.
    """
    # Predict the masks
    preds = model.predict(test_images)
    preds_binary = (preds > threshold).astype(np.uint8)

    # Log evaluation results
    results_file = os.path.join(OUTPUT_DIR, "evaluation_results.txt")
    with open(results_file, "w") as res_file:
        for i in range(len(test_images)):
            acc = accuracy_score(test_masks[i].flatten(), preds_binary[i].flatten())
            res_file.write(f"Image {i + 1}: Accuracy = {acc:.4f}\n")

            # If accuracy exceeds the threshold, visualize the predictions
            if acc >= accuracy_threshold:
                print(f"High accuracy ({acc:.2f}) on sample {i + 1}")
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(test_images[i])
                ax[0].set_title('Scan')
                ax[0].axis('off')
                ax[1].imshow(test_masks[i], cmap='gray')
                ax[1].set_title('Ground Truth Mask')
                ax[1].axis('off')
                ax[2].imshow(preds[i], cmap='gray')
                ax[2].set_title('Predicted Mask')
                ax[2].axis('off')

                # Save the visualized prediction
                prediction_plot_path = os.path.join(PLOTS_DIR, f"prediction_{i + 1}.png")
                fig.savefig(prediction_plot_path)
                print(f"Saved prediction visualization for image {i + 1} in {PLOTS_DIR}")
                plt.close(fig)
    print(f"Evaluation results saved in {results_file}")


# Load or define your model
def load_trained_model(model_path, custom_objects=None):
    """
    Load a pre-trained model.

    Args:
        model_path (str): Path to the model file.
        custom_objects (dict): Custom objects used in the model.

    Returns:
        The loaded model.
    """
    
    return load_model(model_path, custom_objects=custom_objects)


# Example run of functions
if __name__ == "__main__":

    # Load the trained model (update model_path accordingly)
    
    model = load_trained_model(GLOBAL_WEIGHTS_FILE, custom_objects={
        'dice_loss': dice_loss, 'iou': iou, 'sen': sen, 'spe': spe, 'f1': f1
    })
    
    # Load the test data
    client_data, global_test_images, global_test_masks = prepare_federated_data(dataset_paths, image_size)

    # Evaluate and visualize predictions
    evaluate_and_visualize(model, global_test_images, global_test_masks)

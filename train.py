from preprocess import prepare_federated_data
from models import create_model
from federated_methods import create_client_model, train_client_model, fedprox
from utils import dice_loss, iou, sen, spe, f1, save_checkpoint, load_checkpoint, plot_history, save_training_history
import os, gc

# Define global directories and checkpoint files
OUTPUT_DIR = "./output"
LOGS_FILE = os.path.join(OUTPUT_DIR, "logs.txt")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results.txt")
GLOBAL_WEIGHTS_FILE = os.path.join(OUTPUT_DIR, "global_model_weights.h5")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.txt")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
HISTORY_FILE = os.path.join(OUTPUT_DIR, "history.npy")

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def create_client_model():
    """
    Create and compile the client model.
    """
    client_model = create_model()
    client_model.compile(
        loss=dice_loss, 
        optimizer='adam', 
        metrics=[iou, sen, spe, f1, 'accuracy']
    )
    return client_model

def run(num_rounds, client_data, global_test_imgs, global_test_masks, dice_loss_fn, iou_fn, sen_fn, spe_fn, f1_fn):
    """
    Federated Learning process with FedProx.

    Args:
        num_rounds: Total number of training rounds.
        client_data: Data for each client.
        global_test_imgs: Global test images.
        global_test_masks: Global test masks.
        dice_loss_fn: Dice loss function.
        iou_fn: IoU metric.
        sen_fn: Sensitivity metric.
        spe_fn: Specificity metric.
        f1_fn: F1 Score metric.
    """
    last_round = load_checkpoint(CHECKPOINT_FILE) + 1  # Start from the next round
    print(f"Resuming from round {last_round}...")

    for round_num in range(last_round, num_rounds):
        print(f"\n--- Round {round_num} ---")
        global_model = create_model()
        client_models = [create_client_model() for _ in range(3)]
        global_model.compile(
            loss=dice_loss_fn, 
            optimizer='adam', 
            metrics=[iou_fn, sen_fn, spe_fn, f1_fn, 'accuracy']
        )

        # Load saved global model weights if continuing
        if round_num > 0 and os.path.exists(GLOBAL_WEIGHTS_FILE):
            global_model.load_weights(GLOBAL_WEIGHTS_FILE)
            for client_model in client_models:
                client_model.set_weights(global_model.get_weights())

        # Open log and result files
        with open(LOGS_FILE, "a") as logs_file, open(RESULTS_FILE, "a") as res_file:
            logs_file.write(f"\nRound {round_num}:\n")
            res_file.write(f"\nRound {round_num}:\n")

            # Train each client model
            client_weights = []
            for client_num in range(3):
                weights = train_client_model(
                    client_models[client_num], 
                    client_data[client_num], 
                    client_num, 
                    res_file, 
                    logs_file
                )
                client_weights.append(weights)

            # Update global model using FedProx
            global_model = fedprox(global_model, client_weights, client_models, num_clients=3)

            # Evaluate global model on the global test set
            global_metrics = global_model.evaluate(global_test_imgs, global_test_masks, verbose=0)
            res_file.write(f"Server: {global_metrics}\n")
            print(f"Server  : Dice Loss: {global_metrics[0]:.2f}, IoU: {global_metrics[1]:.2f}, "
                  f"Sensitivity: {global_metrics[2]:.2f}, Specificity: {global_metrics[3]:.2f}, "
                  f"F1 Score: {global_metrics[4]:.2f}, Accuracy: {global_metrics[5]:.2f}")

            # Save global model weights and checkpoint
            global_model.save_weights(GLOBAL_WEIGHTS_FILE)
            save_checkpoint(round_num, CHECKPOINT_FILE)

            # Save training history and plots
            save_training_history(global_model.history, round_num, HISTORY_FILE)
            plot_history(global_model.history, round_num, PLOTS_DIR)


        # Clean up for the next round
        del global_model, client_models, client_weights
        gc.collect()

if __name__ == "__main__":
    # Load and preprocess data
    dataset_paths = [
        {"path": "BUS1"},  # Structured dataset
        "BUS2"  # Unstructured dataset
    ]
    image_size = (256, 256)

    # Prepare federated data
    client_data, global_test_images, global_test_masks = prepare_federated_data(dataset_paths, image_size)

    # Check the summary
    for i, client in enumerate(client_data):
        print(f"Client {i + 1}:")
        print(f"  Train images: {client[0].shape}, Train masks: {client[1].shape}")
        print(f"  Test images: {client[2].shape}, Test masks: {client[3].shape}")

    print("Server:")
    print(f"  Test images: {global_test_images.shape}, Test masks: {global_test_masks.shape}")

    # Run the Federated Learning process
    run(5, client_data, global_test_images, global_test_masks, dice_loss, iou, sen, spe, f1)

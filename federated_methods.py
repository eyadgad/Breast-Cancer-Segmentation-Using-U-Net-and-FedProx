
def train_client_model(client_model, client_data, client_num, res_file, logs_file):
    """
    Train a single client model and evaluate it.

    Args:
        client_model: The client model to train.
        client_data: Tuple containing training and testing data for the client.
        client_num: The client index.
        res_file: File object to log results.
        logs_file: File object to log training history.

    Returns:
        Client model weights after training.
    """
    history = client_model.fit(client_data[0], client_data[1], epochs=10, batch_size=16, verbose=0)
    metrics = client_model.evaluate(client_data[2], client_data[3], verbose=0)
    res_file.write(f"Client {client_num + 1}: {metrics}\n")
    logs_file.write(f"Client {client_num + 1} History: {history.history}\n")
    print(f"Client {client_num + 1}: Dice Loss: {metrics[0]:.2f}, IoU: {metrics[1]:.2f}, "
          f"Sensitivity: {metrics[2]:.2f}, Specificity: {metrics[3]:.2f}, F1 Score: {metrics[4]:.2f}, "
          f"Accuracy: {metrics[5]:.2f}")
    return client_model.get_weights()


def fedprox(global_model, client_weights, client_models, num_clients, mu=0.01):
    """
    Apply FedProx algorithm to aggregate weights from clients.

    Args:
        global_model: The global model.
        client_weights: List of weights from client models.
        client_models: List of client models.
        num_clients: Number of clients.
        mu: FedProx regularization term.

    Returns:
        Updated global model.
    """
    global_weights = global_model.get_weights()
    for i in range(len(global_weights)):
        update = 0
        for j in range(num_clients):
            update += client_weights[j][i]
            update += mu * (client_models[j].get_weights()[i] - global_weights[i])
        global_weights[i] = update / num_clients
    global_model.set_weights(global_weights)
    for client_model in client_models:
        client_model.set_weights(global_weights)
    return global_model
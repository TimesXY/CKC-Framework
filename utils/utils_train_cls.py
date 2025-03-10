import os
import math
import torch
import numpy as np
import torch.nn as nn
import sklearn.metrics as metrics

from .utils_similarity import similarity_frame
from .utils_loss import SupConLossWithMemoryBank


def save_model_if_best(model_cls, current_accuracy, best_accuracy):
    """
    Save the model to the "save_weights" folder in the current path
    if the current accuracy is greater than or equal to the best recorded accuracy.

    Parameters:
        model_cls (torch.nn.Module): The model instance to be saved.
        current_accuracy (float): The current accuracy of the model on the validation set.
        best_accuracy (float): The previously recorded best validation accuracy.

    Returns:
        float: The best accuracy (does not affect the save decision).
    """

    # Get the main program's running directory
    save_dir = os.path.join(os.getcwd(), "save_weights")

    # Ensure the target directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the model if the current accuracy is greater than or equal to the best recorded accuracy
    if current_accuracy >= best_accuracy:
        best_accuracy = current_accuracy
        model_filename = f"best_model.pth"
        save_path = os.path.join(save_dir, model_filename)

        # Save the model
        torch.save(model_cls.state_dict(), save_path)
        print(f"Current accuracy: {current_accuracy:.4f}, model saved to {save_path}")

    return best_accuracy


def calculate_metrics(labels, predictions):
    """
    Compute various classification metrics: accuracy, recall, F1 score, and precision.

    Parameters:
        labels (list or array): Ground truth labels.
        predictions (list or array): Model-predicted labels.

    Returns:
        dict: A dictionary containing accuracy, recall, F1 score, and precision.
    """
    accuracy = metrics.accuracy_score(labels, predictions)
    recall = metrics.recall_score(labels, predictions)
    f1 = metrics.f1_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions)
    return {'accuracy': accuracy, 'recall': recall, 'f1': f1, 'precision': precision}


def validate_model(model, valid_loader, device):
    """
    Evaluate the model's performance on the validation set.

    Parameters:
        model (torch.nn.Module): The model to be evaluated.
        valid_loader (DataLoader): The validation dataset loader.
        device (str): Device type, e.g., 'cuda' or 'cpu'.

    Returns:
        tuple: (Validation accuracy, Average loss)
    """
    model.eval()  # Set model to evaluation mode

    total_loss = 0.0  # Accumulated loss
    correct_cls = 0  # Number of correctly classified samples
    total_cls = 0  # Total number of samples

    all_predictions = []  # Store all predicted results
    all_labels = []  # Store all ground truth labels

    # Define loss functions
    criterion_con_loss = SupConLossWithMemoryBank(memory_size=128)
    criterion_classifier = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    attribute_correct_counts = None  # Number of correctly predicted samples for each attribute
    attribute_total_counts = 0  # Total number of attribute samples

    # Disable gradient calculations to speed up inference
    with torch.no_grad():
        for i, (videos, labels, key_index, attributes, gauss_dis, _) in enumerate(valid_loader):
            # Move data to the specified device
            videos = videos.to(device)
            labels = labels.to(device)
            gauss_dis = gauss_dis.to(device)
            key_index = key_index.to(device)
            attributes = attributes.to(device)

            # Forward pass through the model to get predictions
            predict_label, time_attention, clinical_output, feature_key, feature_video = model(videos, key_index)

            # Compute similarity matrix
            similarity = (similarity_frame(videos, key_index) + gauss_dis) / 2.0

            # Compute classification loss
            classification_loss = criterion_classifier(predict_label, labels)

            # Compute attribute loss
            attributes_loss = 0.0
            batch_size = attributes.size(0)
            num_attributes = clinical_output.size(2)  # Number of attributes

            # Initialize attribute accuracy counters
            if attribute_correct_counts is None:
                attribute_correct_counts = np.zeros(num_attributes)

            # Iterate over each attribute to compute loss and accuracy
            for attr_idx in range(num_attributes):
                attribute_output = clinical_output[:, :, attr_idx]  # Output for the current attribute
                attribute_target = attributes[:, attr_idx].long()  # Ground truth for the current attribute

                # Compute loss for the current attribute
                loss = criterion_classifier(attribute_output, attribute_target)
                attributes_loss += loss

                # Compute accuracy for the current attribute
                _, predicted = torch.max(attribute_output, 1)
                correct = (predicted == attribute_target).sum().item()
                attribute_correct_counts[attr_idx] += correct

            # Average attribute loss
            attributes_loss /= num_attributes

            # Compute time attention loss
            time_attention_loss = mse_loss(time_attention, similarity)

            # Compute supervised contrastive loss
            sup_con_loss = criterion_con_loss(feature_key, feature_video, labels)

            # Compute total loss
            combined_loss = classification_loss + attributes_loss + 2.5 * time_attention_loss + 0.5 * sup_con_loss
            total_loss += combined_loss.item()

            # Record predictions and ground truth labels
            predictions = predict_label.argmax(dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

            attribute_total_counts += batch_size

            # Update correctly classified sample count
            correct_cls += np.sum(predictions == labels.cpu().numpy())
            total_cls += len(labels)

    # Compute overall accuracy and average loss
    accuracy = correct_cls / total_cls
    avg_loss = total_loss / len(valid_loader)

    # Compute other metrics
    metrics_result = calculate_metrics(all_labels, all_predictions)

    # Compute accuracy for each attribute
    per_attribute_accuracy = attribute_correct_counts / attribute_total_counts
    average_attribute_accuracy = per_attribute_accuracy.mean()

    # Print validation results
    print(f"Validation Set - Total Loss: {avg_loss:.4f}, "
          f"Accuracy: {metrics_result['accuracy'] * 100:.2f}%, "
          f"Recall: {metrics_result['recall']:.4f}, "
          f"F1 Score: {metrics_result['f1']:.4f}, "
          f"Precision: {metrics_result['precision']:.4f}, "
          f"Average Attribute Accuracy: {average_attribute_accuracy * 100:.2f}%")

    # Print accuracy for each attribute
    for idx, acc in enumerate(per_attribute_accuracy):
        print(f"Attribute {idx + 1} Accuracy: {acc * 100:.2f}%")

    return accuracy, avg_loss


def train(model, train_loader, valid_loader, optimizer, num_epochs, device='cuda'):
    """
    Train the model and evaluate its performance on the validation set during training.

    Parameters:
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): The training dataset loader.
        valid_loader (DataLoader): The validation dataset loader.
        optimizer (torch.optim.Optimizer): The optimizer.
        num_epochs (int): Number of training epochs.
        device: Device type, e.g., 'cuda' or 'cpu'.

    Returns:
        dict: Contains training and validation loss, accuracy, and other relevant information.
    """
    # Lists to record loss and accuracy for each epoch
    all_train_accuracy = []
    all_valid_accuracy = []
    all_train_loss = []
    all_valid_loss = []
    all_per_attribute_accuracies = []
    all_average_attribute_accuracies = []

    os.makedirs("./save_weights", exist_ok=True)  # Create directory to save models

    # Initialize loss functions
    criterion_con_loss = SupConLossWithMemoryBank(memory_size=128)
    criterion_classifier = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    def lr_lambda(current_step):
        warmup_steps = 2 * len(train_loader)  # Warm-up steps
        restart_interval = 10000  # Initial cycle length for cosine annealing, increasing over time
        T_mult = 2  # Factor by which the restart interval increases after each restart

        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Determine the number of restarts that have occurred
            restart_count = math.floor(math.log(1 + (current_step - warmup_steps) / restart_interval, T_mult))

            # Compute the current cycle length
            T_current = restart_interval * (T_mult ** restart_count)

            # Compute the current position within the cycle
            current_step_in_restart = (current_step - warmup_steps) - (T_current * (T_mult ** restart_count - 1)) // (
                    T_mult - 1)

            # Apply cosine annealing schedule
            progress = current_step_in_restart / T_current
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_accuracy = 0.0  # Initialize the best accuracy

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0.0  # Accumulate total loss
        all_predictions = []  # Store all predictions
        all_labels = []  # Store all ground truth labels
        attribute_correct_counts = None  # Correctly predicted counts per attribute
        attribute_total_counts = 0  # Total attribute samples

        for i, (videos, labels, key_index, attributes, gauss_dis, _) in enumerate(train_loader):
            # Move data to the specified device
            videos = videos.to(device)
            labels = labels.to(device)
            gauss_dis = gauss_dis.to(device)
            key_index = key_index.to(device)
            attributes = attributes.to(device)

            optimizer.zero_grad()  # Clear gradients

            # Forward pass
            predict_label, time_attention, clinical_output, feature_key, feature_video = model(videos, key_index)

            # Compute similarity
            similarity = (similarity_frame(videos, key_index) + gauss_dis) / 2.0

            # Compute classification loss
            classification_loss = criterion_classifier(predict_label, labels)

            # Compute attribute loss
            attributes_loss = 0.0
            batch_size = attributes.size(0)
            num_attributes = clinical_output.size(2)

            if attribute_correct_counts is None:
                attribute_correct_counts = np.zeros(num_attributes)

            # Iterate over each attribute to compute loss and accuracy
            for attr_idx in range(num_attributes):
                attribute_output = clinical_output[:, :, attr_idx]  # Output for the current attribute
                attribute_target = attributes[:, attr_idx].long()  # Ground truth for the current attribute

                # Compute loss for the current attribute
                loss = criterion_classifier(attribute_output, attribute_target)
                attributes_loss += loss

                # Compute accuracy for the current attribute
                _, predicted = torch.max(attribute_output, 1)
                correct = (predicted == attribute_target).sum().item()
                attribute_correct_counts[attr_idx] += correct

            # Compute average attribute loss
            attributes_loss /= num_attributes

            # Compute time attention loss
            time_attention_loss = mse_loss(time_attention, similarity)

            # Compute supervised contrastive loss
            sup_con_loss = criterion_con_loss(feature_key, feature_video, labels)

            # Compute total loss
            combined_loss = classification_loss + attributes_loss + 2.5 * time_attention_loss + 0.5 * sup_con_loss
            combined_loss.backward()  # Backpropagation

            optimizer.step()  # Update parameters
            scheduler.step()  # Update learning rate

            # Record loss and predictions
            total_loss += combined_loss.item()
            predictions = predict_label.argmax(dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

            attribute_total_counts += batch_size

        avg_train_loss = total_loss / len(train_loader)
        metrics_result = calculate_metrics(all_labels, all_predictions)

        # Compute accuracy for each attribute and the average attribute accuracy
        per_attribute_accuracy = attribute_correct_counts / attribute_total_counts
        average_attribute_accuracy = per_attribute_accuracy.mean()
        all_per_attribute_accuracies.append(per_attribute_accuracy)
        all_average_attribute_accuracies.append(average_attribute_accuracy)

        # Print training results
        print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}, "
              f"Accuracy: {metrics_result['accuracy'] * 100:.2f}%, "
              f"Recall: {metrics_result['recall']:.4f}, "
              f"F1 Score: {metrics_result['f1']:.4f}, "
              f"Precision: {metrics_result['precision']:.4f}, "
              f"Average Attribute Accuracy: {average_attribute_accuracy * 100:.2f}%")

        # Print accuracy for each attribute
        for idx, acc in enumerate(per_attribute_accuracy):
            print(f"Attribute {idx + 1} Accuracy: {acc * 100:.2f}%")

        # Evaluate the model on the validation set
        valid_accuracy, avg_valid_loss = validate_model(model, valid_loader, device)
        best_accuracy = save_model_if_best(model, valid_accuracy, best_accuracy)

        # Record training and validation accuracy and loss
        all_train_accuracy.append(metrics_result['accuracy'])
        all_train_loss.append(avg_train_loss)
        all_valid_accuracy.append(valid_accuracy)
        all_valid_loss.append(avg_valid_loss)

    # Return relevant results from the training and validation process
    return {
        'model': model,
        'train_accuracy': all_train_accuracy,
        'train_loss': all_train_loss,
        'valid_accuracy': all_valid_accuracy,
        'valid_loss': all_valid_loss,
        'per_attribute_accuracies': all_per_attribute_accuracies,
        'average_attribute_accuracies': all_average_attribute_accuracies
    }

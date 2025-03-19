import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from .utils_similarity import similarity_frame
from .utils_loss import SupConLossWithMemoryBank
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, recall_score, f1_score, precision_score


def calculate_metrics(labels, predictions, average='binary'):
    """
    Calculate and return several evaluation metrics.
    Parameters:
        labels (list or np.array): True labels.
        predictions (list or np.array): Predicted labels.
        average (str): Averaging method for multiclass classification, default is 'binary'.
    Returns:
        dict: A dictionary containing 'accuracy', 'recall', 'f1_score', 'precision'.
    """
    metrics_result = {
        'accuracy': accuracy_score(labels, predictions),
        'recall': recall_score(labels, predictions, average=average),
        'f1_score': f1_score(labels, predictions, average=average),
        'precision': precision_score(labels, predictions, average=average)}
    return metrics_result


def validate_model(model, valid_loader, device):
    """
    Evaluate the model's performance on the validation set and generate an evaluation report and visualizations.
    Parameters:
        model (torch.nn.Module): The model to be evaluated.
        valid_loader (DataLoader): Data loader for the validation set.
        device (torch.device): Device type, e.g., 'cuda' or 'cpu'.
    Returns:
        tuple: (accuracy on the validation set, average loss)
    """
    model.eval()  # Set the model to evaluation mode

    total_loss = 0.0  # Accumulated loss
    all_predictions = []  # Store all predictions
    all_labels = []  # Store all true labels
    all_probabilities = []  # Store all prediction probabilities
    all_file_names = []  # Store all file names (if needed)

    # Define loss functions
    criterion_con_loss = SupConLossWithMemoryBank(memory_size=128)
    criterion_classifier = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    attribute_correct_counts = None  # Number of correctly predicted samples for each attribute
    attribute_total_counts = 0  # Total number of attribute samples

    # Ensure the directory for saving evaluation results exists
    save_images_dir = './save_images'
    os.makedirs(save_images_dir, exist_ok=True)

    # Disable gradient calculation to speed up inference
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            # Unpack the batch data
            videos, labels, key_index, attributes, gauss_dis, file_name = batch

            # Move data to the specified device
            videos = videos.to(device)
            labels = labels.to(device)
            gauss_dis = gauss_dis.to(device)
            key_index = key_index.to(device)
            attributes = attributes.to(device)

            # Forward pass through the model to get outputs
            predict_label, time_attention, clinical_output, feature_key, feature_video = model(videos, key_index)

            # Calculate similarity matrix
            similarity = (similarity_frame(videos, key_index) + gauss_dis) / 2.0

            # Calculate classification loss
            classification_loss = criterion_classifier(predict_label, labels)

            # Calculate attribute loss
            attributes_loss = 0.0
            batch_size = attributes.size(0)
            num_attributes = clinical_output.size(2)  # Number of attributes

            # Initialize counter for correctly predicted attributes
            if attribute_correct_counts is None:
                attribute_correct_counts = np.zeros(num_attributes)

            # Iterate over each attribute to calculate loss and accuracy
            for attr_idx in range(num_attributes):
                attribute_output = clinical_output[:, :, attr_idx]  # Output for the current attribute
                attribute_target = attributes[:, attr_idx].long()  # True labels for the current attribute

                # Calculate loss for the current attribute
                loss = criterion_classifier(attribute_output, attribute_target)
                attributes_loss += loss.item()

                # Calculate prediction accuracy for the current attribute
                _, predicted = torch.max(attribute_output, 1)
                correct = (predicted == attribute_target).sum().item()
                attribute_correct_counts[attr_idx] += correct

            # Average attribute loss
            attributes_loss /= num_attributes

            # Calculate time attention loss
            time_attention_loss = mse_loss(time_attention, similarity).item()

            # Calculate supervised contrastive loss
            sup_con_loss = criterion_con_loss(feature_key, feature_video, labels).item()

            # Calculate total loss
            combined_loss = (classification_loss.item() + attributes_loss + 2.5 *
                             time_attention_loss + 0.5 * sup_con_loss)
            total_loss += combined_loss

            # Record predictions and true labels
            softmax = nn.Softmax(dim=1)
            probabilities = softmax(predict_label).cpu().numpy()
            predictions = predict_label.argmax(dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities)
            all_file_names.extend(file_name)
            attribute_total_counts += batch_size

    # Calculate average loss
    avg_loss = total_loss / len(valid_loader)

    # Calculate other metrics (returns a dictionary containing 'accuracy', 'recall', 'f1_score', 'precision')
    metrics_result = calculate_metrics(all_labels, all_predictions)

    # Calculate accuracy for each attribute
    per_attribute_accuracy = attribute_correct_counts / attribute_total_counts
    average_attribute_accuracy = per_attribute_accuracy.mean()

    # Output validation results
    print(f"Validation Set - Total Loss: {avg_loss:.4f}, "
          f"Accuracy: {metrics_result['accuracy'] * 100:.2f}%, "
          f"Recall: {metrics_result['recall']:.4f}, "
          f"F1 Score: {metrics_result['f1_score']:.4f}, "
          f"Precision: {metrics_result['precision']:.4f}, "
          f"Average Attribute Accuracy: {average_attribute_accuracy * 100:.2f}%")

    # Output accuracy for each attribute
    for idx, acc in enumerate(per_attribute_accuracy):
        print(f"Attribute {idx + 1} Accuracy: {acc * 100:.2f}%")

    # Generate ROC curve and save
    labels_np = np.array(all_labels)
    probabilities_np = np.array(all_probabilities)
    if probabilities_np.shape[1] < 2:
        raise ValueError("Drawing the ROC curve requires probability outputs for at least two classes.")

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels_np, probabilities_np[:, 1])
    roc_auc = auc(fpr, tpr)

    # Create a DataFrame for ROC curve data
    roc_data = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr,
        'Thresholds': thresholds
    })
    roc_data['AUC'] = roc_auc  # Add AUC as a constant column

    # Save ROC curve data to CSV
    roc_csv_path = os.path.join(save_images_dir, 'CKC_ROC_validation.csv')
    roc_data.to_csv(roc_csv_path, index=False)
    print(f"ROC curve data has been saved to {roc_csv_path}")

    # Plot ROC curve
    plt.figure(figsize=(8, 6))  # Increase figure size for better readability
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # Set font to Times New Roman
    plt.rcParams['axes.unicode_minus'] = False  # Resolve issue with negative sign display

    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})', color='blue', linestyle='-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)  # Diagonal line

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)  # Add grid for better readability
    plt.tight_layout()  # Adjust layout

    # Save ROC curve plot
    roc_save_path = os.path.join(save_images_dir, 'CKC_ROC_validation.jpg')
    plt.savefig(roc_save_path, dpi=600)
    plt.close()
    print(f"ROC curve has been saved to {roc_save_path}")

    # Print AUC value
    print(f"AUC Value: {roc_auc:.4f}")

    # Generate confusion matrix and save
    cf_matrix = confusion_matrix(labels_np, np.array(all_predictions))
    plt.figure(figsize=(8, 6))  # Increase figure size for better readability
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # Set font to Times New Roman
    plt.rcParams['axes.unicode_minus'] = False  # Resolve issue with negative sign display

    ax = sns.heatmap(cf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
    ax.set_title("Confusion Matrix", fontsize=16)
    ax.set_xlabel("Prediction Labels", fontsize=14)
    ax.set_ylabel("True Labels", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()  # Adjust layout

    # Save confusion matrix plot
    confusion_save_path = os.path.join(save_images_dir, 'CKC_ConfusionMatrix_validation.jpg')
    plt.savefig(confusion_save_path, dpi=600)
    plt.close()
    print(f"Confusion matrix has been saved to {confusion_save_path}")

    # Save prediction results to CSV
    predictions_csv = pd.DataFrame({
        'file_name': all_file_names,
        'real_label': all_labels,
        'predict_label': all_predictions,
        'prob_class_0': probabilities_np[:, 0],
        'prob_class_1': probabilities_np[:, 1]
    })
    predictions_csv_path = os.path.join(save_images_dir, 'predictions_validation.csv')
    predictions_csv.to_csv(predictions_csv_path, index=False)
    print(f"Prediction results have been saved to {predictions_csv_path}")

    return avg_loss

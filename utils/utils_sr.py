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
    计算并返回多个评估指标。
    参数:
        labels (list or np.array): 真实标签。
        predictions (list or np.array): 预测标签。
        average (str): 用于多分类的平均方法，默认为 'binary'。
    返回:
        dict: 包含 'accuracy', 'recall', 'f1_score', 'precision' 的字典。
    """
    metrics_result = {
        'accuracy': accuracy_score(labels, predictions),
        'recall': recall_score(labels, predictions, average=average),
        'f1_score': f1_score(labels, predictions, average=average),
        'precision': precision_score(labels, predictions, average=average)}
    return metrics_result


def validate_model(model, valid_loader, device):
    """
    在验证集上评估模型的性能，并生成评估报告和可视化图表。
    参数：
        model (torch.nn.Module): 需要评估的模型。
        valid_loader (DataLoader): 验证集的数据加载器。
        device (torch.device): 设备类型，例如 'cuda' 或 'cpu'。
    返回：
        tuple: (验证集上的准确率, 平均损失)
    """
    model.eval()  # 设置模型为评估模式

    total_loss = 0.0  # 累计损失
    all_predictions = []  # 存储所有的预测结果
    all_labels = []  # 存储所有的真实标签
    all_probabilities = []  # 存储所有的预测概率
    all_file_names = []  # 存储所有的文件名（如果需要）

    # 定义损失函数
    criterion_con_loss = SupConLossWithMemoryBank(memory_size=128)
    criterion_classifier = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    attribute_correct_counts = None  # 每个属性正确预测的样本数
    attribute_total_counts = 0  # 属性样本总数

    # 确保保存评估结果的目录存在
    save_images_dir = './save_images'
    os.makedirs(save_images_dir, exist_ok=True)

    # 关闭梯度计算，加速推理
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            # 解包批次数据
            videos, labels, key_index, attributes, gauss_dis, file_name = batch

            # 将数据移动到指定设备
            videos = videos.to(device)
            labels = labels.to(device)
            gauss_dis = gauss_dis.to(device)
            key_index = key_index.to(device)
            attributes = attributes.to(device)

            # 模型前向传播，获取输出
            predict_label, time_attention, clinical_output, feature_key, feature_video = model(videos, key_index)

            # 计算相似度矩阵
            similarity = (similarity_frame(videos, key_index) + gauss_dis) / 2.0

            # 计算分类损失
            classification_loss = criterion_classifier(predict_label, labels)

            # 计算属性损失
            attributes_loss = 0.0
            batch_size = attributes.size(0)
            num_attributes = clinical_output.size(2)  # 属性数量

            # 初始化属性正确预测计数器
            if attribute_correct_counts is None:
                attribute_correct_counts = np.zeros(num_attributes)

            # 遍历每个属性，计算损失和准确率
            for attr_idx in range(num_attributes):
                attribute_output = clinical_output[:, :, attr_idx]  # 当前属性的输出
                attribute_target = attributes[:, attr_idx].long()  # 当前属性的真实标签

                # 计算当前属性的损失
                loss = criterion_classifier(attribute_output, attribute_target)
                attributes_loss += loss.item()

                # 计算当前属性的预测准确率
                _, predicted = torch.max(attribute_output, 1)
                correct = (predicted == attribute_target).sum().item()
                attribute_correct_counts[attr_idx] += correct

            # 平均属性损失
            attributes_loss /= num_attributes

            # 计算时间注意力损失
            time_attention_loss = mse_loss(time_attention, similarity).item()

            # 计算监督对比损失
            sup_con_loss = criterion_con_loss(feature_key, feature_video, labels).item()

            # 计算总损失
            combined_loss = (classification_loss.item() + attributes_loss + 2.5 *
                             time_attention_loss + 0.5 * sup_con_loss)
            total_loss += combined_loss

            # 记录预测结果和真实标签
            softmax = nn.Softmax(dim=1)
            probabilities = softmax(predict_label).cpu().numpy()
            predictions = predict_label.argmax(dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities)
            all_file_names.extend(file_name)
            attribute_total_counts += batch_size

    # 计算平均损失
    avg_loss = total_loss / len(valid_loader)

    # 计算其他指标（返回一个包含 'accuracy', 'recall', 'f1_score', 'precision' 的字典）
    metrics_result = calculate_metrics(all_labels, all_predictions)

    # 计算每个属性的准确率
    per_attribute_accuracy = attribute_correct_counts / attribute_total_counts
    average_attribute_accuracy = per_attribute_accuracy.mean()

    # 输出验证结果
    print(f"验证集 - 总损失: {avg_loss:.4f}, "
          f"准确率: {metrics_result['accuracy'] * 100:.2f}%, "
          f"召回率: {metrics_result['recall']:.4f}, "
          f"F1 分数: {metrics_result['f1_score']:.4f}, "
          f"精度: {metrics_result['precision']:.4f}, "
          f"属性平均准确率: {average_attribute_accuracy * 100:.2f}%")

    # 输出每个属性的准确率
    for idx, acc in enumerate(per_attribute_accuracy):
        print(f"属性 {idx + 1} 准确率: {acc * 100:.2f}%")

    # 生成 ROC 曲线并保存
    labels_np = np.array(all_labels)
    probabilities_np = np.array(all_probabilities)
    if probabilities_np.shape[1] < 2:
        raise ValueError("ROC 曲线绘制需要至少两个类别的概率输出。")

    # 计算 ROC 曲线和 AUC
    fpr, tpr, thresholds = roc_curve(labels_np, probabilities_np[:, 1])
    roc_auc = auc(fpr, tpr)

    # 创建 ROC 曲线数据的 DataFrame
    roc_data = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr,
        'Thresholds': thresholds
    })
    roc_data['AUC'] = roc_auc  # 添加 AUC 作为常数列

    # 保存 ROC 曲线数据到 CSV
    roc_csv_path = os.path.join(save_images_dir, 'CKC_ROC_validation.csv')
    roc_data.to_csv(roc_csv_path, index=False)
    print(f"ROC 曲线数据已保存至 {roc_csv_path}")

    # 绘制 ROC 曲线
    plt.figure(figsize=(8, 6))  # 增加图形尺寸以提高可读性
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置新罗马字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})', color='blue', linestyle='-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)  # 对角线

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格以提高可读性
    plt.tight_layout()  # 调整布局

    # 保存 ROC 曲线图
    roc_save_path = os.path.join(save_images_dir, 'CKC_ROC_validation.jpg')
    plt.savefig(roc_save_path, dpi=600)
    plt.close()
    print(f"ROC 曲线已保存至 {roc_save_path}")

    # 打印 AUC 值
    print(f"AUC 值: {roc_auc:.4f}")

    # 生成混淆矩阵并保存
    cf_matrix = confusion_matrix(labels_np, np.array(all_predictions))
    plt.figure(figsize=(8, 6))  # 增加图形尺寸以提高可读性
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置新罗马字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    ax = sns.heatmap(cf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
    ax.set_title("Confusion Matrix", fontsize=16)
    ax.set_xlabel("Prediction Labels", fontsize=14)
    ax.set_ylabel("True Labels", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()  # 调整布局

    # 保存混淆矩阵图
    confusion_save_path = os.path.join(save_images_dir, 'CKC_ConfusionMatrix_validation.jpg')
    plt.savefig(confusion_save_path, dpi=600)
    plt.close()
    print(f"混淆矩阵已保存至 {confusion_save_path}")

    # 保存预测结果到 CSV
    predictions_csv = pd.DataFrame({
        'file_name': all_file_names,
        'real_label': all_labels,
        'predict_label': all_predictions,
        'prob_class_0': probabilities_np[:, 0],
        'prob_class_1': probabilities_np[:, 1]
    })
    predictions_csv_path = os.path.join(save_images_dir, 'predictions_validation.csv')
    predictions_csv.to_csv(predictions_csv_path, index=False)
    print(f"预测结果已保存至 {predictions_csv_path}")

    return avg_loss

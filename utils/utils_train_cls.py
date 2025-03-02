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
    如果当前准确率大于或等于最佳值，则保存模型到当前路径下的 save_weights 文件夹中。

    参数：
        model_cls (torch.nn.Module): 要保存的模型实例。
        current_accuracy (float): 当前模型在验证集上的准确率。
        best_accuracy (float): 之前最佳的验证集准确率
    返回：
        float: 更新后的最佳准确率（不会影响保存决定）。
    """

    # 获取主程序运行路径
    save_dir = os.path.join(os.getcwd(), "save_weights")

    # 确保目标目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 如果当前准确率大于或等于最佳值，则保存模型
    if current_accuracy >= best_accuracy:
        best_accuracy = current_accuracy
        model_filename = f"best_model.pth"
        save_path = os.path.join(save_dir, model_filename)

        # 保存模型
        torch.save(model_cls.state_dict(), save_path)
        print(f"当前准确率为 {current_accuracy:.4f}，已保存到 {save_path}")

    return best_accuracy


def calculate_metrics(labels, predictions):
    """
    计算分类任务的各项指标：准确率、召回率、F1 分数和精度。

    参数：
        labels (list 或 array): 真实标签。
        predictions (list 或 array): 模型预测的标签。

    返回：
        dict: 包含准确率、召回率、F1 分数和精度的字典。
    """
    accuracy = metrics.accuracy_score(labels, predictions)
    recall = metrics.recall_score(labels, predictions)
    f1 = metrics.f1_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions)
    return {'accuracy': accuracy, 'recall': recall, 'f1': f1, 'precision': precision}


def validate_model(model, valid_loader, device):
    """
    在验证集上评估模型的性能。

    参数：
        model (torch.nn.Module): 需要评估的模型。
        valid_loader (DataLoader): 验证集的数据加载器。
        device (str): 设备类型，例如 'cuda' 或 'cpu'。

    返回：
        tuple: (验证集上的准确率, 平均损失)
    """
    model.eval()  # 设置模型为评估模式

    total_loss = 0.0  # 累计损失
    correct_cls = 0  # 分类正确的样本数
    total_cls = 0  # 总的样本数

    all_predictions = []  # 存储所有的预测结果
    all_labels = []  # 存储所有的真实标签

    # 定义损失函数
    criterion_con_loss = SupConLossWithMemoryBank(memory_size=128)
    criterion_classifier = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    attribute_correct_counts = None  # 每个属性正确预测的样本数
    attribute_total_counts = 0  # 属性样本总数

    # 关闭梯度计算，加速推理
    with torch.no_grad():
        for i, (videos, labels, key_index, attributes, gauss_dis, _) in enumerate(valid_loader):
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
                attributes_loss += loss

                # 计算当前属性的预测准确率
                _, predicted = torch.max(attribute_output, 1)
                correct = (predicted == attribute_target).sum().item()
                attribute_correct_counts[attr_idx] += correct

            # 平均属性损失
            attributes_loss /= num_attributes

            # 计算时间注意力损失
            time_attention_loss = mse_loss(time_attention, similarity)

            # 计算监督对比损失
            sup_con_loss = criterion_con_loss(feature_key, feature_video, labels)

            # 计算总损失
            combined_loss = classification_loss + attributes_loss + 2.5 * time_attention_loss + 0.5 * sup_con_loss
            total_loss += combined_loss.item()

            # 记录预测结果和真实标签
            predictions = predict_label.argmax(dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

            attribute_total_counts += batch_size

            # 更新分类正确的样本数和总样本数
            correct_cls += np.sum(predictions == labels.cpu().numpy())
            total_cls += len(labels)

    # 计算总体准确率和平均损失
    accuracy = correct_cls / total_cls
    avg_loss = total_loss / len(valid_loader)

    # 计算其他指标
    metrics_result = calculate_metrics(all_labels, all_predictions)

    # 计算每个属性的准确率
    per_attribute_accuracy = attribute_correct_counts / attribute_total_counts
    average_attribute_accuracy = per_attribute_accuracy.mean()

    # 输出验证结果
    print(f"验证集 - 总损失: {avg_loss:.4f}, "
          f"准确率: {metrics_result['accuracy'] * 100:.2f}%, "
          f"召回率: {metrics_result['recall']:.4f}, "
          f"F1 分数: {metrics_result['f1']:.4f}, "
          f"精度: {metrics_result['precision']:.4f}, "
          f"属性平均准确率: {average_attribute_accuracy * 100:.2f}%")

    # 输出每个属性的准确率
    for idx, acc in enumerate(per_attribute_accuracy):
        print(f"属性 {idx + 1} 准确率: {acc * 100:.2f}%")

    return accuracy, avg_loss


def train(model, train_loader, valid_loader, optimizer, num_epochs, device='cuda'):
    """
    训练模型，并在训练过程中在验证集上评估性能。

    参数：
        model (torch.nn.Module): 待训练的模型。
        train_loader (DataLoader): 训练集的数据加载器。
        valid_loader (DataLoader): 验证集的数据加载器。
        optimizer (torch.optim.Optimizer): 优化器。
        num_epochs (int): 训练的轮数。
        device (str): 设备类型，例如 'cuda' 或 'cpu'。

    返回：
        dict: 包含训练和验证期间的损失和准确率等信息。
    """
    # 用于记录每个 epoch 的损失和准确率
    all_train_accuracy = []
    all_valid_accuracy = []
    all_train_loss = []
    all_valid_loss = []
    all_per_attribute_accuracies = []
    all_average_attribute_accuracies = []

    os.makedirs("./save_weights", exist_ok=True)  # 创建保存模型的目录

    # 初始化损失函数
    criterion_con_loss = SupConLossWithMemoryBank(memory_size=128)
    criterion_classifier = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    def lr_lambda(current_step):
        warmup_steps = 2 * len(train_loader)  # 预热步数
        restart_interval = 10000  # 退火周期的初始步数，之后周期会逐渐增加
        T_mult = 2  # 每次重启后周期增加的倍数

        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # 当前处于第几个重启周期
            restart_count = math.floor(math.log(1 + (current_step - warmup_steps) / restart_interval, T_mult))

            # 当前周期的长度
            T_current = restart_interval * (T_mult ** restart_count)

            # 当前周期内的位置
            current_step_in_restart = (current_step - warmup_steps) - (T_current * (T_mult ** restart_count - 1)) // (
                    T_mult - 1)

            # 余弦退火计算
            progress = current_step_in_restart / T_current
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_accuracy = 0.0  # 初始化最佳准确率

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        total_loss = 0.0  # 累计损失
        all_predictions = []  # 记录所有预测结果
        all_labels = []  # 记录所有真实标签
        attribute_correct_counts = None  # 每个属性的正确预测计数
        attribute_total_counts = 0  # 属性样本总数

        for i, (videos, labels, key_index, attributes, gauss_dis, _) in enumerate(train_loader):
            # 将数据移动到指定设备
            videos = videos.to(device)
            labels = labels.to(device)
            gauss_dis = gauss_dis.to(device)
            key_index = key_index.to(device)
            attributes = attributes.to(device)

            optimizer.zero_grad()  # 清空梯度

            # 模型前向传播
            predict_label, time_attention, clinical_output, feature_key, feature_video = model(videos, key_index)

            # 计算相似度
            similarity = (similarity_frame(videos, key_index) + gauss_dis) / 2.0

            # 计算分类损失
            classification_loss = criterion_classifier(predict_label, labels)

            # 计算属性损失
            attributes_loss = 0.0
            batch_size = attributes.size(0)
            num_attributes = clinical_output.size(2)

            if attribute_correct_counts is None:
                attribute_correct_counts = np.zeros(num_attributes)

            # 遍历每个属性，计算损失和准确率
            for attr_idx in range(num_attributes):
                attribute_output = clinical_output[:, :, attr_idx]  # 当前属性的输出
                attribute_target = attributes[:, attr_idx].long()  # 当前属性的真实标签

                # 计算当前属性的损失
                loss = criterion_classifier(attribute_output, attribute_target)
                attributes_loss += loss

                # 计算当前属性的预测准确率
                _, predicted = torch.max(attribute_output, 1)
                correct = (predicted == attribute_target).sum().item()
                attribute_correct_counts[attr_idx] += correct

            # 平均属性损失
            attributes_loss /= num_attributes

            # 计算时间注意力损失
            time_attention_loss = mse_loss(time_attention, similarity)

            # 计算监督对比损失
            sup_con_loss = criterion_con_loss(feature_key, feature_video, labels)

            # 综合损失
            combined_loss = classification_loss + attributes_loss + 2.5 * time_attention_loss + 0.5 * sup_con_loss
            combined_loss.backward()  # 反向传播

            optimizer.step()  # 更新参数
            scheduler.step()  # 更新学习率

            # 记录损失和预测结果
            total_loss += combined_loss.item()
            predictions = predict_label.argmax(dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

            attribute_total_counts += batch_size

        avg_train_loss = total_loss / len(train_loader)
        metrics_result = calculate_metrics(all_labels, all_predictions)

        # 计算每个属性的准确率和平均准确率
        per_attribute_accuracy = attribute_correct_counts / attribute_total_counts
        average_attribute_accuracy = per_attribute_accuracy.mean()
        all_per_attribute_accuracies.append(per_attribute_accuracy)
        all_average_attribute_accuracies.append(average_attribute_accuracy)

        # 输出训练结果
        print(f"Epoch {epoch + 1}/{num_epochs} - 训练集平均损失: {avg_train_loss:.4f}, "
              f"准确率: {metrics_result['accuracy'] * 100:.2f}%, "
              f"召回率: {metrics_result['recall']:.4f}, "
              f"F1 分数: {metrics_result['f1']:.4f}, "
              f"精度: {metrics_result['precision']:.4f}, "
              f"属性平均准确率: {average_attribute_accuracy * 100:.2f}%")

        # 输出每个属性的准确率
        for idx, acc in enumerate(per_attribute_accuracy):
            print(f"属性 {idx + 1} 准确率: {acc * 100:.2f}%")

        # 在验证集上评估模型
        valid_accuracy, avg_valid_loss = validate_model(model, valid_loader, device)
        best_accuracy = save_model_if_best(model, valid_accuracy, best_accuracy)

        # 记录训练和验证的准确率和损失
        all_train_accuracy.append(metrics_result['accuracy'])
        all_train_loss.append(avg_train_loss)
        all_valid_accuracy.append(valid_accuracy)
        all_valid_loss.append(avg_valid_loss)

    # 返回训练和验证期间的相关结果
    return {
        'model': model,
        'train_accuracy': all_train_accuracy,
        'train_loss': all_train_loss,
        'valid_accuracy': all_valid_accuracy,
        'valid_loss': all_valid_loss,
        'per_attribute_accuracies': all_per_attribute_accuracies,
        'average_attribute_accuracies': all_average_attribute_accuracies
    }

# this file contains all the drawers for the project
# keep the style in a certain mode
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
import os
import pandas as pd

# a drawer to draw the curves while training and testing
def train_epoch_plot(train_records, y_label, sheet_name, output_path):
    y_cols = ['ux1', 'us1', 'ux2', 'us2', 'cs', 'dy_tau', 'p_v', 'Gt']
    y_labels = ['$\sigma_{fc}$', '$\epsilon_{fc}$', '$\sigma_{pk}$', '$\epsilon_{pk}$', '$\sigma_{cs}$', '$\tau_0$', '$\eta$', '$G_t$']
    y_col_label_mapping = dict(zip(y_cols, y_labels))

    plt.figure(figsize=(6, 5))
    plt.plot(train_records, label='Training', color='blue', alpha=0.7, linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.title(f'{y_label} Curve of {y_col_label_mapping[sheet_name]}')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, len(train_records) - 1)  
    plt.savefig(f"{output_path}/{sheet_name}_{y_label}_curve.png")  # 保存图片到指定路径
    plt.show()
  
def curve_epoch_plot(train_records, test_records, y_label, sheet_name, output_path, train_only=True):
    plt.figure(figsize=(10, 5))
    plt.plot(train_records, label='Training', color='blue', alpha=0.7, linewidth=2)
    if not train_only:
        plt.plot(test_records, label='Testing', color='gray', alpha=0.7, linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.title(f'{y_label} Curve of {sheet_name}')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, len(train_records) - 1)  
    plt.ylim(min(min(train_records), min(test_records if not train_only else [float('inf')])) * 0.9, 
             max(max(train_records), max(test_records if not train_only else [float('-inf')])) * 1.1)  # 设置纵坐标范围
    plt.savefig(f"{output_path}/{sheet_name}_{y_label}_curve.png")  # 保存图片到指定路径
    plt.show()

# a drawer to draw the scatter plot which have the R2 index
def scatter_plot(real_sets_train, pred_sets_train, real_sets_test, pred_sets_test, sheet_name, output_path, test_only=True):
    plt.figure(figsize=(6, 6))
    plt.scatter(real_sets_test, pred_sets_test, label='Tesing', alpha=0.6, color='blue', edgecolors='w', s=50)
    if test_only==False:
         plt.scatter(real_sets_train, pred_sets_train, label='Training', alpha=0.6, color='grey', edgecolors='w', s=50)
    plt.plot([min(real_sets_train), max(real_sets_train)], [min(real_sets_train), max(real_sets_train)], color='grey', alpha=0.6, lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Scatter Plot of {sheet_name}')
    plt.grid(True)
    
    # Calculate R2 and MAE
    r2_train = r2_score(real_sets_train, pred_sets_train)
    r2_test = r2_score(real_sets_test, pred_sets_test)
    
    # Add text box with R2 and MAE
    textstr = f'R2 for Training: {r2_train:.4f}\nR2 for Testing: {r2_test:.4f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top')
    
    # Save the figure
    plt.savefig(f"{output_path}/{sheet_name}_scatter_plot.png")
    plt.show()

# a drawer to plot a line
def line_plot(x, y, sheet_name, y_val_cols, output_path):
    plt.figure(figsize=(10, 5))
    plt.xlabel(sheet_name)
    plt.plot(x, y, linestyle='-', color='b', alpha=0.7, linewidth=2)
    plt.ylabel(f'Predicted Result of {y_val_cols}')
    plt.title(f'Plot of {sheet_name} vs {y_val_cols}')
    plt.grid(True)
    
    # 计算曲线的最小值和最大值
    y_min = min(y)
    y_max = max(y)
    
    # 计算 Y 轴的范围，使得曲线的中点位于 Y 轴的中间
    y_center = (y_min + y_max) / 2

    # 设置 Y 轴范围
    plt.ylim(0, max(y_max,1.4*y_center))
    
    plt.savefig(f"{output_path}/{sheet_name} vs {y_val_cols}_plot.png")
    plt.show()

def plot_prediction_vs_true(target_train, mean_train_pred, std_train_pred, 
                            target_test, mean_test_pred, r2_train, mae_train, r2_test, mae_test,
                            sheet_name, output_path):
    """
    绘制真实值 vs 预测均值的散点图，显示置信区间（美化版）。
    
    参数：
        target_train: 训练集真实值
        mean_train_pred: 训练集预测均值
        std_train_pred: 训练集预测标准差
        target_test: 测试集真实值
        mean_test_pred: 测试集预测均值
        sheet_name: 指标名称
        output_path: 输出图像路径
    """
    # 映射字典（保持不变）
    x_cols = ['FA/C', 'w/b', 's/b', 'SP/B', 'If']
    x_labels = ['FA/C', 'w/b', 's/b', 'SP/B', '$I_f$']
    y_cols = ['ux1', 'us1', 'ux2', 'us2', 'cs', 'dy_tau', 'p_v', 'Gt']
    y_labels = ['$\sigma_{fc}$', '$\epsilon_{fc}$', '$\sigma_{pk}$', '$\epsilon_{pk}$', '$\sigma_{cs}$', '$\tau_0$', '$\eta$', '$G_t$']
    y_col_label_mapping = dict(zip(y_cols, y_labels))

    # 创建图形和轴（调整尺寸和DPI）
    plt.style.use('seaborn-v0_8-whitegrid')  # 使用现代风格
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    
    # 计算95%置信区间
    confidence_interval = 1.96 * std_train_pred
    upper_bound = mean_train_pred + confidence_interval
    lower_bound = mean_train_pred - confidence_interval
    
    # 绘制置信区间区域（改进填充效果）
    ax.fill_between(
        target_train, lower_bound, upper_bound,
        color='#3498db', alpha=0.15, label='95% Confidence Interval'
    )
    
    # 绘制训练集预测（改进标记样式）
    train_scatter = ax.scatter(
        target_train, mean_train_pred,
        s=70, alpha=0.8, edgecolors='w', linewidth=0.5,
        color='#2980b9', marker='o', label='Training Data'
    )
    
    # 绘制测试集预测
    test_scatter = ax.scatter(
        target_test, mean_test_pred,
        s=70, alpha=0.8, edgecolors='w', linewidth=0.5,
        color='#e74c3c', marker='s', label='Testing Data'
    )
    
    # 绘制完美预测对角线
    min_val = min(np.min(target_train), np.min(mean_train_pred))
    max_val = max(np.max(target_train), np.max(mean_train_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 
            linestyle='--', color='#2c3e50', linewidth=1.5, 
            alpha=0.7, label='Perfect Prediction')

    # 设置轴标签和标题（改进字体）
    ax.set_xlabel('True Values', fontsize=14, fontweight='bold', labelpad=12)
    ax.set_ylabel('Predicted Values', fontsize=14, fontweight='bold', labelpad=12)
    ax.set_title(
        f'True vs Predicted Values for {y_col_label_mapping[sheet_name]}\nwith 95% Confidence Interval',
        fontsize=15, pad=20, fontweight='bold'
    )

    # 自定义坐标轴刻度
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # 添加网格（更精细的网格线）
    ax.grid(True, linestyle='-', alpha=0.1, color='black')
    
    # 改进图例
    legend = ax.legend(
        loc='upper left', frameon=True, 
        framealpha=0.9, fontsize=12,
        borderpad=0.8, handletextpad=0.5
    )
    legend.get_frame().set_edgecolor('#bdc3c7')

    # 格式化评估指标文本
    metrics_text = (
        f"Training Performance:\n"
        f"R² = {r2_train:.3f}   MAE = {mae_train:.3f}\n"
        f"Testing Performance:\n"
        f"R² = {r2_test:.3f}   MAE = {mae_test:.3f}"
    )
    
    # 添加文本框（改进样式）
    bbox_props = dict(
        boxstyle="round,pad=0.5", 
        facecolor="white", 
        edgecolor="#bdc3c7",
        alpha=0.85,
        linewidth=1.5
    )
    
    ax.text(
        0.98, 0.02, metrics_text,
        transform=ax.transAxes,
        verticalalignment='bottom',
        horizontalalignment='right',
        fontsize=11,
        fontfamily='monospace',
        bbox=bbox_props
    )
    
    # 调整布局并保存
    plt.tight_layout()
    
    # 确保保存路径存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存图像（调整质量和格式）
    plt.savefig(output_path, bbox_inches='tight', dpi=300, format='png')
    plt.show()


def plot_prediction(var, mean_pred, std_pred, y_val_col, target_feature, output_path):
    """
    绘制真实值 vs 预测均值的散点图，显示置信区间。
    
    参数：
        target (array-like): 真实值数组
        mean_pred (array-like): 预测均值数组
        std_pred (array-like): 预测标准差数组
    """
    # 创建图表和轴
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 计算95%置信区间的上限和下限
    confidence_interval = 1.96 * std_pred
    upper_bound = mean_pred + confidence_interval
    lower_bound = mean_pred - confidence_interval
    
    
    # 绘制置信区间上限和下限
    ax.plot(var, upper_bound, color='darkgray', linestyle='--', alpha=0.7, label='Upper Bound')
    ax.plot(var, lower_bound, color='darkgray', linestyle='--', alpha=0.7, label='Lower Bound')
    ax.plot(var, mean_pred, color='red', linestyle='--', alpha=0.7, label='Mean Prediction')
    # 填充置信区间区域
    ax.fill_between(var, lower_bound, upper_bound, color='gray', alpha=0.2)    
    
    # 设置轴标签和标题
    ax.set_xlabel(f'{target_feature}', fontsize=14)
    ax.set_ylabel(f'Predict Values of {y_val_col}', fontsize=14)
    ax.set_title('True vs Predicted Values with 95% Confidence Interval', fontsize=14)
    
    plt.ylim(min(lower_bound)*0.9,max(upper_bound)*1.1)
    # 添加图例
    plt.legend()
    
    # 调整布局并保存图表
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def plot_predict_designed(sample_indices, mean_predictions, uncertainties, y_val_col, output_path):
    """
    绘制散点图，显示每个样本的预测均值及其不确定性。

    参数：
        sample_indices (array-like): 样本的次序（横轴）
        mean_predictions (array-like): 预测均值（纵轴）
        uncertainties (array-like): 预测的不确定性（误差条）
        sheet_name (str): 图表标题中的表名
        output_path (str): 保存图表的路径
    """
    plt.figure(figsize=(10, 6))
    plt.errorbar(sample_indices, mean_predictions, yerr=uncertainties, fmt='o', 
                     ecolor='gray', alpha=0.7, capsize=3, label='Predictions with Uncertainty')
    plt.xlabel('Sample Index', fontsize=14)
    plt.ylabel('Predicted Mean', fontsize=14)
    plt.title(f'Predicted Mean with Uncertainty for {y_val_col}', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_path}/{y_val_col}_uncertainty_plot.png")
    plt.show()
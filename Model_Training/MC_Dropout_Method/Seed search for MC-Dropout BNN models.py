import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import optuna
from optuna.samplers import TPESampler
from functools import partial
import matplotlib.pyplot as plt
import optuna.pruners as pruners
from optuna.pruners import MedianPruner
import functools
import random


# 从您的工具模块导入（保持原样）
from utils.Drawers import train_epoch_plot, scatter_plot, plot_prediction_vs_true
from utils.Data_worker import load_data, load_file, standard_scale_data, inverse_standard_scale_data
from utils.Setup_seed import setup_seed

# 设置目录结构（保持原样）
OUTPUT_DIR = 'output_BayesianNN'
DEVICE = torch.device('cpu')
SCALER_PATH = OUTPUT_DIR + '/' + 'scaler'
FEATURE_SCALER_PATH = SCALER_PATH + '/' + 'feature_scaler'
TARGET_SCALER_PATH = SCALER_PATH + '/' + 'target_scaler'
MODEL_PATH = OUTPUT_DIR + '/' + 'models'
FIGURE_PATH = OUTPUT_DIR + '/' + 'figures'
EXCEL_PATH = OUTPUT_DIR + '/' + 'excel'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FEATURE_SCALER_PATH, exist_ok=True)
os.makedirs(TARGET_SCALER_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(FIGURE_PATH, exist_ok=True)
os.makedirs(EXCEL_PATH, exist_ok=True)

# 改进的贝叶斯神经网络架构
class BayesianRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=1, dropout_p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2*hidden_size)
        self.output = nn.Linear(2*hidden_size, output_size*2)  # 输出均值和log方差
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.output(x)
        
        # 将输出分成均值和log方差
        mean = output[:, :1]
        logvar = output[:, 1:]
        
        return mean, logvar

class BayesianNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=1, dropout_p=0.1):

        super().__init__()
        self.model = BayesianRegressor(input_size, hidden_size, output_size, dropout_p)
        
    def forward(self, x):
        mean, logvar = self.model(x)
        return mean, logvar
    
    def predict_with_uncertainty(model, x, n_samples=100):
        # 数据不确定性 (eval模式)
        model.eval()
        mean, logvar = model(x)
        data_var = torch.exp(logvar).squeeze() 
        
        # 模型不确定性 (train模式)
        model.train()
        mc_samples = torch.stack([model(x)[0] for _ in range(n_samples)])
        model_var = mc_samples.var(dim=0).squeeze()  # 计算模型不确定性
        
        total_var = data_var + model_var
        return mean, total_var.sqrt(), data_var.sqrt(), model_var.sqrt()
    
    def nll(self, x, y):
        """计算负对数似然损失"""
        mean, logvar = self(x)
        inv_var = torch.exp(-logvar)
        loss = torch.sum(inv_var * (y - mean)**2 + logvar) / y.shape[0]
        return loss

# 数据集类（保持原样）
class Data(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.device = DEVICE

    def __len__(self):
        assert len(self.X) == len(self.y)
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.FloatTensor(self.X[i]).to(self.device),
            torch.FloatTensor(self.y[i]).to(self.device)
        )

# 数据加载和预处理（保持原样）
x_cols = ['SF/C', 'FA/C', 'W/B', 'S/B', 'SP/B', 'Vf(%)', 'If']
x_labels = ['SF/C', 'FA/C', 'W/B', 'S/B', 'SP/B', '$V_f$', '$I_f$']
y_cols = ['G(KJm3)', 'UTX(Mpa)','UTS(%)','FCX(MPa)', 'CX(MPa)', 'PV(Pas)', 'YS(Pa)', 'MiniSF(cm)']
y_labels = ['$G_t (kJ/m^3)$', '$\sigma_{u} (MPa)$','$\epsilon_{u} (%)$','$\sigma_{kc} (MPa)$','$\sigma_{cs} (MPa)$', '$\eta (Pa\cdot s)$', '$\\tau_y (Pa)$', '$D_{spread} (cm)$']

x_label_mapping = dict(zip(x_cols, x_labels))
y_label_mapping = dict(zip(y_cols, y_labels))

# 加载原始数据
raw_data = load_data(r'utils/Database.xlsx', sheet_names=['Sheet1'])

# 数据标准化
standard_scale_data(raw_data['Sheet1'][x_cols].values, train=True, 
                   scaler_file=f"{FEATURE_SCALER_PATH}/input_feature_scaler.pkl")
for y_col in y_cols:
    standard_scale_data(raw_data['Sheet1'][y_col].dropna().values.reshape(-1,1), 
                       train=True, scaler_file=f"{TARGET_SCALER_PATH}/{y_col}_target_scaler.pkl")

def extract_nonnull_dataset(raw_data, x_cols, y_col):
    sub_data = raw_data.dropna(subset=[y_col], how='any')
    sub_data = sub_data[x_cols + [y_col]]
    return sub_data

original_data = {y_col: extract_nonnull_dataset(raw_data['Sheet1'], x_cols, y_col) for y_col in y_cols}

@functools.lru_cache(maxsize=1)
def load_scaled_data(y_col):
    feature_scaler = load_file(f"{FEATURE_SCALER_PATH}/input_feature_scaler.pkl")
    target_scaler = load_file(f"{TARGET_SCALER_PATH}/{y_col}_target_scaler.pkl")
    raw_X = original_data[y_col][x_cols].values
    raw_y = original_data[y_col][y_col].values.reshape(-1,1)
    return (
        feature_scaler.transform(raw_X), 
        target_scaler.transform(raw_y),
        raw_X, raw_y
    )

def plot_prediction_vs_true(y_train_true, y_train_pred, train_uncertainties,
                          y_test_true, y_test_pred, test_uncertainties,
                          train_r2, test_r2, train_mae, test_mae,
                          y_col, save_path, figsize=(16, 6)): # Adjusted figsize for better layout with 3 subplots
    """
    改进版绘图函数：区分数据不确定性和模型不确定性
    参数:
        train_uncertainties/test_uncertainties: 字典包含:
            - 'data_std': 数据不确定性（来自BNN的logvar）
            - 'model_std': 模型不确定性（来自MC Dropout）
            - 'total_std': 总不确定性
    """
    # **修改开始**
    # 全局设置字体为 Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'cm' # For LaTeX-like math text if needed
    # **修改结束**
    # 转换为numpy数组（如为PyTorch张量）
    def to_numpy(x):
        return x.cpu().numpy() if torch.is_tensor(x) else x
    
    y_train_true = y_train_true.reshape(-1)
    y_train_pred = y_train_pred.reshape(-1)
    y_test_true = y_test_true.reshape(-1) 
    y_test_pred = y_test_pred.reshape(-1)
    y_train_true = to_numpy(y_train_true)
    y_test_true = to_numpy(y_test_true)
    y_train_pred = to_numpy(y_train_pred)
    y_test_pred = to_numpy(y_test_pred)
    train_data_std = to_numpy(train_uncertainties['data_std'].reshape(-1))
    train_model_std = to_numpy(train_uncertainties['model_std'].reshape(-1))
    train_total_std = to_numpy(train_uncertainties['total_std'].reshape(-1))
    test_data_std = to_numpy(test_uncertainties['data_std'].reshape(-1))
    test_model_std = to_numpy(test_uncertainties['model_std'].reshape(-1))
    test_total_std = to_numpy(test_uncertainties['total_std'].reshape(-1))
    
    # **修改开始**
    plt.style.use('seaborn-v0_8-whitegrid') # 保持样式
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, dpi=300) # 将dpi改小一点，在显示时可能更合适，保存时再设300
    # **修改结束**
    
    # 共享坐标轴范围
    all_values = np.concatenate([y_train_true.flatten(), y_train_pred.flatten(), 
                               y_test_true.flatten(), y_test_pred.flatten()])
    min_val = np.floor(all_values.min() * 0.95)
    max_val = np.ceil(all_values.max() * 1.05)
    lims = [min_val, max_val]
    confidence_interval_data = 1.96 * train_data_std
    confidence_interval_model = 1.96 * train_model_std
    confidence_interval_total = 1.96 * train_total_std
    upper_bound_data = y_train_pred + confidence_interval_data
    lower_bound_data = y_train_pred - confidence_interval_data
    upper_bound_model = y_train_pred + confidence_interval_model
    lower_bound_model = y_train_pred - confidence_interval_model
    upper_bound_total = y_train_pred + confidence_interval_total
    lower_bound_total = y_train_pred - confidence_interval_total
    # Figure 1. Data Uncertainty
    ax1.plot(y_train_true, upper_bound_data, color='darkgray', linestyle='--', alpha=0.7, label='Upper Bound')
    ax1.plot(y_train_true, lower_bound_data, color='darkgray', linestyle='--', alpha=0.7, label='Lower Bound')
    ax1.fill_between(y_train_true, lower_bound_data, upper_bound_data, color='gray', alpha=0.2)
    ax1.scatter(y_train_true, y_train_pred, s=70, alpha=0.8, edgecolors='w',
               color='#2980b9', marker='o')
    ax1.scatter(y_test_true, y_test_pred, s=70, alpha=0.8, edgecolors='w',
               color='#ff6b6b', marker='s')
    ax1.plot(lims, lims, '--', color='#2c3e50', linewidth=1.5, alpha=0.7)
    # ax1.set_title('Aleatoric Uncertainty', fontsize=20, pad=15) # 标题保持注释或在所有子图的共同标题中设置
    ax1.set_xlabel('True Values', fontweight='bold', fontsize=20)
    ax1.set_ylabel('Predicted Values', fontweight='bold', fontsize=20)
    # **修改开始**
    legend1 = ax1.legend(loc='upper left', prop={'family': 'Times New Roman', 'size': 16})
    legend1.set_zorder(20)
    # 增大刻度字体
    ax1.tick_params(axis='both', which='major', labelsize=16)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontname("Times New Roman")
    # **修改结束**
    # Figure 2. Model Uncertainty
    ax2.plot(y_train_true, upper_bound_model, color='darkgray', linestyle='--', alpha=0.7, label='Upper Bound')
    ax2.plot(y_train_true, lower_bound_model, color='darkgray', linestyle='--', alpha=0.7, label='Lower Bound')
    ax2.fill_between(y_train_true, lower_bound_model, upper_bound_model, color='gray', alpha=0.2)
    ax2.scatter(y_train_true, y_train_pred, s=70, alpha=0.8, edgecolors='w',
               color='#2980b9', marker='o')  
    ax2.scatter(y_test_true, y_test_pred, s=70, alpha=0.8, edgecolors='w',
               color='#ff6b6b', marker='s')
    ax2.plot(lims, lims, '--', color='#2c3e50', linewidth=1.5, alpha=0.7)
    # ax2.set_title('Epistemic Uncertainty', fontsize=20, pad=15)
    ax2.set_xlabel('True Values', fontweight='bold', fontsize=20)
    # ax2.set_ylabel('Predicted Values', fontweight='bold', fontsize=16) # 第二个图不需要 Y 轴标签
    # **修改开始**
    ax2.legend(loc='upper left', fontsize=18, prop={'family':'Times New Roman'}).set_visible(False) # 移到总图例
    ax2.tick_params(axis='both', which='major', labelsize=16)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontname("Times New Roman")
    # **修改结束**
    # Figure 3. Total Uncertainty
    ax3.fill_between(y_train_true, lower_bound_model, upper_bound_model, color='blue', alpha=0.2, label='Epistemic Uncertainty Region') # 更改label
    ax3.fill_between(y_train_true, lower_bound_total, lower_bound_model, color='red', alpha=0.2, label='Aleatoric Uncertainty Region') # 确保颜色一致性，并更改label
    ax3.fill_between(y_train_true, upper_bound_model, upper_bound_total, color='red', alpha=0.2) # 保持没有label
    ax3.scatter(y_train_true, y_train_pred, s=70, alpha=0.8, edgecolors='w',
               color='#2980b9', marker='o', label=f'Training Data (R²={train_r2:.3f})') # 更新label
    ax3.scatter(y_test_true, y_test_pred, s=70, alpha=0.8, edgecolors='w',
               color='#ff6b6b', marker='s', label=f'Test Data (R²={test_r2:.3f})') # 更新label
    ax3.plot(lims, lims, '--', color='#2c3e50', linewidth=1.5, alpha=0.7)
    # ax3.set_title('Total Uncertainty (Aleatoric + Epistemic)', fontsize=20, pad=15)
    ax3.set_xlabel('True Values', fontweight='bold', fontsize=20)
    # ax3.set_ylabel('Predicted Values', fontweight='bold', fontsize=16) # 第三个图不需要 Y 轴标签
    # **修改开始**
    legend3 = ax3.legend(loc='upper left', prop={'family': 'Times New Roman', 'size': 16})
    legend3.set_zorder(20)
    ax3.tick_params(axis='both', which='major', labelsize=16)
    for label in ax3.get_xticklabels() + ax3.get_yticklabels():
        label.set_fontname("Times New Roman")
    # **修改结束**
    # ===== 公共设置 =====
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.grid(True, linestyle='-', alpha=0.2)
        # **修改开始**
        # 统一x轴和y轴标签的字体和字号
        ax.xaxis.label.set_fontname("Times New Roman")
        ax.yaxis.label.set_fontname("Times New Roman")
        ax.xaxis.label.set_fontsize(20)
        ax.yaxis.label.set_fontsize(20)
        # **修改结束**
    
    # ===== 添加子图标题 =====
    # 在每个子图上方添加描述
    # ax1.set_title('Aleatoric Uncertainty', fontsize=18, pad=15, fontname="Times New Roman", fontweight='bold')
    # ax2.set_title('Epistemic Uncertainty', fontsize=18, pad=15, fontname="Times New Roman", fontweight='bold')
    # ax3.set_title('Total Uncertainty', fontsize=18, pad=15, fontname="Times New Roman", fontweight='bold')
    # ===== 添加总标题和指标 =====
    # **修改开始**
    # 增大总标题字体
    plt.suptitle(f'{y_label_mapping[y_col]} Prediction with Uncertainty', 
                y=1.02, fontsize=24, fontweight='bold', fontname='Times New Roman')
    
    # 移除被注释掉的指标文本，因为它不会被渲染
    # '''BufferErrormetrics_text ... '''
    # **修改结束**
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_loss(loss_history, y_col, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training Loss for {y_col}')
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()

def kfold_cross_validation(X, y, hidden_size, lr, dropout_p, epochs, n_splits, objective_metric='nll'):
    """执行K折交叉验证，返回平均损失"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    val_mses = []
    val_losses = []
    if objective_metric == 'nll':
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            # 训练模型
            model = BayesianNeuralNetwork(
                input_size=len(x_cols), 
                hidden_size=hidden_size
            ).to(DEVICE)
            
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            # 训练过程
            best_val_loss = float('inf')
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()

                loss = model.nll(torch.FloatTensor(X_train).to(DEVICE), 
                                torch.FloatTensor(y_train).to(DEVICE))
                loss.backward()
                optimizer.step()
                
                # 验证损失
                model.eval()
                with torch.no_grad():
                    val_loss = model.nll(torch.FloatTensor(X_val).to(DEVICE), 
                                            torch.FloatTensor(y_val).to(DEVICE))
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                
            val_losses.append(best_val_loss)

        return np.mean(val_losses)  # 返回平均验证损失

    else:
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            # 训练模型
            model = BayesianNeuralNetwork(
                input_size=len(x_cols), 
                hidden_size=hidden_size
            ).to(DEVICE)
            
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            # 训练过程
            best_val_mse = float('inf')
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()

                loss = model.nll(torch.FloatTensor(X_train).to(DEVICE), 
                                torch.FloatTensor(y_train).to(DEVICE))
                loss.backward()
                optimizer.step()
                
                # 验证损失
                model.eval()
                with torch.no_grad():
                    y_val_pred, _ = model(torch.FloatTensor(X_val).to(DEVICE))
                    val_mse = mean_squared_error(y_val, y_val_pred.cpu().numpy())
                    
                    if val_mse < best_val_mse:
                        best_val_mse = val_mse
            
            val_mses.append(best_val_mse)

        return np.mean(val_mses)  # 返回平均验证损失

def optimize_hyperparameters(y_col, n_trials, test_size=0.2, search_epoch=300, n_splits=10, phase=1, seed=None):
    """分阶段超参数优化，自动保存每一阶段的完整结果"""
    X_scaled, y_scaled, _, _ = load_scaled_data(y_col)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size, random_state=seed)
    
    global best_hidden_size, best_lr  # 声明为全局变量以在阶段间传递
    
    def objective(trial):
        # 公共参数记录逻辑
        record = {
            'phase': phase,
            'datetime': pd.Timestamp.now(),
            'y_col': y_col
        }
        
        if phase == 1:
            params = {
                'hidden_size': trial.suggest_int('hidden_size', 8, 128, step=8),
                'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                'dropout_p': 0.2
            }
            record.update(params)
            record['value'] = kfold_cross_validation(X_train, y_train, **params, 
                                                   epochs=search_epoch, n_splits=n_splits,
                                                   objective_metric='mse')
        
        elif phase == 2:
            params = {
                'hidden_size': trial.study.user_attrs['best_hidden_size'],
                'lr': trial.study.user_attrs['best_lr'],
                'dropout_p': trial.suggest_float('dropout_p', 0.1, 0.5, step=0.05)
            }
            record.update(params)
            record['value'] = kfold_cross_validation(X_train, y_train, **params,
                                                   epochs=search_epoch, n_splits=n_splits,
                                                   objective_metric='nll')
        
        return record['value'] if 'value' in record else None
    
    # 创建Optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=30)
    )
    
    # 阶段2需要继承阶段1的最佳参数
    if phase == 2:
        study.set_user_attr('best_hidden_size', best_hidden_size)
        study.set_user_attr('best_lr', best_lr)
    
    # 执行优化
    study.optimize(objective, n_trials=n_trials // 2)
    
    # === 新增：统一保存所有阶段结果 ===
    os.makedirs(f"{EXCEL_PATH}/{y_col}", exist_ok=True)
    excel_file = f"{EXCEL_PATH}/{y_col}/{y_col}_{seed}_hyperparameter_optimization.xlsx"
    
    # 构建完整记录
    records = []
    for trial in study.trials:
        record = {
            'trial_number': trial.number,
            'phase': phase,
            'datetime': pd.Timestamp.now(),
            'y_col': y_col,
            'hidden_size': trial.params.get('hidden_size'),
            'lr': trial.params.get('lr'),
            'dropout_p': trial.params.get('dropout_p', 0.2 if phase==1 else None),
            'value': trial.value,
            'is_best': trial.number == study.best_trial.number
        }
        records.append(record)
    
    # 转换为DataFrame并保存
    results_df = pd.DataFrame(records)
    
    # 智能写入Excel（追加模式）
    if os.path.exists(excel_file):
        with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            existing_df = pd.read_excel(excel_file)
            combined_df = pd.concat([existing_df, results_df])
            combined_df.to_excel(writer, index=False)
    else:
        results_df.to_excel(excel_file, index=False)
    
    print(f"优化结果已保存到: {excel_file}")
    
    # 更新全局最佳参数（供阶段2使用）
    if phase == 1:
        best_hidden_size = study.best_params['hidden_size']
        best_lr = study.best_params['lr']
    
    return study.best_params, study.best_value, X_train, X_test, y_train, y_test

def train_final_model(y_col, hidden_size=32, lr=0.01, dropout_p=0.2,epochs=3000, test_size=0.2, X_train=None, X_test=None, y_train=None, y_test=None, seed=None):
    """
    使用最优参数训练最终模型，保持测试集划分一致
    返回训练和测试的损失及各项指标
    """
    target_scaler = load_file(f"{TARGET_SCALER_PATH}/{y_col}_target_scaler.pkl")
    
    # 2. 创建模型和优化器
    model = BayesianNeuralNetwork(
        input_size=len(x_cols),
        hidden_size=hidden_size,
        dropout_p=dropout_p  # 新增参数
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 3. 训练准备
    train_loss_history = []
    test_loss_history = []
    best_loss = float('inf')
    samples_per_ephemeral_model = 100  # 用于预测的采样次数
    
    # 4. 训练循环
    for epoch in range(epochs):
        # 4.1 训练步骤
        model.train()
        optimizer.zero_grad()
        
        # 计算损失
        loss = model.nll(torch.FloatTensor(X_train).to(DEVICE), 
                             torch.FloatTensor(y_train).to(DEVICE))
        # 反向传播
        loss.backward()
        optimizer.step()
        
        train_loss_history.append(loss.item())
        
    torch.save(model, f"{MODEL_PATH}/{y_col}_{seed}_best_model.pth")
                    
    # 5. 加载最佳模型进行最终评估
    model = torch.load(f"{MODEL_PATH}/{y_col}_{seed}_best_model.pth", weights_only=False)
    model.eval()
    
    # 6. 最终预测和评估
    with torch.no_grad():
        # 6.1 训练集预测
        train_true = target_scaler.inverse_transform(y_train).flatten()
        test_true = target_scaler.inverse_transform(y_test).flatten()
        
        train_mean, train_total_std, train_data_std, train_model_std = model.predict_with_uncertainty(
            torch.FloatTensor(X_train).to(DEVICE), n_samples=samples_per_ephemeral_model)
        test_mean, test_total_std, test_data_std, test_model_std = model.predict_with_uncertainty(
            torch.FloatTensor(X_test).to(DEVICE), n_samples=samples_per_ephemeral_model)
        
        train_mean = target_scaler.inverse_transform(train_mean.reshape(-1, 1)).flatten()
        test_mean = target_scaler.inverse_transform(test_mean.reshape(-1, 1)).flatten()

        train_data_std = (train_data_std * target_scaler.data_range_[0]).cpu().numpy()
        train_model_std = (train_model_std * target_scaler.data_range_[0]).cpu().numpy()
        train_total_std = (train_total_std * target_scaler.data_range_[0]).cpu().numpy()

        test_data_std = (test_data_std * target_scaler.data_range_[0]).cpu().numpy()
        test_model_std = (test_model_std * target_scaler.data_range_[0]).cpu().numpy()
        test_total_std = (test_total_std * target_scaler.data_range_[0]).cpu().numpy()

    train_metrics = {
        'loss': train_loss_history[-1],  # 最后一个epoch的损失
        'r2': r2_score(train_true, train_mean),
        'mae': mean_absolute_error(train_true, train_mean),
        'rmse': np.sqrt(mean_squared_error(train_true, train_mean)),
    }

    test_metrics = {
        'r2': r2_score(test_true, test_mean),
        'mae': mean_absolute_error(test_true, test_mean),
        'rmse': np.sqrt(mean_squared_error(test_true, test_mean)),
    }

    # 8. 可视化
    graph_path = f"{FIGURE_PATH}/{y_col}_{seed}_graph"
    os.makedirs(graph_path, exist_ok=True)
    
    # 8.1 按真实值排序（训练集和测试集分开排序）
    # 训练集排序
    train_sort_idx = np.argsort(train_true)
    train_true_sorted = train_true[train_sort_idx]
    train_pred_sorted = train_mean[train_sort_idx]
    train_data_std_sorted = train_data_std[train_sort_idx]
    train_model_std_sorted = train_model_std[train_sort_idx]
    train_total_std_sorted = train_total_std[train_sort_idx]

    # 测试集排序
    test_sort_idx = np.argsort(test_true)
    test_true_sorted = test_true[test_sort_idx]
    test_pred_sorted = test_mean[test_sort_idx]
    test_data_std_sorted = test_data_std[test_sort_idx]
    test_model_std_sorted = test_model_std[test_sort_idx]
    test_total_std_sorted = test_total_std[test_sort_idx]

    # 8.2 绘制排序后的预测效果图
    plot_prediction_vs_true(
        train_true_sorted, 
        train_pred_sorted, 
        {'data_std': train_data_std_sorted, 'model_std': train_model_std_sorted, 'total_std': train_total_std_sorted},
        test_true_sorted, 
        test_pred_sorted, 
        {'data_std': test_data_std_sorted, 'model_std': test_model_std_sorted, 'total_std': test_total_std_sorted},
        train_metrics['r2'], 
        test_metrics['r2'],
        train_metrics['mae'], 
        test_metrics['mae'],
        y_col, 
        f"{graph_path}/uncertainty_breakdown.png"
    )

    # 8.2 损失曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Training Loss', color='blue')
    plt.plot(
        np.linspace(0, len(train_loss_history)-1, len(test_loss_history)), 
        test_loss_history, label='Test Loss', color='red'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Negative Log Likelihood)')
    plt.title(f'Loss Curves for {y_col}')
    plt.legend()
    plt.grid()
    plt.savefig(f"{graph_path}/loss_curves.png")
    plt.close()
    
    # 9. 返回结果
    return {
        'train': train_metrics,
        'test': test_metrics,
        'model': model,
        'target_scaler': target_scaler,
        'best_params': {
            'hidden_size': hidden_size,
            'lr': lr,
            'epochs': epochs
        },
        'loss_history': {
            'train': train_loss_history
        }
    }

if __name__ == "__main__":
    train_cols = ['MiniSF(cm)', 'G(KJm3)', 'UTX(Mpa)', 'CX(MPa)']
    all_results = {}
    n_trials = 256
    search_epoch = 300 
    n_splits = 10  
    final_epochs = 2000

    random.seed(3047)
    SEED_GRID = random.sample(range(1000), 20)  # 随机种子列表
    print(f"随机种子列表: {SEED_GRID}")
    for y_col in train_cols:
        for seed in SEED_GRID:
            setup_seed(seed)
            print(f"\n=== 阶段1：优化 lr 和 hidden_size（{y_col}）===")
            best_params_phase1, _, X_train, X_test, y_train, y_test = optimize_hyperparameters(
                y_col, n_trials=n_trials, search_epoch=search_epoch, n_splits=n_splits, phase=1, seed=seed)
            best_hidden_size = best_params_phase1['hidden_size']
            best_lr = best_params_phase1['lr']

            print(f"\n=== 阶段2：优化 dropout_p（{y_col}）===")
            best_params_phase2, *_ = optimize_hyperparameters(
                y_col, n_trials // 4, phase=2, seed=seed)
            best_dropout_p = best_params_phase2['dropout_p']

            # 合并最佳参数
            optimal_params = {
                'hidden_size': best_hidden_size,
                'lr': best_lr,
                'dropout_p': best_dropout_p
            }

            print(f"\n=== 训练最终模型（{y_col}）===")
            results = train_final_model(
                y_col,
                epochs=final_epochs,
                hidden_size=optimal_params['hidden_size'],
                lr=optimal_params['lr'],
                dropout_p=optimal_params['dropout_p'],
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test, 
                seed=seed
            )
            all_results[y_col] = results




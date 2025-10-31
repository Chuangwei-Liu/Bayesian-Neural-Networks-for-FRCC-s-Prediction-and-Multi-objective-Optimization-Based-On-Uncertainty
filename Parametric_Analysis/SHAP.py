import os
import numpy as np
import torch
import shap
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.File_saver import load_file, save_file

DEVICE = torch.device('cpu')
SEED = 33
x_cols = ['SF/C', 'FA/C', 'W/B', 'S/B', 'SP/B', 'Vf(%)', 'If']
x_labels = ['SF/C', 'FA/C', 'W/B', 'S/B', 'SP/B', '$V_f$', '$I_f$']
y_cols = ['G(KJm3)', 'UTX(Mpa)', 'CX(MPa)', 'MiniSF(cm)']
y_labels = ['$G_t$', '$\sigma_u$','$\sigma_{cs}$','$D_{spread}$']

y_col_label_mapping = dict(zip(y_cols, y_labels))
x_col_label_mapping = dict(zip(x_cols, x_labels))

MODEL_DIR = 'best_models'
SCALER_DIR = 'index_analysis_output/scaler'
FEATURE_SCALER_DIR = f"{SCALER_DIR}/feature_scaler"
TARGET_SCALER_DIR = f"{SCALER_DIR}/target_scaler"
SHAP_RESULT_DIR = 'SHAP_results'

os.makedirs(SHAP_RESULT_DIR, exist_ok=True)
os.makedirs(FEATURE_SCALER_DIR, exist_ok=True)
os.makedirs(TARGET_SCALER_DIR, exist_ok=True)

BACKGROUND_SIZE = 30
NSAMPLES = 2000
NUM_EXPLAIN_SAMPLES = 40


best_model_list = {
    'UTX(Mpa)': 'UTX(Mpa)_424_best_model.pth',
    'G(KJm3)': "G(KJm3)_857_best_model.pth",
    'MiniSF(cm)': "MiniSF(cm)_114_best_model.pth",
    'CX(MPa)': "CX(MPa)_618_best_model.pth"  
}

class BayesianRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=1, dropout_p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * hidden_size)
        self.output = nn.Linear(2 * hidden_size, output_size * 2)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.output(x)
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


class Data(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.device = DEVICE

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.FloatTensor(self.X[i]).to(self.device),
            torch.FloatTensor(self.y[i]).to(self.device)
        )

def load_model_and_data(target_var):
    """加载数据和模型"""
    raw_data = pd.read_excel('utils/Database.xlsx', sheet_name='Sheet1')
    valid_data = raw_data.dropna(subset=[target_var])[x_cols + [target_var]]
    
    feature_scaler = load_file(f"{SCALER_DIR}/feature_scaler/input_feature_scaler.pkl")
    target_scaler = load_file(f"{SCALER_DIR}/target_scaler/{target_var}_target_scaler.pkl")
    
    X = feature_scaler.transform(valid_data[x_cols].values)
    y = target_scaler.transform(valid_data[target_var].values.reshape(-1, 1))
    
    model_path = f"{MODEL_DIR}/{best_model_list[target_var]}"
    model = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    return X, y, model, target_scaler

def bnn_predict(model, X, n_samples=50):
    """返回预测均值"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        samples = torch.stack([model(X_tensor)[0] for _ in range(n_samples)], dim=0)
        return samples.mean(dim=0).cpu().numpy()

def analyze_feature_importance(target_var, shap_values, feature_names):
    """计算特征重要性（不保存文件）"""
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values[0])
    
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_shap
    }).sort_values('Importance', ascending=True)  # 升序排列（用于绘图时的倒序显示）
    
    print(f"\n[{target_var}] 参数重要性排序：")
    print(importance_df.sort_values('Importance', ascending=False).to_string(index=False))
    
    return importance_df

def save_shap_plots(target_var, shap_values, samples_to_explain):
    """生成可视化图表（精简版）"""
    feature_labels = [x_col_label_mapping[col] for col in x_cols]
    target_label = y_col_label_mapping[target_var]
    
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 1:
        shap_values = np.expand_dims(shap_values, axis=0)

    # === 1. 特征重要性分析 ===
    importance_df = analyze_feature_importance(target_var, shap_values, feature_labels)
    
    # 过滤掉SF/C特征
    importance_df_filtered = importance_df[importance_df['Feature'] != 'SF/C']
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.labelweight": "bold",
        "font.weight": "bold"
    })

    # === 2. 重要性条形图 ===
    plt.figure(figsize=(5, 5))
    plt.barh(
        importance_df_filtered['Feature'],
        importance_df_filtered['Importance'],
        color='#4682B4',
        height=0.6,
    )
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.xlabel('Mean Absolute SHAP Value', fontsize=16, fontweight='bold')
    plt.title(f'Feature Importance: {target_label}', fontsize=20, pad=10, fontweight='bold')
    # plt.gca().invert_yaxis()  # 使重要性从上到下递减
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{SHAP_RESULT_DIR}/{target_var}_importance.jpg", bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    print(f"SHAP结果保存路径: {SHAP_RESULT_DIR}")
    
    for target_var in tqdm(y_cols, desc="Processing"):
        # 1. 加载数据
        X, y, model, _ = load_model_and_data(target_var)
        
        # 2. 计算SHAP值
        explainer = shap.PermutationExplainer(
            lambda x: bnn_predict(model, x),
            shap.sample(X, min(BACKGROUND_SIZE, len(X)))
        )
        _, X_test = train_test_split(X, test_size=0.2, random_state=SEED)
        shap_values = explainer.shap_values(X_test[:NUM_EXPLAIN_SAMPLES])
        
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values[0])
        shap_values = shap_values.squeeze()
        
        # 3. 保存结果
        save_shap_plots(target_var, shap_values, X_test[:NUM_EXPLAIN_SAMPLES])
    
    print("\n分析完成！重要性排序已显示在终端")
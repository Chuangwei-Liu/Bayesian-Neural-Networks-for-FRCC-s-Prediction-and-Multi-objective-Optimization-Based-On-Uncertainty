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
from torch.distributions import Normal
import math
import time
import sys
import random

from utils.Drawers import train_epoch_plot, scatter_plot, plot_prediction_vs_true
from utils.Data_worker import load_data, load_file, standard_scale_data, inverse_standard_scale_data
from utils.Setup_seed import setup_seed

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

seed = 3047
val_col = 'CX(MPa)'

if len(sys.argv) > 1:
    seed = int(sys.argv[1])
    print(f"Program 2: Received and set random seed to {seed}")

if len(sys.argv) > 2:
    val_col = sys.argv[2]
    print(f"Program 2: Received and set validation column to {val_col}")

setup_seed(seed)

print(f"---{seed} Model Validation Script Started---")
sys.stdout.flush()

class Linear_BBB(nn.Module):
    def __init__(self, input_features, output_features, prior_var=1.0):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        
        self.w_mu = nn.Parameter(torch.Tensor(output_features, input_features).normal_(0, 0.1))
        self.w_rho = nn.Parameter(torch.Tensor(output_features, input_features).uniform_(-3, -2))
        
        self.b_mu = nn.Parameter(torch.Tensor(output_features).normal_(0, 0.1))
        self.b_rho = nn.Parameter(torch.Tensor(output_features).uniform_(-3, -2))
        
        self.prior = Normal(0, math.sqrt(prior_var))
        
        self.w_epsilon = None
        self.b_epsilon = None
        
    def forward(self, x, sample=True):
        if sample or self.w_epsilon is None:
            device = x.device
            w_sigma = torch.log1p(torch.exp(self.w_rho))
            self.w_epsilon = Normal(0, 1).sample(self.w_mu.shape).to(device)
            w = self.w_mu + w_sigma * self.w_epsilon
            
            b_sigma = torch.log1p(torch.exp(self.b_rho))
            self.b_epsilon = Normal(0, 1).sample(self.b_mu.shape).to(device)
            b = self.b_mu + b_sigma * self.b_epsilon
        else:
            w = self.w_mu + torch.log1p(torch.exp(self.w_rho)) * self.w_epsilon
            b = self.b_mu + torch.log1p(torch.exp(self.b_rho)) * self.b_epsilon
        
        return F.linear(x, w, b)
    
class BayesianNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=1, prior_var=0.5):
        super().__init__()
        self.fc1 = Linear_BBB(input_size, hidden_size, prior_var)
        self.fc2 = Linear_BBB(hidden_size, hidden_size, prior_var)
        self.out = Linear_BBB(hidden_size, output_size*2, prior_var)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean, logvar = self.out(x).chunk(2, dim=1)  
        return mean, logvar
    
    def sample_predictions(model, x, n_samples=100):
        with torch.no_grad():
            means = []
            logvars = []
            
            for _ in range(n_samples):
                mean, logvar = model(x)
                means.append(mean)
                logvars.append(logvar)
                
            means = torch.stack(means)  
            logvars = torch.stack(logvars)

            model_std = means.std(dim=0)  
            
            data_var = torch.exp(logvars).mean(dim=0)
            data_std = torch.sqrt(data_var)

            total_std = torch.sqrt(model_std**2 + data_std**2)
            
            return {
                'mean': means.mean(dim=0),
                'total_std': total_std,
                'model_std': model_std,
                'data_std': data_std
            }

    def nll(self, x, y):
        mean, logvar = self(x)
        inv_var = torch.exp(-logvar)
        loss = torch.sum(inv_var * (y - mean)**2 + logvar) / y.shape[0]
        return loss
    
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
    
x_cols = ['SF/C', 'FA/C', 'W/B', 'S/B', 'SP/B', 'Vf(%)', 'If']
x_labels = ['SF/C', 'FA/C', 'W/B', 'S/B', 'SP/B', '$V_f$', '$I_f$']
y_cols = ['G(KJm3)', 'UTX(Mpa)','UTS(%)','FCX(MPa)', 'CX(MPa)', 'PV(Pas)', 'YS(Pa)', 'MiniSF(cm)']
y_labels = ['$G_t (kJ/m^3)$', '$\sigma_{u} (MPa)$','$\epsilon_{u} (%)$','$\sigma_{kc} (MPa)$','$\sigma_{cs} (MPa)$', '$\eta (Pa\cdot s)$', '$\\tau_y (Pa)$', '$D_{spread} (cm)$']

x_label_mapping = dict(zip(x_cols, x_labels))
y_label_mapping = dict(zip(y_cols, y_labels))

raw_data = load_data(r'utils/Database.xlsx', sheet_names=['Sheet1'])

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
                          y_col, save_path, figsize=(16, 6)):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'cm'

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
    
    plt.style.use('seaborn-v0_8-whitegrid') 
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, dpi=300)

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

    ax1.plot(y_train_true, upper_bound_data, color='darkgray', linestyle='--', alpha=0.7, label='Upper Bound')
    ax1.plot(y_train_true, lower_bound_data, color='darkgray', linestyle='--', alpha=0.7, label='Lower Bound')
    ax1.fill_between(y_train_true, lower_bound_data, upper_bound_data, color='gray', alpha=0.2)
    ax1.scatter(y_train_true, y_train_pred, s=70, alpha=0.8, edgecolors='w',
               color='#2980b9', marker='o')
    ax1.scatter(y_test_true, y_test_pred, s=70, alpha=0.8, edgecolors='w',
               color='#ff6b6b', marker='s')
    ax1.plot(lims, lims, '--', color='#2c3e50', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('True Values', fontweight='bold', fontsize=20)
    ax1.set_ylabel('Predicted Values', fontweight='bold', fontsize=20)
    legend1 = ax1.legend(loc='upper left', prop={'family': 'Times New Roman', 'size': 16})
    legend1.set_zorder(20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontname("Times New Roman")

    ax2.plot(y_train_true, upper_bound_model, color='darkgray', linestyle='--', alpha=0.7, label='Upper Bound')
    ax2.plot(y_train_true, lower_bound_model, color='darkgray', linestyle='--', alpha=0.7, label='Lower Bound')
    ax2.fill_between(y_train_true, lower_bound_model, upper_bound_model, color='gray', alpha=0.2)
    ax2.scatter(y_train_true, y_train_pred, s=70, alpha=0.8, edgecolors='w',
               color='#2980b9', marker='o')  
    ax2.scatter(y_test_true, y_test_pred, s=70, alpha=0.8, edgecolors='w',
               color='#ff6b6b', marker='s')
    ax2.plot(lims, lims, '--', color='#2c3e50', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('True Values', fontweight='bold', fontsize=20)
    ax2.legend(loc='upper left', fontsize=18, prop={'family':'Times New Roman'}).set_visible(False) # 移到总图例
    ax2.tick_params(axis='both', which='major', labelsize=16)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontname("Times New Roman")

    ax3.fill_between(y_train_true, lower_bound_model, upper_bound_model, color='blue', alpha=0.2, label='Epistemic Uncertainty Region') # 更改label
    ax3.fill_between(y_train_true, lower_bound_total, lower_bound_model, color='red', alpha=0.2, label='Aleatoric Uncertainty Region') # 确保颜色一致性，并更改label
    ax3.fill_between(y_train_true, upper_bound_model, upper_bound_total, color='red', alpha=0.2) # 保持没有label
    ax3.scatter(y_train_true, y_train_pred, s=70, alpha=0.8, edgecolors='w',
               color='#2980b9', marker='o', label=f'Training Data (R²={train_r2:.3f})') # 更新label
    ax3.scatter(y_test_true, y_test_pred, s=70, alpha=0.8, edgecolors='w',
               color='#ff6b6b', marker='s', label=f'Test Data (R²={test_r2:.3f})') # 更新label
    ax3.plot(lims, lims, '--', color='#2c3e50', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('True Values', fontweight='bold', fontsize=20)

    legend3 = ax3.legend(loc='upper left', prop={'family': 'Times New Roman', 'size': 16})
    legend3.set_zorder(20)
    ax3.tick_params(axis='both', which='major', labelsize=16)
    for label in ax3.get_xticklabels() + ax3.get_yticklabels():
        label.set_fontname("Times New Roman")

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.grid(True, linestyle='-', alpha=0.2)
        ax.xaxis.label.set_fontname("Times New Roman")
        ax.yaxis.label.set_fontname("Times New Roman")
        ax.xaxis.label.set_fontsize(20)
        ax.yaxis.label.set_fontsize(20)

    plt.suptitle(f'{y_label_mapping[y_col]} Prediction with Uncertainty', 
                y=1.02, fontsize=24, fontweight='bold', fontname='Times New Roman')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_loss(train_loss_history, y_col, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Negative Log Likelihood)')
    plt.title(f'Loss Curves for {y_col}')
    plt.legend()
    plt.grid()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def validate_and_plot(y_col, hidden_size, lr, epochs, train_loss_history, test_size=0.2, prior_var=0.5):   
    X_scaled, y_scaled, _, _ = load_scaled_data(y_col)
    target_scaler = load_file(f"{TARGET_SCALER_PATH}/{y_col}_target_scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=seed
    )
    
    model = torch.load(f"{MODEL_PATH}/{y_col}/{y_col}_{seed}_best_model.pth", weights_only=False)
    model.eval()

    with torch.no_grad():
        train_true = target_scaler.inverse_transform(y_train).flatten()
        train_results = model.sample_predictions(
            torch.FloatTensor(X_train).to(DEVICE), n_samples=100)
        train_pred = target_scaler.inverse_transform(train_results['mean'].reshape(-1, 1)).flatten()
        train_data_std = (train_results['data_std'] * target_scaler.data_range_[0]).cpu().numpy()   
        train_model_std = (train_results['model_std'] * target_scaler.data_range_[0]).cpu().numpy()
        train_total_std = (train_results['total_std'] * target_scaler.data_range_[0]).cpu().numpy()

        test_true = target_scaler.inverse_transform(y_test).flatten()
        test_results = model.sample_predictions(
            torch.FloatTensor(X_test).to(DEVICE), n_samples=100)
        test_pred = target_scaler.inverse_transform(test_results['mean'].reshape(-1, 1)).flatten()
        test_data_std = (test_results['data_std'] * target_scaler.data_range_[0]).cpu().numpy()
        test_model_std = (test_results['model_std'] * target_scaler.data_range_[0]).cpu().numpy()
        test_total_std = (test_results['total_std'] * target_scaler.data_range_[0]).cpu().numpy()

    train_metrics = {
        'loss': train_loss_history[-1], 
        'r2': r2_score(train_true, train_pred),
        'mae': mean_absolute_error(train_true, train_pred),
        'rmse': np.sqrt(mean_squared_error(train_true, train_pred)),
    }

    test_metrics = {
        'r2': r2_score(test_true, test_pred),
        'mae': mean_absolute_error(test_true, test_pred),
        'rmse': np.sqrt(mean_squared_error(test_true, test_pred)),
    }

    graph_path = f"{FIGURE_PATH}/{y_col}/{y_col}_{seed}_graph"
    os.makedirs(graph_path, exist_ok=True)

    train_sort_idx = np.argsort(train_true)
    train_true_sorted = train_true[train_sort_idx]
    train_pred_sorted = train_pred[train_sort_idx]
    train_data_std_sorted = train_data_std[train_sort_idx]
    train_model_std_sorted = train_model_std[train_sort_idx]
    train_std_sorted = train_total_std[train_sort_idx]

    test_sort_idx = np.argsort(test_true)
    test_true_sorted = test_true[test_sort_idx]
    test_pred_sorted = test_pred[test_sort_idx]
    test_data_std_sorted = test_data_std[test_sort_idx]
    test_model_std_sorted = test_model_std[test_sort_idx]
    test_std_sorted = test_total_std[test_sort_idx]

    train_true_sorted = train_true_sorted.flatten()
    train_pred_sorted = train_pred_sorted.flatten()
    train_data_std_sorted = train_data_std_sorted.flatten()
    train_model_std_sorted = train_model_std_sorted.flatten()
    train_std_sorted = train_std_sorted.flatten()
    test_true_sorted = test_true_sorted.flatten()
    test_pred_sorted = test_pred_sorted.flatten()
    test_data_std_sorted = test_data_std_sorted.flatten()
    test_model_std_sorted = test_model_std_sorted.flatten()
    test_std_sorted = test_std_sorted.flatten()

    plot_prediction_vs_true(
        train_true_sorted, 
        train_pred_sorted, 
        {'data_std': train_data_std_sorted, 'model_std': train_model_std_sorted, 'total_std': train_std_sorted},
        test_true_sorted, 
        test_pred_sorted, 
        {'data_std': test_data_std_sorted, 'model_std': test_model_std_sorted, 'total_std': test_std_sorted},
        train_metrics['r2'], 
        test_metrics['r2'],
        train_metrics['mae'], 
        test_metrics['mae'],
        y_col, 
        f"{graph_path}/uncertainty_breakdown.png"
    )

    plot_loss(train_loss_history, y_col, f"{graph_path}/{y_col}_loss_curve.png")
    
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
    all_results = {}
    print(f"\n=== Validating and plotting for {val_col} ===")
    #从excel中读取最优超参数
    optimal_params_df = pd.read_excel(f"{EXCEL_PATH}/{val_col}/{val_col}_{seed}_excel/{seed}_optimal_hyperparameters.xlsx", index_col=0)
    train_loss_history = pd.read_excel(f"{EXCEL_PATH}/{val_col}/{val_col}_{seed}_excel/{seed}_train_loss_histories.xlsx", index_col=0)[val_col].dropna().tolist()
    results = validate_and_plot(
        val_col,
        epochs = 2000,
        hidden_size = optimal_params_df.loc[val_col, 'hidden_size'],
        lr = optimal_params_df.loc[val_col, 'lr'],
        prior_var = optimal_params_df.loc[val_col, 'prior_var'],
        train_loss_history = train_loss_history
    )

    all_results[val_col] = results

    results_df = pd.DataFrame.from_dict(all_results, orient='index')
    results_df.to_excel(f"{EXCEL_PATH}/{val_col}/{val_col}_{seed}_excel/{seed}_all_results.xlsx")
    print("\n=== ALL PLOTTING COMPLETED ===")

time.sleep(3)
print(f"---{seed} Model Validation Script Finished---")
sys.stdout.flush()
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
import sys
import time

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
train_col = 'CX(MPa)'

if len(sys.argv) > 1:
    seed = int(sys.argv[1])
    print(f"Program 1: Received and set random seed to {seed}")

if len(sys.argv) > 2:
    train_col = sys.argv[2]
    print(f"Program 1: Received and set training column to {train_col}")    

setup_seed(seed)

print(f"---{seed} Model Training Script Started---")
sys.stdout.flush()

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

def kfold_cross_validation(X, y, hidden_size, lr, dropout_p, epochs, n_splits, objective_metric='nll'):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    val_mses = []
    val_losses = []
    if objective_metric == 'nll':
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            model = BayesianNeuralNetwork(
                input_size=len(x_cols), 
                hidden_size=hidden_size
            ).to(DEVICE)
            
            optimizer = optim.Adam(model.parameters(), lr=lr)

            best_val_loss = float('inf')
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()

                loss = model.nll(torch.FloatTensor(X_train).to(DEVICE), 
                                torch.FloatTensor(y_train).to(DEVICE))
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_loss = model.nll(torch.FloatTensor(X_val).to(DEVICE), 
                                            torch.FloatTensor(y_val).to(DEVICE))
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                
            val_losses.append(best_val_loss)

        return np.mean(val_losses) 
    
    else:
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

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

                model.eval()
                with torch.no_grad():
                    y_val_pred, _ = model(torch.FloatTensor(X_val).to(DEVICE))
                    val_mse = mean_squared_error(y_val, y_val_pred.cpu().numpy())
                    
                    if val_mse < best_val_mse:
                        best_val_mse = val_mse
            
            val_mses.append(best_val_mse)

        return np.mean(val_mses) 
    
def optimize_hyperparameters(y_col, n_trials, test_size=0.2, search_epoch=300, n_splits=10, phase=1):
    X_scaled, y_scaled, _, _ = load_scaled_data(y_col)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size, random_state=seed)
    
    global best_hidden_size, best_lr  
    
    def objective(trial):
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
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=30)
    )
    
    if phase == 2:
        study.set_user_attr('best_hidden_size', best_hidden_size)
        study.set_user_attr('best_lr', best_lr)
    
    study.optimize(objective, n_trials=n_trials // 2)
    
    os.makedirs(f"{EXCEL_PATH}/{y_col}/{y_col}_{seed}_excel", exist_ok=True)
    excel_file = f"{EXCEL_PATH}/{y_col}/{y_col}_{seed}_excel/{seed}_{y_col}_hyperparameter_optimization.xlsx"
    
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
    
    results_df = pd.DataFrame(records)
    
    if os.path.exists(excel_file):
        with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            existing_df = pd.read_excel(excel_file)
            combined_df = pd.concat([existing_df, results_df])
            combined_df.to_excel(writer, index=False)
    else:
        results_df.to_excel(excel_file, index=False)

    print(f"Optimization Results has been saved: {excel_file}")

    if phase == 1:
        best_hidden_size = study.best_params['hidden_size']
        best_lr = study.best_params['lr']
    
    return study.best_params, study.best_value

def train_final_model(y_col, hidden_size=32, lr=0.01, dropout_p=0.2,epochs=2000, test_size=0.2):
    X_scaled, y_scaled, _, _ = load_scaled_data(y_col)
    target_scaler = load_file(f"{TARGET_SCALER_PATH}/{y_col}_target_scaler.pkl")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=seed
    )

    model = BayesianNeuralNetwork(
        input_size=len(x_cols),
        hidden_size=hidden_size,
        dropout_p=dropout_p  
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_loss_history = []
    test_loss_history = []
    best_loss = float('inf')
    samples_per_ephemeral_model = 100  
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        loss = model.nll(torch.FloatTensor(X_train).to(DEVICE), 
                             torch.FloatTensor(y_train).to(DEVICE))
        loss.backward()
        optimizer.step()
        
        train_loss_history.append(loss.item())
    os.makedirs(f"{MODEL_PATH}/{y_col}", exist_ok=True)
    torch.save(model, f"{MODEL_PATH}/{y_col}/{y_col}_{seed}_best_model.pth")
    return train_loss_history

if __name__ == "__main__":
    n_trials = 256
    optimal_params = {}
    train_loss_histories = {}

    print(f"\n=== Phase 1 :Optimizing hidden size and learning rate for {train_col} ===")
    best_params_phase1, best_value_phase1 = optimize_hyperparameters(
        train_col, n_trials=n_trials, phase=1)
    best_hidden_size = best_params_phase1['hidden_size']
    best_lr = best_params_phase1['lr']

    print(f"\n=== Phase 2 :Optimizing dropout rate for {train_col} ===")
    best_params_phase2, best_value_phase2 = optimize_hyperparameters(
        train_col, n_trials=n_trials, phase=2)
    best_dropout_p = best_params_phase2['dropout_p']

    optimal_params[train_col] = {
        'hidden_size': best_hidden_size,
        'lr': best_lr,
        'dropout_p': best_dropout_p
    }
    # save optimal params to a excel file
    optimal_params_df = pd.DataFrame.from_dict(optimal_params, orient='index')
    optimal_params_df.to_excel(f"{EXCEL_PATH}/{train_col}/{train_col}_{seed}_excel/{seed}_{train_col}_optimal_params.xlsx")
    print(f"Best params for {train_col}: {optimal_params[train_col]} with validation value: {best_value_phase2}")
    print(f"\n=== Training final model for {train_col} ===")
    train_loss_history = train_final_model(
        train_col, 
        hidden_size=best_hidden_size, 
        lr=best_lr, 
        dropout_p=best_dropout_p, 
        epochs=2000
    )
    train_loss_histories[train_col] = train_loss_history
    print("\n=== ALL TRAINING COMPLETED ===")
    pd.DataFrame(dict([(k,pd.Series(v)) for k,v in train_loss_histories.items()])).to_excel(f"{EXCEL_PATH}/{train_col}/{train_col}_{seed}_excel/{seed}_train_loss_histories.xlsx")
    print("=== TRAIN LOSS HISTORIES SAVED ===")
time.sleep(2)
print(f"---{seed} Model Training Script Finished---")
sys.stdout.flush()
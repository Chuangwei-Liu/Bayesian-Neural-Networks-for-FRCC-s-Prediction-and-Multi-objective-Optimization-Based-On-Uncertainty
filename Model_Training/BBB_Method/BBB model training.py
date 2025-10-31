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

def kfold_cross_validation(X, y, hidden_size, lr, prior_var, epochs, n_splits):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    val_losses = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
            
        model = BayesianNeuralNetwork(
            input_size=len(x_cols), 
            hidden_size=hidden_size,
            prior_var=prior_var
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
                
        val_losses.append(best_val_loss.item())

        return np.mean(val_losses)  

def optimize_hyperparameters(y_col, n_trials, test_size=0.2):
    X_scaled, y_scaled, _, _ = load_scaled_data(y_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=seed
    )
    
    def objective(trial):
        params = {
            'hidden_size': trial.suggest_int('hidden_size', 8, 128, step=8),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'prior_var': trial.suggest_float('prior_var', 0.01, 1.0, log=True)
        }

        return kfold_cross_validation(X_train, y_train, **params, epochs=300, n_splits=5)
    
    study = optuna.create_study(
        direction='minimize', 
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=30)
    )
    
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value  

def train_final_model(y_col, hidden_size=32, lr=0.01, epochs=5000, test_size=0.2, prior_var=0.5):
    X_scaled, y_scaled, _, _ = load_scaled_data(y_col)
    target_scaler = load_file(f"{TARGET_SCALER_PATH}/{y_col}_target_scaler.pkl")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=seed
    )

    model = BayesianNeuralNetwork(
        input_size=len(x_cols),
        hidden_size=hidden_size,
        prior_var=prior_var  
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

    print(f"\n=== Optimizing hyperparameters for {train_col} ===")
    best_params, best_value = optimize_hyperparameters(train_col, n_trials)
    optimal_params[train_col] = best_params
    os.makedirs(f"{EXCEL_PATH}/{train_col}/{train_col}_{seed}_excel", exist_ok=True)
    pd.DataFrame.from_dict(optimal_params, orient='index').to_excel(f"{EXCEL_PATH}/{train_col}/{train_col}_{seed}_excel/{seed}_optimal_hyperparameters.xlsx")
    print(f"Best params for {train_col}: {best_params} with validation loss: {best_value}")
    print(f"\n=== Training final model for {train_col} ===")
    train_loss_history = train_final_model(
        train_col,
        epochs=2000,  
        hidden_size=optimal_params[train_col]['hidden_size'],
        lr=optimal_params[train_col]['lr'],
        prior_var=optimal_params[train_col]['prior_var']
    )
    train_loss_histories[train_col] = train_loss_history
    print("\n=== ALL TRAINING COMPLETED ===")
    pd.DataFrame(dict([(k,pd.Series(v)) for k,v in train_loss_histories.items()])).to_excel(f"{EXCEL_PATH}/{train_col}/{train_col}_{seed}_excel/{seed}_train_loss_histories.xlsx")
    print("=== TRAIN LOSS HISTORIES SAVED ===")

time.sleep(2)
print(f"---{seed} Model Training Script Finished---")
sys.stdout.flush()
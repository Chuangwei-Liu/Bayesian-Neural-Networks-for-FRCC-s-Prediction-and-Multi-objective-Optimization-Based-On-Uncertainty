#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Objective Optimization of FRCC Design Based on BNN Model & NSGA-II & TOPSIS 
with separate handling for PE and PVA fibers.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# Add parent directory to path to import custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.Setup_seed import setup_seed
from utils.Data_worker import load_data, standard_scale_data, inverse_standard_scale_data

# Set random seed for reproducibility
setup_seed(42)
print("Random seed set to 42")

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
best_models_dir = os.path.join(base_dir, "best_models")
output_dir = os.path.join(base_dir, "optimization_results")
scaler_path = os.path.join(output_dir, "scaler")
feature_scaler_path = os.path.join(scaler_path, "feature_scaler")
target_scaler_path = os.path.join(scaler_path, "target_scaler")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(feature_scaler_path, exist_ok=True)
os.makedirs(target_scaler_path, exist_ok=True)

x_cols = ['SF/C', 'FA/C', 'W/B', 'S/B', 'SP/B', 'Vf(%)', 'If']
x_labels = ['SF/C', 'FA/C', 'W/B', 'S/B', 'SP/B', '$V_f$', '$I_f$']
y_cols = ['G(KJm3)', 'UTX(Mpa)','UTS(%)','FCX(MPa)', 'CX(MPa)', 'PV(Pas)', 'YS(Pa)', 'MiniSF(cm)']
y_labels = ['$G_t (kJ/m^3)$', '$\sigma_{u} (MPa)$','$\epsilon_{u} (%)$','$\sigma_{kc} (MPa)$','$\sigma_{cs} (MPa)$', '$\eta (Pa\cdot s)$', '$\\tau_y (Pa)$', '$D_{spread} (cm)$']

x_label_mapping = dict(zip(x_cols, x_labels))
y_label_mapping = dict(zip(y_cols, y_labels))

raw_data = load_data(r'utils/Database.xlsx', sheet_names=['Sheet1'])

standard_scale_data(raw_data['Sheet1'][x_cols].values, train=True, 
                   scaler_file=f"{feature_scaler_path}/input_feature_scaler.pkl")
for y_col in y_cols:
    standard_scale_data(raw_data['Sheet1'][y_col].dropna().values.reshape(-1,1), 
                       train=True, scaler_file=f"{target_scaler_path}/{y_col}_target_scaler.pkl")

# Define BNN model classes to match the structure used in training
class BayesianRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=1, dropout_p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2*hidden_size)
        self.output = nn.Linear(2*hidden_size, output_size*2)  # outputs mean and log variance
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.output(x)

        # Split output into mean and log variance
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
    
    def predict_with_uncertainty(self, x, n_samples=100):
        # Data uncertainty (eval mode)
        self.eval()
        mean, logvar = self(x)
        data_var = torch.exp(logvar).squeeze()
        
        # Model uncertainty (train mode)
        self.train()
        mc_samples = torch.stack([self(x)[0] for _ in range(n_samples)])
        model_var = mc_samples.var(dim=0).squeeze()
        
        total_var = data_var + model_var
        return mean, total_var.sqrt(), data_var.sqrt(), model_var.sqrt()

# Register the classes to make loading safe
torch.serialization.add_safe_globals([BayesianRegressor, BayesianNeuralNetwork])

# Load BNN models - enhanced version to handle different model structures
def load_bnn_model(model_path):
    """Load a BNN model from file with various fallback strategies."""
    print(f"Trying to load model from: {model_path}")

    # Try direct loading
    try:
        model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
        print(f"Successfully loaded model: {type(model).__name__}")
        model.eval()
        return model
    except Exception as e:
        print(f"Direct loading failed: {e}")

        # Try reconstructing the model
        try:
            # Create a new model instance with typical architecture
            reconstructed_model = BayesianNeuralNetwork(input_size=7, hidden_size=64, output_size=1, dropout_p=0.1)
            # Try loading state dict
            try:
                # Try loading with weights_only=False
                state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
                if hasattr(state_dict, "state_dict"):
                    state_dict = state_dict.state_dict()
                reconstructed_model.load_state_dict(state_dict)
                print("Successfully loaded model using state_dict")
                reconstructed_model.eval()
                return reconstructed_model
            except Exception as e1:
                print(f"State dict loading failed: {e1}")

                # Last resort: try with weights_only=True
                try:
                    state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
                    reconstructed_model.load_state_dict(state_dict)
                    print("Successfully loaded model using weights_only=True")
                    reconstructed_model.eval()
                    return reconstructed_model
                except Exception as e2:
                    print(f"weights_only=True loading failed: {e2}")
                    raise Exception("All model loading attempts failed")
        except Exception as e:
            print(f"Model reconstruction failed: {e}")
            raise

# Define function to get BNN prediction with uncertainty
def get_bnn_prediction(model, X, scaler_file=None, n_samples=30, feature_scaler_file=None):
    """Get BNN prediction with uncertainty estimation using MC Dropout."""
    # Apply feature scaling if a feature scaler file is provided
    if feature_scaler_file is not None and os.path.exists(feature_scaler_file):
        try:
            with open(feature_scaler_file, "rb") as f:
                feature_scaler = pickle.load(f)
            # Apply feature scaling to input
            X = feature_scaler.transform(X)
        except Exception as e:
            print(f"Error applying feature scaling: {e}")
    
    # Convert input to tensor
    X_tensor = torch.FloatTensor(X)

    # Get predictions with uncertainty
    mean, total_std, data_std, model_std = model.predict_with_uncertainty(X_tensor, n_samples=n_samples)
    
    # Detach tensors from computation graph to avoid gradient errors
    mean = mean.detach()
    total_std = total_std.detach()
    data_std = data_std.detach()
    model_std = model_std.detach()

    # Apply inverse scaling if a scaler file is provided
    if scaler_file is not None and os.path.exists(scaler_file):
        try:
            with open(scaler_file, "rb") as f:
                scaler = pickle.load(f)
            # Reshape mean for inverse transform
            mean_reshaped = mean.reshape(-1, 1).numpy()
            # Apply inverse transform to mean
            mean_inv = scaler.inverse_transform(mean_reshaped).flatten()
            
            # Scale uncertainty by data range
            uncertainty_scale = scaler.data_range_[0] if hasattr(scaler, "data_range_") else 1.0
            total_uncertainty = (total_std * uncertainty_scale).numpy().flatten()
            data_uncertainty = (data_std * uncertainty_scale).numpy().flatten()
            model_uncertainty = (model_std * uncertainty_scale).numpy().flatten()

            return mean_inv, total_uncertainty, data_uncertainty, model_uncertainty
        except Exception as e:
            print(f"Error applying inverse scaling: {e}")
            return (mean.numpy().flatten(), 
                   total_std.numpy().flatten(), 
                   data_std.numpy().flatten(), 
                   model_std.numpy().flatten())

    return (mean.numpy().flatten(), 
           total_std.numpy().flatten(), 
           data_std.numpy().flatten(), 
           model_std.numpy().flatten())

# Define the NSGA-II optimization problems - one for PE and one for PVA fiber
class FRCCOptimizationProblemBase(ElementwiseProblem):
    """Base class for FRCC optimization problems."""
    def __init__(self, UTX_model, G_model, MiniSF_model, fiber_type):
        # Store models
        self.UTX_model = UTX_model
        self.G_model = G_model
        self.MiniSF_model = MiniSF_model
        self.fiber_type = fiber_type  # "PE" or "PVA"
        
        # Set fiber constant based on type
        if fiber_type == "PE":
            self.I_f_value = 0.430752174
            # Define lower and upper bounds for design variables
            # SF/C, FA/C, W/B, S/B, SP/B, V_f (fiber volume fraction)
            xl = np.array([0.0, 0.25, 0.25, 0.3, 0.001, 0.0])
            xu = np.array([0.0, 1.5, 0.37, 0.6, 0.01, 2.0])
        else:  # PVA
            self.I_f_value = 0.247431979
            # Define lower and upper bounds for design variables
            # SF/C, FA/C, W/B, S/B, SP/B, V_f (fiber volume fraction)
            xl = np.array([0.0, 0.25, 0.25, 0.36, 0.001, 0.0])
            xu = np.array([0.0, 4.4, 0.4, 0.6, 0.02, 2.0])

        # Define paths to scalers
        base_dir = os.path.dirname(os.path.abspath(__file__))
        target_scaler_dir = os.path.join(base_dir, "index_analysis_output", "scaler", "target_scaler")
        feature_scaler_dir = os.path.join(base_dir, "index_analysis_output", "scaler", "feature_scaler")
        
        # Target scalers
        self.UTX_scaler = os.path.join(target_scaler_dir, "UTX(Mpa)_target_scaler.pkl")
        self.G_scaler = os.path.join(target_scaler_dir, "G(KJm3)_target_scaler.pkl")
        self.MiniSF_scaler = os.path.join(target_scaler_dir, "MiniSF(cm)_target_scaler.pkl")
        
        # Feature scaler
        self.feature_scaler = os.path.join(feature_scaler_dir, "input_feature_scaler.pkl")

        # Check if scaler files exist
        for scaler_file in [self.UTX_scaler, self.G_scaler, self.MiniSF_scaler, self.feature_scaler]:
            if not os.path.exists(scaler_file):
                print(f"Warning: Scaler file not found: {scaler_file}")


        # Initialize the problem with 6 variables and 6 objectives
        # We don't include I_f as a variable since it's fixed based on fiber type
        super().__init__(n_var=6, n_obj=6, n_ieq_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        # Extract design variables
        SF_C, FA_C, W_B, S_B, SP_B, V_f = x

        # Prepare input for BNN models (add fixed I_f value)
        X = np.array([[SF_C, FA_C, W_B, S_B, SP_B, V_f, self.I_f_value]])

        # Get predictions with uncertainty and apply inverse scaling
        UTX_mean, UTX_uncertainty, _, _ = get_bnn_prediction(self.UTX_model, X, self.UTX_scaler, feature_scaler_file=self.feature_scaler)
        G_mean, G_uncertainty, _, _ = get_bnn_prediction(self.G_model, X, self.G_scaler, feature_scaler_file=self.feature_scaler)
        MiniSF_mean, MiniSF_uncertainty, _, _ = get_bnn_prediction(self.MiniSF_model, X, self.MiniSF_scaler, feature_scaler_file=self.feature_scaler)

        # Define objectives (negative for maximization)
        objectives = [
            -UTX_mean[0],           # Maximize UTX
            -G_mean[0],             # Maximize G
            -MiniSF_mean[0],        # Maximize MiniSF
            UTX_uncertainty[0],     # Minimize UTX uncertainty
            G_uncertainty[0],       # Minimize G uncertainty
            MiniSF_uncertainty[0]   # Minimize MiniSF uncertainty
        ]

        out["F"] = objectives

# PE Fiber specific optimization problem
class PEFiberOptimizationProblem(FRCCOptimizationProblemBase):
    def __init__(self, UTX_model, G_model, MiniSF_model):
        super().__init__(UTX_model, G_model, MiniSF_model, fiber_type="PE")

# PVA Fiber specific optimization problem
class PVAFiberOptimizationProblem(FRCCOptimizationProblemBase):
    def __init__(self, UTX_model, G_model, MiniSF_model):
        super().__init__(UTX_model, G_model, MiniSF_model, fiber_type="PVA")

# TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
def topsis(decision_matrix, weights=None):
    """Implementation of TOPSIS method for multi-criteria decision making."""
    # Normalize the decision matrix
    norm_matrix = decision_matrix / np.sqrt(np.sum(decision_matrix**2, axis=0))

    # Apply weights if provided, otherwise equal weights
    if weights is None:
        weights = np.ones(decision_matrix.shape[1]) / decision_matrix.shape[1]

    # Ensure weights sum to 1
    weights = weights / np.sum(weights)

    # Apply weights to the normalized matrix
    weighted_matrix = norm_matrix * weights

    # Determine ideal and anti-ideal solutions
    ideal_best = np.max(weighted_matrix, axis=0)
    ideal_worst = np.min(weighted_matrix, axis=0)

    # Calculate separation measures
    separation_best = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1))
    separation_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))

    # Calculate relative closeness to the ideal solution
    closeness = separation_worst / (separation_best + separation_worst)

    # Rank alternatives (higher score = better)
    ranking = np.argsort(closeness)[::-1]

    return ranking, closeness

# Create a mapping between the original labels and their LaTeX representations
def get_latex_label(label, with_units=True):
    """Convert raw text labels to LaTeX format with optional units."""
    latex_map = {
        "UTX(MPa)": r"$\sigma_u$",
        "G(KJm3)": r"$G_t$",
        "MiniSF(cm)": r"$D_{spread}$",
        "UTX Uncertainty": r"$\Delta\sigma_u$",  # 添加不确定性的直接映射
        "G Uncertainty": r"$\Delta G_t$",
        "MiniSF Uncertainty": r"$\Delta D_{spread}$",
        "UTX Cert.": r"$\Delta\sigma_u$",
        "G Cert.": r"$\Delta G_t$",
        "MiniSF Cert.": r"$\Delta D_{spread}$",
    }
    
    units_map = {
        "UTX(MPa)": r" (MPa)",
        "G(KJm3)": r" (kJ/m^3)",
        "MiniSF(cm)": r" (cm)",
        "UTX Uncertainty": r" (MPa)",
        "G Uncertainty": r" (kJ/m^3)",
        "MiniSF Uncertainty": r" (cm)",
        "UTX Cert.": r" (MPa)",
        "G Cert.": r" (kJ/m^3)",
        "MiniSF Cert.": r" (cm)",
    }
    
    if label in latex_map:
        if with_units and label in units_map:
            return latex_map[label] + units_map[label]
        return latex_map[label]
    
    # 处理可能的字符串分离情况(如果标签经过了处理)
    if "Uncertainty" in label:
        base = label.split(" Uncertainty")[0]
        if base == "UTX":
            return r"$\sigma_{\sigma_u}$" + (r" (MPa)" if with_units else "")
        elif base == "G":
            return r"$\sigma_{G_t}$" + (r" (kJ/m^3)" if with_units else "")
        elif base == "MiniSF":
            return r"$\sigma_{D_{spread}}$" + (r" (cm)" if with_units else "")
    
    # 处理 Cert. 情况
    if "Cert." in label:
        base = label.split(" Cert.")[0]
        if base == "UTX" or base == r"$\sigma_u$":
            return r"$\sigma_{\sigma_u}$" + (r" (MPa)" if with_units else "")
        elif base == "G" or base == r"$G_t$":
            return r"$\sigma_{G_t}$" + (r" (kJ/m^3)" if with_units else "")
        elif base == "MiniSF" or base == r"$D_{spread}$":
            return r"$\sigma_{D_{spread}}$" + (r" (cm)" if with_units else "")
    
    # 返回原始标签如果没有找到
    return label

def create_combined_pareto_plots(F_original, scores, ranking, output_dir, fiber_type):
    """Create a combined figure with three Pareto front plots showing different relationships with TOPSIS score coloring."""
    # Set global font to Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"
    # Create a figure with subplots (1x3)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    # Get best solution index
    best_idx = ranking[0]
    
    # 更大的字体大小
    AXIS_LABEL_SIZE = 20
    TITLE_SIZE = 26
    TICK_SIZE = 20
    LEGEND_SIZE = 20
    COLORBAR_LABEL_SIZE = 20
    COLORBAR_TICK_SIZE = 20
    
    # UTX vs G subplot
    ax1 = axes[0]
    scatter1 = ax1.scatter(F_original[:, 0], F_original[:, 1],
                         c=scores, s=100, cmap="viridis", alpha=0.7)
    cbar1 = fig.colorbar(scatter1, ax=ax1)
    cbar1.set_label("TOPSIS Score", fontname="Times New Roman", fontsize=COLORBAR_LABEL_SIZE, labelpad=15)
    cbar1.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
    for label in cbar1.ax.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(COLORBAR_TICK_SIZE)
    
    # Highlight best solution
    ax1.scatter(F_original[best_idx, 0], F_original[best_idx, 1], c="red",
               s=250, marker="*", edgecolors="black", linewidth=2,
               label="Best TOPSIS Solution")
    
    ax1.set_xlabel(get_latex_label("UTX(MPa)"), fontsize=AXIS_LABEL_SIZE, fontname="Times New Roman")
    ax1.set_ylabel(get_latex_label("G(KJm3)"), fontsize=AXIS_LABEL_SIZE, fontname="Times New Roman")
    ax1.set_title(f"{get_latex_label('UTX(MPa)', False)} vs. {get_latex_label('G(KJm3)', False)} ({fiber_type} Fiber)", 
                 fontsize=TITLE_SIZE, color="black", fontname="Times New Roman", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(prop={"family":"Times New Roman", "size": LEGEND_SIZE})
    
    # 增大刻度标签大小
    ax1.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(TICK_SIZE)
    
    # UTX vs MiniSF subplot
    ax2 = axes[1]
    scatter2 = ax2.scatter(F_original[:, 0], F_original[:, 2],
                         c=scores, s=100, cmap="viridis", alpha=0.7)
    cbar2 = fig.colorbar(scatter2, ax=ax2)
    cbar2.set_label("TOPSIS Score", fontname="Times New Roman", fontsize=COLORBAR_LABEL_SIZE, labelpad=15)
    cbar2.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
    for label in cbar2.ax.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(COLORBAR_TICK_SIZE)
        
    # Highlight best solution
    ax2.scatter(F_original[best_idx, 0], F_original[best_idx, 2], c="red",
               s=250, marker="*", edgecolors="black", linewidth=2,
               label="Best TOPSIS Solution")
        
    ax2.set_xlabel(get_latex_label("UTX(MPa)"), fontsize=AXIS_LABEL_SIZE, fontname="Times New Roman")
    ax2.set_ylabel(get_latex_label("MiniSF(cm)"), fontsize=AXIS_LABEL_SIZE, fontname="Times New Roman")
    ax2.set_title(f"{get_latex_label('UTX(MPa)', False)} vs. {get_latex_label('MiniSF(cm)', False)} ({fiber_type} Fiber)", 
                 fontsize=TITLE_SIZE, color="black", fontname="Times New Roman", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(prop={"family":"Times New Roman", "size": LEGEND_SIZE})
    
    # 增大刻度标签大小
    ax2.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(TICK_SIZE)
    
    # G vs MiniSF subplot
    ax3 = axes[2]
    scatter3 = ax3.scatter(F_original[:, 1], F_original[:, 2],
                         c=scores, s=100, cmap="viridis", alpha=0.7)
    cbar3 = fig.colorbar(scatter3, ax=ax3)
    cbar3.set_label("TOPSIS Score", fontname="Times New Roman", fontsize=COLORBAR_LABEL_SIZE, labelpad=15)
    cbar3.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
    for label in cbar3.ax.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(COLORBAR_TICK_SIZE)
        
    # Highlight best solution
    ax3.scatter(F_original[best_idx, 1], F_original[best_idx, 2], c="red",
               s=250, marker="*", edgecolors="black", linewidth=2,
               label="Best TOPSIS Solution")
        
    ax3.set_xlabel(get_latex_label("G(KJm3)"), fontsize=AXIS_LABEL_SIZE, fontname="Times New Roman")
    ax3.set_ylabel(get_latex_label("MiniSF(cm)"), fontsize=AXIS_LABEL_SIZE, fontname="Times New Roman")
    ax3.set_title(f"{get_latex_label('G(KJm3)', False)} vs. {get_latex_label('MiniSF(cm)', False)} ({fiber_type} Fiber)", 
                 fontsize=TITLE_SIZE, color="black", fontname="Times New Roman", fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend(prop={"family":"Times New Roman", "size": LEGEND_SIZE})
    
    # 增大刻度标签大小
    ax3.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    for label in ax3.get_xticklabels() + ax3.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(TICK_SIZE)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f"combined_pareto_plots_{fiber_type}.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    return os.path.join(output_dir, f"combined_pareto_plots_{fiber_type}.png")

def radar_chart(categories, values, title, save_path=None, max_values=None, all_solutions=None, highlight_best=True):
    """Create and save a radar chart with highlighting for the best solution."""
    # Set font family to Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"
    # 更大的字体大小
    CATEGORY_SIZE = 28
    TITLE_SIZE = 32
    TICK_SIZE = 24
    LEGEND_SIZE = 24
    COLORBAR_LABEL_SIZE = 28
    COLORBAR_TICK_SIZE = 24
    # Number of categories
    N = len(categories)
    # Create angles for each category
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Convert categories to LaTeX notation
    latex_categories = [get_latex_label(cat, with_units=False) for cat in categories]
    
    # Create figure and polar axes
    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw=dict(polar=True))
    
    # Set zorder for axes elements to ensure they are on top
    ax.set_zorder(10)  # High zorder for axes
    ax.patch.set_zorder(0)  # Low zorder for background
    ax.patch.set_visible(False)  # Make background transparent
    
    # First plot all individual solutions if provided
    if all_solutions is not None:
        # Create a colormap for the solutions based on TOPSIS Score
        cmap = plt.cm.viridis
        # Sort solutions by TOPSIS Score if available for better color mapping
        if "TOPSIS Score" in all_solutions.columns:
            topsis_scores = all_solutions["TOPSIS Score"].values
            norm = plt.Normalize(topsis_scores.min(), topsis_scores.max())
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
        else:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
            sm.set_array([])
        # Plot each solution with a different color based on TOPSIS score
        for i, (_, solution) in enumerate(all_solutions.iterrows()):
            try:
                # Extract values for each category
                solution_values = np.array([
                    solution["UTX(MPa)"],
                    solution["G(KJm3)"],
                    solution["MiniSF(cm)"],
                    solution["UTX Cert."],  # Use the precalculated certainty columns
                    solution["G Cert."],
                    solution["MiniSF Cert."]
                ])
                # Close the loop
                solution_values_closed = np.append(solution_values, solution_values[0])
                # Scale if max_values is provided
                if max_values is not None:
                    max_values_array = np.array(max_values)
                    max_values_closed = np.append(max_values_array, max_values_array[0])
                    solution_values_closed = solution_values_closed / max_values_closed
                # Plot with color from colormap based on TOPSIS Score if available
                if "TOPSIS Score" in all_solutions.columns:
                    color = cmap(norm(solution["TOPSIS Score"]))
                else:
                    color = cmap(i / len(all_solutions))
                ax.plot(angles, solution_values_closed, color=color, linewidth=1.5, alpha=0.7, zorder=3)
            except Exception as e:
                print(f"Error plotting solution {i}:", e)
                print(f"Solution data: {solution}")
                continue
    
    # Plot the main values (best solution) if provided
    if values is not None and highlight_best:
        values_array = np.array(values)
        values_closed = np.append(values_array, values_array[0])
        # Scale values if max_values is provided
        if max_values is not None:
            max_values_array = np.array(max_values)
            max_values_closed = np.append(max_values_array, max_values_array[0])
            scaled_values = values_closed / max_values_closed
            # Plot the highlighted solution with higher zorder
            ax.fill(angles, scaled_values, alpha=0.25, color="red", zorder=5)
            ax.plot(angles, scaled_values, "o-", linewidth=3, color="red", label="Best TOPSIS Solution", zorder=6)
        else:
            ax.fill(angles, values_closed, alpha=0.25, color="red", zorder=5)
            ax.plot(angles, values_closed, "o-", linewidth=3, color="red", label="Best TOPSIS Solution", zorder=6)
    
    # Set category labels with Times New Roman font - 增大字体
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(latex_categories, fontname="Times New Roman", fontsize=CATEGORY_SIZE)
    
    # Customize ticks - 增大字体并将刻度置于最上层
    ax.tick_params(axis='y', labelsize=TICK_SIZE, zorder=15)  # 添加 zorder=15 确保刻度在最上层
    for label in ax.get_yticklabels():
        label.set_fontname("Times New Roman")
        label.set_fontsize(TICK_SIZE)
        label.set_zorder(20)  # 设置标签的zorder，确保在最上层
    
    # 设置网格线，确保它们在最下层
    ax.grid(True, zorder=1)
    
    # Add title with Times New Roman font, larger size, and black color
    plt.title(title, size=TITLE_SIZE, color="black", y=1.1, fontname="Times New Roman", fontweight="bold")
    
    # Add colorbar to show TOPSIS scores - 增大字体
    if all_solutions is not None and "TOPSIS Score" in all_solutions.columns:
        cbar = fig.colorbar(sm, ax=ax, pad=0.1)
        cbar.set_label("TOPSIS Score", rotation=270, labelpad=25, fontname="Times New Roman", fontsize=COLORBAR_LABEL_SIZE)
        cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
        for label in cbar.ax.get_yticklabels():
            label.set_fontname("Times New Roman")
            label.set_fontsize(COLORBAR_TICK_SIZE)
    
    # Add legend if needed - 增大字体并移到右上角更靠外的位置
    if values is not None and highlight_best:
        # 使用 bbox_to_anchor 将图例移到右上方更远的位置
        legend = plt.legend(loc="upper right", bbox_to_anchor=(1.0, 1.1), 
                          prop={"family": "Times New Roman", "size": LEGEND_SIZE})
        # 设置图例的zorder确保在最上层
        legend.set_zorder(20)
    
    # Save the figure if save_path is provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path
    else:
        plt.show()
        return None

# 创建组合雷达图和Top5性能比较图
def create_combined_radar_top5(categories, best_values, radar_df, top5_chart_path, 
                             max_vals, fiber_type, output_dir):
    """Create a combined figure with radar chart and top5 performance comparison."""
    # Create a figure with 1x2 layout
    plt.figure(figsize=(24, 10))
    
    # Add first subplot for radar chart
    plt.subplot(1, 2, 1)
    
    # Use the radar_chart function with the figure and axes we created
    temp_radar_path = os.path.join(output_dir, "temp_radar.png")
    radar_chart(categories, best_values, f"Pareto Solutions Performance ({fiber_type} Fiber)",
               temp_radar_path, max_vals, radar_df, highlight_best=True)
    
    # Load the radar chart image and the top5 chart image
    radar_img = plt.imread(temp_radar_path)
    top5_img = plt.imread(top5_chart_path)
    
    # Create a new figure for the combined plots
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # Display images
    axes[0].imshow(radar_img)
    axes[0].axis('off')
    
    axes[1].imshow(top5_img)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"combined_radar_top5_{fiber_type}.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Remove temporary file
    if os.path.exists(temp_radar_path):
        os.remove(temp_radar_path)

def create_fiber_comparison_chart(pe_topsis_df, pva_topsis_df, output_dir):
    """Create a comparison chart between the top PE and PVA fiber solutions with y-axis starting from 0."""
    # Set global font to Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"
    
    # 更大的字体大小
    AXIS_LABEL_SIZE = 20
    TITLE_SIZE = 26
    TICK_SIZE = 20
    LEGEND_SIZE = 20
    
    # Get top 3 solutions from each fiber type
    pe_top = pe_topsis_df.iloc[:3]
    pva_top = pva_topsis_df.iloc[:3]
    
    # Metrics to compare
    metrics = ["UTX(MPa)", "G(KJm3)", "MiniSF(cm)"]
    
    # Create a figure with subplots for each metric
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Calculate positions
        x = np.array([0, 1])  # PE, PVA
        width = 0.25
        
        # Plot bars for each rank
        for rank in range(3):
            if rank < len(pe_top):
                pe_val = pe_top.iloc[rank][metric]
                ax.bar(x[0] - width + rank*width, pe_val, width, 
                       label=f"PE Rank {rank+1}" if i == 0 else "", 
                       color=plt.cm.Blues(0.5 + rank*0.2))
            
            if rank < len(pva_top):
                pva_val = pva_top.iloc[rank][metric]
                ax.bar(x[1] - width + rank*width, pva_val, width, 
                       label=f"PVA Rank {rank+1}" if i == 0 else "", 
                       color=plt.cm.Oranges(0.5 + rank*0.2))
        
        # Set labels with larger font
        ax.set_title(get_latex_label(metric, False), fontsize=TITLE_SIZE-4, fontname="Times New Roman", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["PE Fiber", "PVA Fiber"], fontname="Times New Roman", fontsize=TICK_SIZE)
        ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
        ax.grid(True, axis="y", alpha=0.3)
        
        # Set y-axis to start from 0
        max_val = max(pe_top[metric].max() if len(pe_top) > 0 else float("-inf"), 
                      pva_top[metric].max() if len(pva_top) > 0 else float("-inf"))
        
        # Add some margin at the top only
        margin = max_val * 0.1
        ax.set_ylim([0, max_val + margin])
        
        # 增大y轴标签
        if i == 0:  # 只给第一个子图设置y轴标签
            ax.set_ylabel("Value", fontsize=AXIS_LABEL_SIZE, fontname="Times New Roman")
    
    # Add a single legend for the entire figure with larger font
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0), 
              ncol=6, prop={"family": "Times New Roman", "size": LEGEND_SIZE})
    
    plt.suptitle("Comparison of Top PE vs PVA Fiber Solutions", 
                fontsize=TITLE_SIZE, fontname="Times New Roman", fontweight="bold")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(output_dir, "pe_vs_pva_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

# 运行特定纤维类型的优化
# 运行特定纤维类型的优化
def run_optimization(fiber_type, UTX_model, G_model, MiniSF_model, output_dir):
    """Run the optimization for a specific fiber type."""
    # Define the optimization problem based on fiber type
    if fiber_type == "PE":
        problem = PEFiberOptimizationProblem(UTX_model, G_model, MiniSF_model)
    else:  # PVA
        problem = PVAFiberOptimizationProblem(UTX_model, G_model, MiniSF_model)

    # Configure the algorithm
    algorithm = NSGA2(
        pop_size=100,
        n_offsprings=100,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    # Define termination criteria
    termination = get_termination("n_gen", 100)

    # Run the optimization
    print(f"Running NSGA-II optimization for {fiber_type} fiber...")
    result = minimize(problem, algorithm, termination, verbose=True)

    # Extract the Pareto front
    X = result.X  # Design variables
    F = result.F  # Objectives

    # Convert objectives to their original values (negate the maximization objectives)
    F_original = F.copy()
    F_original[:, :3] = -F_original[:, :3]  # Negate the first 3 objectives to get actual values

    # Create DataFrame for Pareto front
    pareto_df = pd.DataFrame()
    
    # Design variables
    variable_names = ["SF/C", "FA/C", "W/B", "S/B", "SP/B", "V_f"]
    for i, name in enumerate(variable_names):
        pareto_df[name] = X[:, i]

    # Add fiber type and I_f value
    if fiber_type == "PE":
        pareto_df["I_f"] = 0.430752174
    else:  # PVA
        pareto_df["I_f"] = 0.247431979
        
    pareto_df["Fiber Type"] = fiber_type

    # Objectives
    objective_names = ["UTX(MPa)", "G(KJm3)", "MiniSF(cm)",
                      "UTX Uncertainty", "G Uncertainty", "MiniSF Uncertainty"]
    for i, name in enumerate(objective_names):
        pareto_df[name] = F_original[:, i]

    # Save Pareto front to CSV
    fiber_output_dir = os.path.join(output_dir, fiber_type)
    os.makedirs(fiber_output_dir, exist_ok=True)
    
    pareto_csv_path = os.path.join(fiber_output_dir, "pareto_front.csv")
    pareto_df.to_csv(pareto_csv_path, index=False)
    print(f"Pareto front for {fiber_type} fiber saved to: {pareto_csv_path}")

    # Apply TOPSIS for ranking Pareto solutions
    # Define weights for objectives (adjust based on preferences)
    weights = np.array([0.2, 0.2, 0.2, 0.133, 0.133, 0.133])  # Equal weights

    # Adjust objective values for TOPSIS (normalize and set direction)
    topsis_matrix = F_original.copy()

    # Rank solutions using TOPSIS
    ranking, scores = topsis(topsis_matrix, weights)

    # Create DataFrame for TOPSIS results
    topsis_df = pareto_df.iloc[ranking].copy()
    topsis_df["TOPSIS Score"] = scores[ranking]
    topsis_df["TOPSIS Rank"] = np.arange(1, len(ranking) + 1)

    # Save TOPSIS ranking to CSV
    topsis_csv_path = os.path.join(fiber_output_dir, "topsis_ranking.csv")
    topsis_df.to_csv(topsis_csv_path, index=False)
    print(f"TOPSIS ranking for {fiber_type} fiber saved to: {topsis_csv_path}")

    # Create visualizations
    print(f"Creating visualizations for {fiber_type} fiber...")

    # Create combined Pareto front plots (3-in-1)
    create_combined_pareto_plots(F_original, scores, ranking, fiber_output_dir, fiber_type)

    print(f"Visualizations for {fiber_type} fiber saved to: {fiber_output_dir}")
    
    # # Create bar chart for TOP 5 solutions (保持注释或如果你需要它独立输出)
    # top5_chart_path = create_top5_performance_chart(pareto_df, topsis_df, fiber_output_dir, fiber_type)

    # ================================================================
    # 将所有雷达图所需变量的定义移动到这里，在调用 radar_chart 之前
    # Prepare data for radar charts
    categories = ["UTX(MPa)", "G(KJm3)", "MiniSF(cm)",
                 "UTX Cert.", "G Cert.", "MiniSF Cert."]
    
    max_vals = np.array([
        pareto_df["UTX(MPa)"].max(),
        pareto_df["G(KJm3)"].max(),
        pareto_df["MiniSF(cm)"].max(),
        1 / (pareto_df["UTX Uncertainty"].min() + 1e-10), # 最小的不确定性对应最大的确定性
        1 / (pareto_df["G Uncertainty"].min() + 1e-10),
        1 / (pareto_df["MiniSF Uncertainty"].min() + 1e-10)
    ])

    # Convert pareto_df to include the inverse of uncertainties for radar plotting
    radar_df = pareto_df.copy()
    radar_df["UTX Cert."] = 1 / (radar_df["UTX Uncertainty"] + 1e-10)
    radar_df["G Cert."] = 1 / (radar_df["G Uncertainty"] + 1e-10)
    radar_df["MiniSF Cert."] = 1 / (radar_df["MiniSF Uncertainty"] + 1e-10)

    # Add TOPSIS scores to radar_df for coloring
    radar_df["TOPSIS Score"] = np.nan
    for rank, score in zip(ranking, scores[ranking]):
        radar_df.loc[radar_df.index == rank, "TOPSIS Score"] = score

    # Get best solution for highlighting
    best_solution = topsis_df.iloc[0]
    best_values = np.array([
        best_solution["UTX(MPa)"],
        best_solution["G(KJm3)"],
        best_solution["MiniSF(cm)"],
        1 / (best_solution["UTX Uncertainty"] + 1e-10),
        1 / (best_solution["G Uncertainty"] + 1e-10),
        1 / (best_solution["MiniSF Uncertainty"] + 1e-10)
    ])
    # ================================================================

    # 现在调用 radar_chart，所有变量都已定义
    radar_chart_save_path = os.path.join(fiber_output_dir, f"radar_chart_{fiber_type}.png")
    radar_chart(categories, best_values, f"Pareto Solutions Performance ({fiber_type} Fiber)",
               radar_chart_save_path, max_vals, radar_df, highlight_best=True)

    # 创建组合雷达图和top5性能图 (保持注释，因为目标是只单独输出雷达图)
    # create_combined_radar_top5(categories, best_values, radar_df, top5_chart_path, 
    #                           max_vals, fiber_type, fiber_output_dir)

    
    return topsis_df


# Main execution
if __name__ == "__main__":
    print("Multi-Objective Optimization of FRCC Design with PE-PVA Fiber Distinction Based on BNN Model & NSGA-II & TOPSIS")

    print("Loading BNN models...")
    # Load the best models
    UTX_model = load_bnn_model(os.path.join(best_models_dir, "UTX(Mpa)_424_best_model.pth"))
    G_model = load_bnn_model(os.path.join(best_models_dir, "G(KJm3)_857_best_model.pth"))
    MiniSF_model = load_bnn_model(os.path.join(best_models_dir, "MiniSF(cm)_114_best_model.pth"))
    print("Models loaded successfully!")

    # Create main output directory
    main_output_dir = os.path.join(output_dir, "optimization_results")
    os.makedirs(main_output_dir, exist_ok=True)

    # Run optimization for PE fiber
    pe_topsis_df = run_optimization("PE", UTX_model, G_model, MiniSF_model, main_output_dir)

    # Run optimization for PVA fiber
    pva_topsis_df = run_optimization("PVA", UTX_model, G_model, MiniSF_model, main_output_dir)
    
    # Create comparative visualizations
    print("Creating comparative visualizations...")
    create_fiber_comparison_chart(pe_topsis_df, pva_topsis_df, main_output_dir)
    
    print("Multi-objective optimization and analysis completed successfully!")

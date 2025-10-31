import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
from matplotlib.gridspec import GridSpec

np.random.seed(33)
OUTPUT_DIR = 'Distributions of Features'

x_cols = ['FA/C', 'W/B', 'S/B', 'SP/B', 'Vf(%)', 'If', 'CX(MPa)', 'FCX(MPa)', 'UTX(Mpa)', 'UTS(%)', 'G(KJm3)', 'MiniSF(cm)']
x_labels = ['FA/C', 'W/B', 'S/B', 'SP/B', '$V_f$', '$I_f$', '$\sigma_{cs}$ (MPa)', '$\sigma_{fc}$ (MPa)', '$\sigma_{u}$ (MPa)', '$\\varepsilon_{u}$ (%)', '$G_t$ (MJ/m³)', '$D_{spread}$ (cm)']
x_col_label_mapping = dict(zip(x_cols, x_labels))

def plot_distribution(df):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'cm'

    sns.set_style("whitegrid")
    sns.set_palette("husl")

    fig = plt.figure(figsize=(18, 20))
    gs = GridSpec(4, 3, figure=fig, wspace=0.3, hspace=0.4)  

    for i, feature in enumerate(x_cols):
        ax = fig.add_subplot(gs[i])
        feature_column = df[feature].dropna()
        
        mean_feature = np.mean(feature_column)
        std_feature = np.std(feature_column)
        
        count, bins, _ = ax.hist(feature_column, bins=30, density=True, 
                                alpha=0.6, color='skyblue', 
                                edgecolor='black', linewidth=0.8)
        
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mean_feature, std_feature)
        ax.plot(x, p, 'darkblue', linewidth=2)

        ax.set_title(f"{x_col_label_mapping[feature]}\n$\mu={mean_feature:.2f}$, $\sigma={std_feature:.2f}$", 
                    fontsize=18, pad=10, weight='bold') 
        ax.set_xlabel('', fontsize=16) 
        ax.set_ylabel('Density', fontsize=16)  
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', labelsize=14)  
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout(pad=1.0, h_pad=1.0, w_pad=1.0) 

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "all_features_distribution.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"Saved combined distribution plot to {output_path}")

    print("\nStatistical Summaries:")
    for feature in x_cols:
        feature_column = df[feature].dropna()
        mean_feature = np.mean(feature_column)
        std_feature = np.std(feature_column)
        print(f"{x_col_label_mapping[feature]:<10} | Mean: {mean_feature:.4f} | Std: {std_feature:.4f}")

if __name__ == "__main__":
    xls = pd.ExcelFile('utils/Database.xlsx')
    data = pd.read_excel(xls, 'Sheet1')
    df = pd.DataFrame(data, columns=x_cols)
    df = df.astype(float)
    plot_distribution(df)
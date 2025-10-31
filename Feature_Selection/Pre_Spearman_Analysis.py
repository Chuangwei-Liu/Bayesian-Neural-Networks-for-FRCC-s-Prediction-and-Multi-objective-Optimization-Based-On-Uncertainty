import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

np.random.seed(33)
OUTPUT_DIR = 'feature_selection_graph'

x_cols = ['FA/C', 'W/B', 'S/B', 'SP/B', 'VMA/B', 'Vf(%)', 'Df(um)', 'Lf(mm)', 'Ef(GPa)', 'Tf(MPa)']
x_labels = ['FA/C', 'W/B', 'S/B', 'SP/B', 'VMA/B', '$V_f$', '$D_f$', '$L_f$', '$E_f$', '$T_f$']

def plot_spearman(df):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'cm'
    
    correlation, p_values = spearmanr(df)
    corr_df = pd.DataFrame(correlation, columns=x_labels, index=x_labels)
    plt.figure(figsize=(10, 10))
    sns.set_theme(font='Times New Roman')
    sns.set_palette("coolwarm")
    sns.set(font_scale=1.2) 

    mask = np.zeros_like(corr_df, dtype=bool)
    cmap = sns.diverging_palette(20, 220, as_cmap=True)

    plt.rcParams['font.family'] = 'serif'  
    plt.rcParams['font.serif'] = ['Times New Roman']  
    plt.rcParams['mathtext.fontset'] = 'custom'  
    plt.rcParams['mathtext.rm'] = 'Times New Roman'

    ax = sns.heatmap(
        corr_df, 
        mask=mask, 
        cmap=cmap, 
        annot=True, 
        fmt=".2f", 
        linewidths=.5, 
        cbar=True, 
        cbar_kws={
            "shrink": .8,
            "label": "Correlation" 
        }, 
        square=True, 
        center=0, 
        vmin=-1, 
        vmax=1,
        annot_kws={"size": 20}  
    )

    plt.title('Spearman Correlation Heatmap before PCA',fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 24}, pad=20)
    plt.xlabel('Input Features',  fontdict={'family': 'Times New Roman', 'size': 22})
    plt.ylabel('Input Features',  fontdict={'family': 'Times New Roman', 'size': 22})

    ax.tick_params(axis='both', which='major', labelsize=16) 
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel("Correlation", fontfamily='Times New Roman', fontsize=22)
    cbar.ax.tick_params(labelsize=18)  # 颜色条刻度字体大小

    output_path = OUTPUT_DIR + '/Spearman_Correlation_Heatmap_before_PCA.png'
    plt.tight_layout()  # 防止标签被截断
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved high-quality heatmap as '{output_path}'")

if __name__ == "__main__":
    xls = pd.ExcelFile('utils/Database.xlsx')
    data = pd.read_excel(xls, 'Sheet1')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame(data, columns=x_cols)
    df = df.astype(float)
    plot_spearman(df)
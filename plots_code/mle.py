import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

OUTPUT_PATH = "images/mle.pdf"

METHODS_ORDER_DOWN_TO_UP = [
    "TabPC",
    "ShallowMixture",
    "Fully Factorized (P)",
    "Fully Factorized",
    "TabDiff",
    "TabSyn",
    "TabDDPM",
    "CoDi",
    "TVAE",
    "CTGAN"
][::-1]

# right plot: mle
df_mle = pd.read_csv("tables/csv/mle.csv")
classification_datasets = ['Adult', 'Default', 'Diabetes', 'Magic', 'Shoppers']
regression_datasets = ['Beijing', 'News']

# Calculate MLE averages separately for classification and regression
mle_class_avg = df_mle[df_mle['dataset'].isin(classification_datasets)].groupby('method')['mean'].mean().reset_index()
mle_class_avg.rename(columns={'mean': 'MLE AUCROC'}, inplace=True)

mle_reg_avg = df_mle[df_mle['dataset'].isin(regression_datasets)].groupby('method')['mean'].mean().reset_index()
mle_reg_avg.rename(columns={'mean': 'MLE RMSE'}, inplace=True)

# Filter out GReaT and STaSy
mle_class_avg = mle_class_avg[~mle_class_avg['method'].isin(['GReaT', 'STaSy'])].reset_index(drop=True)
mle_reg_avg = mle_reg_avg[~mle_reg_avg['method'].isin(['GReaT', 'STaSy'])].reset_index(drop=True)

# Reorder methods according to METHODS_ORDER_DOWN_TO_UP
mle_class_avg['method'] = pd.Categorical(mle_class_avg['method'], categories=METHODS_ORDER_DOWN_TO_UP, ordered=True)
mle_class_avg = mle_class_avg.sort_values('method').reset_index(drop=True)

mle_reg_avg['method'] = pd.Categorical(mle_reg_avg['method'], categories=METHODS_ORDER_DOWN_TO_UP, ordered=True)
mle_reg_avg = mle_reg_avg.sort_values('method').reset_index(drop=True)

# Invert the order of methods
mle_class_avg = mle_class_avg.iloc[::-1].reset_index(drop=True)
mle_reg_avg = mle_reg_avg.iloc[::-1].reset_index(drop=True)

# Set seaborn style
sns.set_style("whitegrid")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 4), sharey=True)

y_pos = np.arange(len(mle_class_avg))
bar_height = 0.5

# LEFT PLOT - MLE AUCROC
color_aucroc = sns.color_palette("Set2", 2)[0]
ax1.barh(y_pos, mle_class_avg['MLE AUCROC'], bar_height, 
        label='MLE (AUC)', color=color_aucroc, alpha=0.8, edgecolor='black', linewidth=0.5)

ax1.set_yticks(y_pos)
labels = [r"$\mathbf{" + label + "}$" if label == "TabPC" else label 
          for label in mle_class_avg['method']]
ax1.set_yticklabels(labels, fontsize=10)
ax1.set_xlabel('Average MLE (AUC)', fontsize=11)
ax1.set_xlim(0-0.001, 1.001)
ax1.set_xticks(np.arange(0, 1.2, 0.2))
#ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), fontsize=10, frameon=True)
ax1.grid(True, alpha=0.3, axis='x')

# RIGHT PLOT - MLE RMSE
color_rmse = sns.color_palette("Set2", 2)[1]
ax2.barh(y_pos, mle_reg_avg['MLE RMSE'], bar_height, 
        label='MLE (RMSE)', color=color_rmse, alpha=0.8, edgecolor='black', linewidth=0.5)

ax2.set_xlabel('Average MLE (RMSE)', fontsize=11)
max_rmse = mle_reg_avg['MLE RMSE'].max()
ax2.set_xlim(0-0.001, max_rmse * 1.05)
#ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), fontsize=10, frameon=True)
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()

# Save figure as PDF
plt.savefig(OUTPUT_PATH, format='pdf', bbox_inches='tight', dpi=300)
print(f"Plot saved to {OUTPUT_PATH}")



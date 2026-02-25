import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

OUTPUT_PATH = "images/saturation_overview.pdf"

# left plot: shape, trend and lr detection
df_shape = pd.read_csv("tables/csv/shape.csv")
df_trend = pd.read_csv("tables/csv/trend.csv")
df_lr_detection = pd.read_csv("tables/csv/lr_detection.csv")

# right plot: mle
df_mle = pd.read_csv("tables/csv/mle.csv")
classification_datasets = ['Adult', 'Default', 'Diabetes', 'Magic', 'Shoppers']
regression_datasets = ['Beijing', 'News']


# Extract Average rows for each metric
trend_avg = df_trend[df_trend['dataset'] == 'Average'][['method', 'mean']].copy()
trend_avg.rename(columns={'mean': 'Trend'}, inplace=True)

shape_avg = df_shape[df_shape['dataset'] == 'Average'][['method', 'mean']].copy()
shape_avg.rename(columns={'mean': 'Shape'}, inplace=True)

lr_detection_avg = df_lr_detection[df_lr_detection['dataset'] == 'Average'][['method', 'mean']].copy()
lr_detection_avg.rename(columns={'mean': 'LR Detection'}, inplace=True)

# Merge all metrics
merged_df = trend_avg.merge(shape_avg, on='method').merge(lr_detection_avg, on='method')

# Filter out GReaT and STaSy
merged_df = merged_df[~merged_df['method'].isin(['GReaT', 'STaSy'])].reset_index(drop=True)

# Invert the order of methods
merged_df = merged_df.iloc[::-1].reset_index(drop=True)

# Calculate MLE averages separately for classification and regression
mle_class_avg = df_mle[df_mle['dataset'].isin(classification_datasets)].groupby('method')['mean'].mean().reset_index()
mle_class_avg.rename(columns={'mean': 'MLE AUCROC'}, inplace=True)

mle_reg_avg = df_mle[df_mle['dataset'].isin(regression_datasets)].groupby('method')['mean'].mean().reset_index()
mle_reg_avg.rename(columns={'mean': 'MLE RMSE'}, inplace=True)

# Filter out GReaT and STaSy
mle_class_avg = mle_class_avg[~mle_class_avg['method'].isin(['GReaT', 'STaSy'])].reset_index(drop=True)
mle_reg_avg = mle_reg_avg[~mle_reg_avg['method'].isin(['GReaT', 'STaSy'])].reset_index(drop=True)

# Invert the order of methods to match left plot
mle_class_avg = mle_class_avg.set_index('method').loc[merged_df['method']].reset_index()
mle_reg_avg = mle_reg_avg.set_index('method').loc[merged_df['method']].reset_index()

# Set seaborn style
print(merged_df)
sns.set_style("whitegrid")

# Create figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

# LEFT PLOT - Shape, Trend, LR Detection
colors_left = sns.color_palette("husl", 3)
metrics_left = ['LR Detection', 'Shape', 'Trend']

y_pos = np.arange(len(merged_df))
bar_height = 0.25

for idx, metric in enumerate(metrics_left):
    offset = (idx - 1) * bar_height
    ax1.barh(y_pos + offset, merged_df[metric], bar_height, 
            label=metric, color=colors_left[idx], alpha=0.8, edgecolor='black', linewidth=0.5)

ax1.set_yticks(y_pos)
labels = [r"$\mathbf{" + label + "}$" if label == "TabPC" else label 
          for label in merged_df['method']]
ax1.set_yticklabels(labels, fontsize=10)
ax1.set_xlabel('Average score', fontsize=11)
ax1.set_xlim(0-0.001, 1.001)
ax1.set_xticks(np.arange(0, 1.1, 0.1))
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), fontsize=10, frameon=True, ncol=3)
ax1.grid(True, alpha=0.3, axis='x')

# MIDDLE PLOT - MLE AUCROC
color_aucroc = sns.color_palette("Set2", 2)[0]
ax2.barh(y_pos, mle_class_avg['MLE AUCROC'], bar_height * 2, 
        label='MLE AUCROC', color=color_aucroc, alpha=0.8, edgecolor='black', linewidth=0.5)

ax2.set_xlabel('MLE AUCROC', fontsize=11)
ax2.set_xlim(0-0.001, 1.001)
ax2.set_xticks(np.arange(0, 1.1, 0.1))
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), fontsize=10, frameon=True)
ax2.grid(True, alpha=0.3, axis='x')

# RIGHT PLOT - MLE RMSE
color_rmse = sns.color_palette("Set2", 2)[1]
ax3.barh(y_pos, mle_reg_avg['MLE RMSE'], bar_height * 2, 
        label='MLE RMSE', color=color_rmse, alpha=0.8, edgecolor='black', linewidth=0.5)

ax3.set_xlabel('MLE RMSE', fontsize=11)
max_rmse = mle_reg_avg['MLE RMSE'].max()
ax3.set_xlim(0-0.001, max_rmse * 1.05)
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), fontsize=10, frameon=True)
ax3.grid(True, alpha=0.3, axis='x')

plt.tight_layout()

# Save figure as PDF
plt.savefig(OUTPUT_PATH, format='pdf', bbox_inches='tight', dpi=300)
print(f"Plot saved to {OUTPUT_PATH}")



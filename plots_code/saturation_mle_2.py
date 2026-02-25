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

# Merge MLE data
mle_merged_df = mle_class_avg.merge(mle_reg_avg, on='method')

# Filter out GReaT and STaSy
mle_merged_df = mle_merged_df[~mle_merged_df['method'].isin(['GReaT', 'STaSy'])].reset_index(drop=True)

print(mle_merged_df)
# Invert the order of methods to match left plot
mle_merged_df = mle_merged_df.set_index('method').loc[merged_df['method']].reset_index()
print(mle_merged_df)
# Set seaborn style
print(merged_df)
sns.set_style("whitegrid")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

# LEFT PLOT
# Define colors for each metric
colors_left = sns.color_palette("husl", 3)

# Metrics to plot
metrics_left = ['LR Detection', 'Shape', 'Trend']

# Y positions
y_pos = np.arange(len(merged_df))
bar_height = 0.25

# Plot bars for each metric
for idx, metric in enumerate(metrics_left):
    offset = (idx - 1) * bar_height
    ax1.barh(y_pos + offset, merged_df[metric], bar_height, 
            label=metric, color=colors_left[idx], alpha=0.8, edgecolor='black', linewidth=0.5)

# Customize left plot
ax1.set_yticks(y_pos)
labels = [r"$\mathbf{" + label + "}$" if label == "TabPC" else label 
          for label in merged_df['method']]
ax1.set_yticklabels(labels, fontsize=10)
ax1.set_xlabel('Average score', fontsize=11)
ax1.set_xlim(0-0.001, 1.001)
ax1.set_xticks(np.arange(0, 1.1, 0.1))
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), fontsize=10, frameon=True, ncol=3)
ax1.grid(True, alpha=0.3, axis='x')

# RIGHT PLOT
# Define colors for MLE metrics
colors_right = sns.color_palette("Set2", 2)

# Metrics to plot
metrics_right = ['MLE AUCROC', 'MLE RMSE']

# Plot bars for each metric
for idx, metric in enumerate(metrics_right):
    offset = (idx - 0.5) * bar_height
    ax2.barh(y_pos + offset, mle_merged_df[metric], bar_height, 
            label=metric, color=colors_right[idx], alpha=0.8, edgecolor='black', linewidth=0.5)

# Customize right plot
ax2.set_xlabel('Average score', fontsize=11)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), fontsize=10, frameon=True, ncol=2)
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()

# Save figure as PDF
plt.savefig(OUTPUT_PATH, format='pdf', bbox_inches='tight', dpi=300)
print(f"Plot saved to {OUTPUT_PATH}")



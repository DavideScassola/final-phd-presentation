import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

OUTPUT_PATH = "images/saturation_overview.pdf"
FIGSIZE = (5, 5)

# left plot: shape, trend and lr detection
df_shape = pd.read_csv("tables/csv/shape.csv")
df_trend = pd.read_csv("tables/csv/trend.csv")
df_lr_detection = pd.read_csv("tables/csv/lr_detection.csv")

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

print(merged_df)

# Set seaborn style
sns.set_style("whitegrid")

# Create figure with single subplot
fig, ax = plt.subplots(figsize=FIGSIZE)

# Define colors for each metric
colors = sns.color_palette("husl", 3)

# Metrics to plot
metrics = ['LR Detection', 'Shape', 'Trend']

# Y positions
y_pos = np.arange(len(merged_df))
bar_height = 0.25

# Plot bars for each metric
for idx, metric in enumerate(metrics):
    offset = (idx - 1) * bar_height
    ax.barh(y_pos + offset, merged_df[metric], bar_height, 
            label=metric, color=colors[idx], alpha=0.8, edgecolor='black', linewidth=0.5)

# Customize plot
ax.set_yticks(y_pos)
labels = [r"$\mathbf{" + label + "}$" if label == "TabPC" else label 
          for label in merged_df['method']]
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Average score', fontsize=11)
ax.set_xlim(0-0.001, 1.001)
ax.set_xticks(np.arange(0, 1.1, 0.1))
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), fontsize=10, frameon=True, ncol=3)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()

# Save figure as PDF
plt.savefig(OUTPUT_PATH, format='pdf', bbox_inches='tight', dpi=300)
print(f"Plot saved to {OUTPUT_PATH}")



import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

USE_SYMBOLS = True  # default marker for all points

def plot_single(input_csv: str, output_pdf: str, x_label: str, x_lim: tuple = None):
    """Plot a single CSV file to its own output file"""
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Create figure with single subplot
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Remove horizontal gridlines, keep only vertical
    ax.yaxis.grid(False)
    
    # Jitter amount for y-axis
    jitter_strength = 1e-4
    
    # Define markers for each dataset
    markers = ['o', (4,0,45), 'd', 'X', '*', (4,1,0), 'P']
    
    df = pd.read_csv(input_csv)
    
    # Get methods in original order and reverse it
    avg_data = df[df['dataset'] == 'Average'][['method', 'mean']].copy()
    avg_data['mean'] = pd.to_numeric(avg_data['mean'], errors='coerce')
    methods_sorted = avg_data['method'].tolist()
    methods_sorted = methods_sorted[::-1]  # Invert the order
    
    # Get unique datasets
    datasets = df['dataset'].unique()
    datasets_to_plot = [d for d in datasets if d != 'Average']
    
    # First, plot average bars (excluding GReaT and STaSy) - these go behind
    avg_dataset_data = df[df['dataset'] == 'Average']
    x_avg_values = []
    y_avg_positions = []
    
    for method_idx, method in enumerate(methods_sorted):
        # Skip GReaT and STaSy
        if method in ['GReaT', 'STaSy']:
            continue
        
        method_data = avg_dataset_data[avg_dataset_data['method'] == method]
        if not method_data.empty and pd.notna(method_data['mean'].values[0]):
            x_avg_values.append(float(method_data['mean'].values[0]))
            y_avg_positions.append(method_idx)
    
    # Plot average bars with black color behind everything
    if x_avg_values:
        ax.barh(y_avg_positions, x_avg_values,
               height=0.6,
               color='black',
               alpha=0.3,
               edgecolor='none',
               zorder=1,  # Behind the markers
               label='Average')
    
    # Plot each dataset
    for dataset_idx, dataset in enumerate(datasets_to_plot):
        dataset_data = df[df['dataset'] == dataset]
        
        x_values = []
        y_values = []
        
        for method_idx, method in enumerate(methods_sorted):
            method_data = dataset_data[dataset_data['method'] == method]
            if not method_data.empty and pd.notna(method_data['mean'].values[0]):
                x_values.append(float(method_data['mean'].values[0]))
                jitter = np.random.normal(0, jitter_strength)
                y_values.append(method_idx + jitter)
        
        # Plot with markers
        scatter_kwargs = {
            'label': dataset,
            's': 100,
            'alpha': 0.75,
            'edgecolors': 'black',
            'linewidth': 0.5,
            'zorder': 5  # On top of bars
        }
        
        if USE_SYMBOLS:
            scatter_kwargs['marker'] = markers[dataset_idx % len(markers)]
        
        ax.scatter(x_values, y_values, **scatter_kwargs)
    
    # Set y-axis labels with bold for TabPC
    ax.set_yticks(range(len(methods_sorted)))
    labels = [r"$\mathbf{" + label + "}$" if label == "TabPC" else label 
              for label in methods_sorted]
    ax.set_yticklabels(labels, fontsize=10)
    
    # Set x-axis range and ticks
    if x_lim:
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_xticks(np.arange(x_lim[0], x_lim[1] + 0.05, 0.1))
    else:
        border = 0.02
        ax.set_xlim(-border, 1 + border)
        ax.set_xticks(np.arange(0, 1.1, 0.1))
    
    ax.tick_params(axis='x', labelsize=10)
    
    # Labels
    ax.set_xlabel(x_label, fontsize=11)
    ax.grid(True, alpha=0.2, axis='x')
    
    # Add legend in top right corner
    ax.legend(title='Dataset', loc='upper right', fontsize=9, title_fontsize=10, framealpha=1.0)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_pdf}")
    plt.close()

def main():
    plot_single("tables/csv/xgb_detection.csv", 
                "images/xgb.pdf",
                "XGBoost Detection Score")
    
    plot_single("tables/csv/nmi_l1_weighted.csv",
                "images/nmi.pdf",
                "wNMIE Score",
                x_lim=(0, 0.4))

if __name__ == "__main__":
    main()
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

USE_SYMBOLS = True  # default marker for all points

def main(input_csvs: list, output_pdf: str, x_labels: list):
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(9, 6), sharey=True)
    
    # Get color palette
    colors = sns.color_palette("husl", 10)
    
    # Store all methods across both CSVs to ensure same y-axis
    all_methods = []
    
    # First pass: collect all methods to determine shared y-axis
    for input_csv in input_csvs:
        df = pd.read_csv(input_csv)
        avg_data = df[df['dataset'] == 'Average'][['method', 'mean']].copy()
        avg_data['mean'] = pd.to_numeric(avg_data['mean'], errors='coerce')
        methods = avg_data['method'].tolist()
        for method in methods:
            if method not in all_methods:
                all_methods.append(method)
    
    # Plot each CSV in its own subplot
    for idx, (input_csv, x_label, ax) in enumerate(zip(input_csvs, x_labels, axes)):
        df = pd.read_csv(input_csv)
        
        # Get methods in original order and reverse it
        avg_data = df[df['dataset'] == 'Average'][['method', 'mean']].copy()
        avg_data['mean'] = pd.to_numeric(avg_data['mean'], errors='coerce')
        methods_sorted = avg_data['method'].tolist()
        methods_sorted = methods_sorted[::-1]  # Invert the order
        
        # Get unique datasets
        datasets = df['dataset'].unique()
        datasets_to_plot = [d for d in datasets if d != 'Average']
        
        # Add 'Average' as a dataset for methods that are not GReaT or STaSy
        has_average = any(method not in ['GReaT', 'STaSy'] for method in methods_sorted)
        if has_average:
            datasets_to_plot = list(datasets_to_plot) + ['Average']
        
        # Bar width and spacing
        bar_height = 0.8 / len(datasets_to_plot)
        
        # Plot each dataset
        for dataset_idx, dataset in enumerate(datasets_to_plot):
            if dataset == 'Average':
                # Plot average bars (excluding GReaT and STaSy)
                dataset_data = df[df['dataset'] == 'Average']
                x_values = []
                y_positions = []
                
                for method_idx, method in enumerate(methods_sorted):
                    if method not in ['GReaT', 'STaSy']:
                        method_data = dataset_data[dataset_data['method'] == method]
                        if not method_data.empty and pd.notna(method_data['mean'].values[0]):
                            x_values.append(float(method_data['mean'].values[0]))
                            # Calculate y position for this bar
                            y_pos = method_idx + (dataset_idx - len(datasets_to_plot)/2 + 0.5) * bar_height
                            y_positions.append(y_pos)
                
                # Plot average bars
                ax.barh(y_positions, x_values,
                       height=bar_height,
                       label='Average',
                       color='black',
                       alpha=0.6,
                       edgecolor='black',
                       linewidth=0.5)
            else:
                # Plot regular dataset bars
                dataset_data = df[df['dataset'] == dataset]
                
                x_values = []
                y_positions = []
                
                for method_idx, method in enumerate(methods_sorted):
                    method_data = dataset_data[dataset_data['method'] == method]
                    if not method_data.empty and pd.notna(method_data['mean'].values[0]):
                        x_values.append(float(method_data['mean'].values[0]))
                        # Calculate y position for this bar
                        y_pos = method_idx + (dataset_idx - len(datasets_to_plot)/2 + 0.5) * bar_height
                        y_positions.append(y_pos)
                
                # Plot horizontal bars
                ax.barh(y_positions, x_values, 
                       height=bar_height,
                       label=dataset,
                       color=colors[dataset_idx % len(colors)],
                       alpha=0.8,
                       edgecolor='black',
                       linewidth=0.5)
        
        # Set y-axis labels with bold for TabPC (only on left subplot)
        if idx == 0:
            ax.set_yticks(range(len(methods_sorted)))
            # Make TabPC label bold
            labels = [r"$\mathbf{" + label + "}$" if label == "TabPC" else label 
                      for label in methods_sorted]
            ax.set_yticklabels(labels, fontsize=10)
        
        # Set x-axis range and ticks
        border = 0.05
        if idx == 0:
            ax.set_xlim(-border, 1 + border)
            ax.set_xticks(np.arange(0, 1.1, 0.1))
            ax.tick_params(axis='x', labelsize=10)
        # else:
        #     ax.set_xlim(-border, 0.3 + border)
        #     ax.set_xticks(np.arange(0, 0.5, 0.1))
        #     ax.tick_params(axis='x', labelsize=10)
        
        # Labels
        ax.set_xlabel(x_label, fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')
    
    # Create a single legend below both plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Dataset', loc='lower center', 
               ncol=len(labels), frameon=True, fancybox=True, shadow=True, 
               fontsize=9, title_fontsize=10, bbox_to_anchor=(0.55, -0.05))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    
    # Determine output path
    if output_pdf:
        output_path = output_pdf
    else:
        output_path = Path("images/combined_plot.pdf")
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    main(input_csvs=["tables/csv/xgb_detection.csv", "tables/csv/nmi_l1_weighted.csv"],
         output_pdf="images/xgb_nmi.pdf",
         x_labels=["XGBoost Detection Score", "wNMIE Score"])
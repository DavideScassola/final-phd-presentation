import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import matplotlib.cm as cm
import math

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot XGB C2ST performance vs training time')
    parser.add_argument('--xgb_file', type=str, default='tables/csv/xgb_detection.csv',
                        help='Path to XGB detection CSV file')
    parser.add_argument('--time_file', type=str, default='tables/csv/training_time.csv',
                        help='Path to training time CSV file')
    parser.add_argument('--output', type=str, default=None, help='Output file path (optional)')
    args = parser.parse_args()
    
    # Read CSV files
    xgb_df = pd.read_csv(args.xgb_file)
    time_df = pd.read_csv(args.time_file)
    parameters_df = pd.read_csv('tables/csv/num_parameters.csv')
    
    # Filter for 'Average' dataset only
    xgb_avg = xgb_df[xgb_df['dataset'] == 'Average'][['method', 'mean']].copy()
    time_avg = time_df[time_df['dataset'] == 'Average'][['method', 'mean']].copy()
    params_avg = parameters_df[parameters_df['dataset'] == 'Average'][['method', 'mean']].copy()
    
    # Rename columns for clarity
    xgb_avg.rename(columns={'mean': 'xgb_c2st'}, inplace=True)
    time_avg.rename(columns={'mean': 'training_time'}, inplace=True)
    params_avg.rename(columns={'mean': 'num_parameters'}, inplace=True)
    
    # Merge dataframes
    merged_df = xgb_avg.merge(time_avg, on='method').merge(params_avg, on='method')
    
    # Convert to numeric, handling any non-numeric values
    merged_df['xgb_c2st'] = pd.to_numeric(merged_df['xgb_c2st'], errors='coerce')
    merged_df['training_time'] = pd.to_numeric(merged_df['training_time'], errors='coerce')
    merged_df['num_parameters'] = pd.to_numeric(merged_df['num_parameters'], errors='coerce')
    
    # Convert training time from seconds to minutes
    merged_df['training_time'] = merged_df['training_time'] / 60
    
    # Remove rows with NaN values
    merged_df = merged_df.dropna()
    
    # Remove STaSy and GReaT from the plot
    merged_df = merged_df[~merged_df['method'].isin(['STaSy', 'GReaT'])]
    
    # Color by log10(number of parameters)
    cmap = plt.get_cmap("RdYlGn_r")  # low=green, high=red
    norm = plt.matplotlib.colors.LogNorm(vmin=merged_df['num_parameters'].min(),
                                         vmax=merged_df['num_parameters'].max())
    
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Create figure with larger size and more padding
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Plot each model with a different color and add method name as label
    label_offsets = {
        "Fully Factorized": (-12, -12),  # move below
        #"STaSy": (-12, -12),
        "CTGAN": (-12, -12), # move below
    }

    for idx, (_, row) in enumerate(merged_df.iterrows()):
        # Plot just a circle marker
        ax.scatter(row['training_time'], row['xgb_c2st'], 
                   s=50, 
                   alpha=0.7,
                   color=cmap(norm(row['num_parameters'])),  # Use raw num_parameters instead of log10_params
                   edgecolors='black', 
                   linewidth=1)
        
        # Add method name as label next to the point
        offset = label_offsets.get(row['method'], (-12, 8))
        # Make TabPC label bold
        fontweight = 'bold' if row['method'] == 'TabPC' else 'normal'
        ax.annotate(row['method'], 
                    (row['training_time'], row['xgb_c2st']),
                    xytext=offset,
                    textcoords='offset points',
                    fontsize=9,
                    fontweight=fontweight,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.2, edgecolor='none'))
    
    # Add colorbar for log10(num_parameters)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Average #parameters', fontsize=10)  # Update label
    
    # Labels and formatting
    ax.set_xlabel('Average Training Time (minutes)', fontsize=12)
    ax.set_ylabel('Average XGB C2ST', fontsize=12)
    #ax.set_title('XGB C2ST Performance vs Training Time', fontsize=13, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.2)
    
    # Set log scale for x-axis (training time varies greatly)
    ax.set_xscale('log')
    
    # Add more margin around the plot to ensure labels fit
    ax.margins(x=0.15, y=0.15)
    
    plt.tight_layout(pad=1.5)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path('images/tabpc_overview.pdf')
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    print(f"Plot saved to {output_path}")
    print("\nData plotted:")
    print(merged_df.to_string(index=False))

if __name__ == "__main__":
    main()
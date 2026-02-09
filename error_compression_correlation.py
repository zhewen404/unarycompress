import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set matplotlib style to match plot.py
plt.rcParams.update({
    'font.family': 'Ubuntu',
    'font.size': 15,
    'axes.linewidth': 1,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
})

# Read the CSV data
df = pd.read_csv('table_35.csv', sep='\t')

# Clean column names by stripping whitespace
df.columns = df.columns.str.strip()

print("Available columns:", df.columns.tolist())

# Create output directory for correlation plots
output_dir = 'compression_correlation'
os.makedirs(output_dir, exist_ok=True)

# Calculate 1/Comp_Ratio
df['Inv_Comp_Ratio'] = 1.0 / df['Comp_Ratio']

# Define the error metrics to plot
error_metrics = ['SCC_Err_Avg', 'ZCE_Err_Avg', 'Acc_Err_Avg', 'AND_Err_Avg']

# Filter data for each method
df_hamming = df[df['Method'] == 'Hamming']
df_value = df[df['Method'] == 'Value']

# Create correlation plots for each error metric
for metric in error_metrics:
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Hamming subplot
    ax1.scatter(df_hamming['Inv_Comp_Ratio'], df_hamming[metric], alpha=0.7, s=50)
    # Add labels to each point
    for i, row in df_hamming.iterrows():
        label = f"{int(row['Codebook_Size'])}_{int(row['Chunk_Size'])}"
        ax1.annotate(label, (row['Inv_Comp_Ratio'], row[metric]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
    ax1.set_xlabel('Compression Ratio', fontsize=15)
    if metric == "Acc_Err_Avg": 
        metric_ = "SA Loss"
    else: metric_ = metric
    ax1.set_ylabel(f'{metric_}', fontsize=15)
    ax1.set_title('Hamming', fontsize=15)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=13)
    
    # Value subplot
    ax2.scatter(df_value['Inv_Comp_Ratio'], df_value[metric], alpha=0.7, s=50, color='orange')
    # Add labels to each point
    for i, row in df_value.iterrows():
        label = f"{int(row['Codebook_Size'])}_{int(row['Chunk_Size'])}"
        ax2.annotate(label, (row['Inv_Comp_Ratio'], row[metric]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
    ax2.set_xlabel('Compression Ratio', fontsize=15)
    if metric == "Acc_Err_Avg": 
        metric_ = "SA Loss"
    else: metric_ = metric
    ax2.set_ylabel(f'{metric_}', fontsize=15)
    ax2.set_title('Value', fontsize=15)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=13)
    
    # Make y-axis scales consistent
    y_min = min(df_hamming[metric].min(), df_value[metric].min())
    y_max = max(df_hamming[metric].max(), df_value[metric].max())
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    
    # Make x-axis scales consistent
    x_min = min(df_hamming['Inv_Comp_Ratio'].min(), df_value['Inv_Comp_Ratio'].min())
    x_max = max(df_hamming['Inv_Comp_Ratio'].max(), df_value['Inv_Comp_Ratio'].max())
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{metric}_vs_inv_comp_ratio.pdf', bbox_inches='tight', 
                pad_inches=0.05, facecolor='white')
    plt.close()
    
    # Also create separate individual plots
    # Hamming individual plot
    fig_h = plt.figure(figsize=(3.3115, 2))
    plt.scatter(df_hamming['Inv_Comp_Ratio'], df_hamming[metric], alpha=0.7, s=50)
    # Add labels to each point
    for i, row in df_hamming.iterrows():
        label = f"{int(row['Codebook_Size'])}_{int(row['Chunk_Size'])}"
        plt.annotate(label, (row['Inv_Comp_Ratio'], row[metric]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
    plt.xlabel('Compression Ratio', fontsize=15)
    if metric == "Acc_Err_Avg": 
        metric_ = "SA Loss"
    else: metric_ = metric
    plt.ylabel(f'{metric_}', fontsize=15)
    # plt.title('Hamming Method', fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=13)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{metric}_vs_inv_comp_ratio_hamming.pdf', bbox_inches='tight', 
                pad_inches=0.05, facecolor='white')
    plt.close()
    
    # Value individual plot
    fig_v = plt.figure(figsize=(3.3115, 2))
    plt.scatter(df_value['Inv_Comp_Ratio'], df_value[metric], alpha=0.7, s=50, color='orange')
    # Add labels to each point
    for i, row in df_value.iterrows():
        label = f"{int(row['Codebook_Size'])}_{int(row['Chunk_Size'])}"
        plt.annotate(label, (row['Inv_Comp_Ratio'], row[metric]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
    plt.xlabel('Compression Ratio', fontsize=15)
    if metric == "Acc_Err_Avg": 
        metric_ = "SA Loss"
    else: metric_ = metric
    plt.ylabel(f'{metric_}', fontsize=15)
    # plt.title('Value Method', fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=13)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{metric}_vs_inv_comp_ratio_value.pdf', bbox_inches='tight', 
                pad_inches=0.05, facecolor='white')
    plt.close()
    
    print(f"Saved {metric} vs Compression Ratio correlation plots")

print("All correlation plots generated successfully!")
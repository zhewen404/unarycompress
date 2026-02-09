from compdecomp import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import seaborn as sns
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os

# Set matplotlib font sizes
plt.rcParams.update({'font.size': 15})

L = 256
TRAILS = 10
GRID_SIZE = 50  # Number of bins for density grid
CHUNK_SIZE = 500  # Process in chunks to avoid memory issues
MAX_WORKERS = min(22, mp.cpu_count())  # Limit workers to avoid memory overload

def print_dict_results(results):
    for key, value in results.items():
        print(f"{key}: {value}")

def process_chunk(chunk_data):
    """Process a chunk of (a,b) pairs and return results for all codecs"""
    codec_dicthamming_4_8 = DictCodec(8,4,True)
    codec_dictvalue_4_8 = DictCodec(8,4,False)
    # codec_dicthamming_4_16 = DictCodec(16,4,True)
    # codec_dictvalue_4_16 = DictCodec(16,4,False)

    # codec_dicthamming_8_8 = DictCodec(8,8,True)
    # codec_dictvalue_8_8 = DictCodec(8,8,False)
    codec_dicthamming_8_16 = DictCodec(16,8,True)
    codec_dictvalue_8_16 = DictCodec(16,8,False)
    codec_dicthamming_8_32 = DictCodec(32,8,True)
    codec_dictvalue_8_32 = DictCodec(32,8,False)

    # codec_dicthamming_16_16 = DictCodec(16,16,True)
    # codec_dictvalue_16_16 = DictCodec(16,16,False)
    codec_dicthamming_16_32 = DictCodec(32,16,True)
    codec_dictvalue_16_32 = DictCodec(32,16,False)
    codec_dicthamming_16_64 = DictCodec(64,16,True)
    codec_dictvalue_16_64 = DictCodec(64,16,False)


    codec_arr = [codec_dicthamming_4_8, codec_dictvalue_4_8,
                codec_dicthamming_8_16, codec_dictvalue_8_16, codec_dicthamming_8_32, codec_dictvalue_8_32,
                codec_dicthamming_16_32, codec_dictvalue_16_32, codec_dicthamming_16_64, codec_dictvalue_16_64]

    
    chunk_results = {}
    for codec in codec_arr:
        codec_name = codec.get_name()
        chunk_results[codec_name] = {
            'scc_input': [], 'scc_output': [],
            'zce_input': [], 'zce_output': [],
            'and_input': [], 'and_output': []
        }
    
    unco = BinaryUncorrelated()
    
    for a, b in chunk_data:
        for trial in range(TRAILS):
            a_in, b_in = unco.gen(a, b, L)
            for codec in codec_arr:
                # if codec.get_name() == "Dict_8_4_Hamming" or codec.get_name() == "Dict_8_4_Value":
                #     print(codec.get_name())
                #     print(codec.get_codebook())  # Print the codebook for Dict_8_4_Hamming
                
                results = codec.evaluate_all(a_in, b_in)
                codec_name = codec.get_name()
                
                # Store SCC, ZCE, and AND data
                input_scc = results["scc"][0]
                output_scc = results["scc"][1]
                input_zce = results["zce"][0]
                output_zce = results["zce"][1]
                input_and = results["and"][0]
                output_and = results["and"][1]
                
                chunk_results[codec_name]['scc_input'].append(input_scc)
                chunk_results[codec_name]['scc_output'].append(output_scc)
                chunk_results[codec_name]['zce_input'].append(input_zce)
                chunk_results[codec_name]['zce_output'].append(output_zce)
                chunk_results[codec_name]['and_input'].append(input_and)
                chunk_results[codec_name]['and_output'].append(output_and)
    
    return chunk_results

# Create chunks of (a,b) pairs to process
def create_chunks(L, chunk_size):
    """Create chunks of (a,b) pairs"""
    all_pairs = [(a, b) for a in range(0, L+1) for b in range(0, L+1)]
    for i in range(0, len(all_pairs), chunk_size):
        yield all_pairs[i:i + chunk_size]

# Storage for all codec results
codec_dicthamming_4_8 = DictCodec(8,4,True)
codec_dictvalue_4_8 = DictCodec(8,4,False)
# codec_dicthamming_4_16 = DictCodec(16,4,True)
# codec_dictvalue_4_16 = DictCodec(16,4,False)

# codec_dicthamming_8_8 = DictCodec(8,8,True)
# codec_dictvalue_8_8 = DictCodec(8,8,False)
codec_dicthamming_8_16 = DictCodec(16,8,True)
codec_dictvalue_8_16 = DictCodec(16,8,False)
codec_dicthamming_8_32 = DictCodec(32,8,True)
codec_dictvalue_8_32 = DictCodec(32,8,False)

# codec_dicthamming_16_16 = DictCodec(16,16,True)
# codec_dictvalue_16_16 = DictCodec(16,16,False)
codec_dicthamming_16_32 = DictCodec(32,16,True)
codec_dictvalue_16_32 = DictCodec(32,16,False)
codec_dicthamming_16_64 = DictCodec(64,16,True)
codec_dictvalue_16_64 = DictCodec(64,16,False)


codec_arr = [codec_dicthamming_4_8, codec_dictvalue_4_8,
             codec_dicthamming_8_16, codec_dictvalue_8_16, codec_dicthamming_8_32, codec_dictvalue_8_32,
             codec_dicthamming_16_32, codec_dictvalue_16_32, codec_dicthamming_16_64, codec_dictvalue_16_64]

comp_ratio_arr = [codec.get_compression_ratio() for codec in codec_arr]
codec_data = {}
for codec in codec_arr:
    codec_name = codec.get_name()
    codec_data[codec_name] = {
        'scc_input': [], 'scc_output': [],
        'zce_input': [], 'zce_output': [],
        'and_input': [], 'and_output': []
    }

total_pairs = (L+1)**2
print(f"Running {TRAILS} trials for each (a,b) pair...")
print(f"Total pairs: {total_pairs:,}, Processing in chunks of {CHUNK_SIZE}")
print(f"Using {MAX_WORKERS} parallel workers...")

# Process chunks in parallel
with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    chunk_generator = create_chunks(L, CHUNK_SIZE)
    
    for i, chunk_results in enumerate(executor.map(process_chunk, chunk_generator)):
        # Accumulate results from this chunk
        for codec_name in codec_data.keys():
            codec_data[codec_name]['scc_input'].extend(chunk_results[codec_name]['scc_input'])
            codec_data[codec_name]['scc_output'].extend(chunk_results[codec_name]['scc_output'])
            codec_data[codec_name]['zce_input'].extend(chunk_results[codec_name]['zce_input'])
            codec_data[codec_name]['zce_output'].extend(chunk_results[codec_name]['zce_output'])
            codec_data[codec_name]['and_input'].extend(chunk_results[codec_name]['and_input'])
            codec_data[codec_name]['and_output'].extend(chunk_results[codec_name]['and_output'])
        
        # Progress update
        processed_pairs = len(codec_data[list(codec_data.keys())[0]]['scc_input']) // TRAILS
        if (i + 1) % 10 == 0:
            print(f"Processed {processed_pairs:,}/{total_pairs:,} pairs ({100*processed_pairs/total_pairs:.1f}%)")

print(f"Completed processing {len(codec_data[list(codec_data.keys())[0]]['scc_input']):,} data points")

# Calculate global min/max for consistent axes
global_scc_min = float('inf')
global_scc_max = float('-inf')
global_zce_min = float('inf')
global_zce_max = float('-inf')
global_and_min = float('inf')
global_and_max = float('-inf')

for codec in codec_arr:
    codec_name = codec.get_name()
    scc_inputs = codec_data[codec_name]['scc_input']
    scc_outputs = codec_data[codec_name]['scc_output']
    zce_inputs = codec_data[codec_name]['zce_input']
    zce_outputs = codec_data[codec_name]['zce_output']
    and_inputs = codec_data[codec_name]['and_input']
    and_outputs = codec_data[codec_name]['and_output']
    
    if scc_inputs and scc_outputs:
        global_scc_min = min(global_scc_min, min(scc_inputs), min(scc_outputs))
        global_scc_max = max(global_scc_max, max(scc_inputs), max(scc_outputs))
    
    if zce_inputs and zce_outputs:
        global_zce_min = min(global_zce_min, min(zce_inputs), min(zce_outputs))
        global_zce_max = max(global_zce_max, max(zce_inputs), max(zce_outputs))
    
    if and_inputs and and_outputs:
        global_and_min = min(global_and_min, min(and_inputs), min(and_outputs))
        global_and_max = max(global_and_max, max(and_inputs), max(and_outputs))

print(f"Global SCC range: [{global_scc_min:.3f}, {global_scc_max:.3f}]")
print(f"Global ZCE range: [{global_zce_min:.3f}, {global_zce_max:.3f}]")
print(f"Global AND range: [{global_and_min:.3f}, {global_and_max:.3f}]")

# Create output directory if it doesn't exist
output_dir = 'correlation_sensitivity'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Calculate average distances and organize data for heatmaps
scc_data = []
zce_data = []
and_data = []

for codec in codec_arr:
    codec_name = codec.get_name()
    scc_inputs = codec_data[codec_name]['scc_input']
    scc_outputs = codec_data[codec_name]['scc_output']
    zce_inputs = codec_data[codec_name]['zce_input']
    zce_outputs = codec_data[codec_name]['zce_output']
    and_inputs = codec_data[codec_name]['and_input']
    and_outputs = codec_data[codec_name]['and_output']
    
    # Calculate average distances
    scc_distances = [abs(scc_outputs[j] - scc_inputs[j]) for j in range(len(scc_inputs))]
    zce_distances = [abs(zce_outputs[j] - zce_inputs[j]) for j in range(len(zce_inputs))]
    and_distances = [abs(and_outputs[j] - and_inputs[j]) for j in range(len(and_inputs))]
    
    avg_scc_distance = np.mean(scc_distances)
    avg_zce_distance = np.mean(zce_distances)
    avg_and_distance = np.mean(and_distances)
    
    # Parse codec name to extract parameters
    # Format: "Dict_CodebookSize_ChunkSize_Method"
    parts = codec_name.split('_')
    codebook_size = int(parts[1])
    chunk_size = int(parts[2])
    method = parts[3]
    
    scc_data.append({
        'codebook_size': codebook_size,
        'chunk_size': chunk_size,
        'method': method,
        'avg_distance': avg_scc_distance
    })
    
    zce_data.append({
        'codebook_size': codebook_size,
        'chunk_size': chunk_size,
        'method': method,
        'avg_distance': avg_zce_distance
    })
    
    and_data.append({
        'codebook_size': codebook_size,
        'chunk_size': chunk_size,
        'method': method,
        'avg_distance': avg_and_distance
    })

# Create DataFrames
scc_df = pd.DataFrame(scc_data)
zce_df = pd.DataFrame(zce_data)
and_df = pd.DataFrame(and_data)

# Filter data by method
scc_hamming_df = scc_df[scc_df['method'] == 'Hamming']
scc_value_df = scc_df[scc_df['method'] == 'Value']
zce_hamming_df = zce_df[zce_df['method'] == 'Hamming']
zce_value_df = zce_df[zce_df['method'] == 'Value']
and_hamming_df = and_df[and_df['method'] == 'Hamming']
and_value_df = and_df[and_df['method'] == 'Value']

# Create pivot tables for heatmaps
scc_hamming_pivot = scc_hamming_df.pivot(index='chunk_size', columns='codebook_size', values='avg_distance')
scc_value_pivot = scc_value_df.pivot(index='chunk_size', columns='codebook_size', values='avg_distance')
zce_hamming_pivot = zce_hamming_df.pivot(index='chunk_size', columns='codebook_size', values='avg_distance')
zce_value_pivot = zce_value_df.pivot(index='chunk_size', columns='codebook_size', values='avg_distance')
and_hamming_pivot = and_hamming_df.pivot(index='chunk_size', columns='codebook_size', values='avg_distance')
and_value_pivot = and_value_df.pivot(index='chunk_size', columns='codebook_size', values='avg_distance')

# Calculate global min/max for consistent color scaling in heatmaps
scc_heatmap_global_min = min(scc_hamming_pivot.min().min(), scc_value_pivot.min().min())
scc_heatmap_global_max = max(scc_hamming_pivot.max().max(), scc_value_pivot.max().max())
zce_heatmap_global_min = min(zce_hamming_pivot.min().min(), zce_value_pivot.min().min())
zce_heatmap_global_max = max(zce_hamming_pivot.max().max(), zce_value_pivot.max().max())
and_heatmap_global_min = min(and_hamming_pivot.min().min(), and_value_pivot.min().min())
and_heatmap_global_max = max(and_hamming_pivot.max().max(), and_value_pivot.max().max())

# Create SCC Hamming heatmap
fig, ax = plt.subplots(1, 1, figsize=(5,2.5))
sns.heatmap(scc_hamming_pivot, annot=True, fmt='.3f', cmap='viridis', 
            vmin=0, vmax=scc_heatmap_global_max, ax=ax, 
            #cbar_kws={'label': 'Average Distance'}, annot_kws={'size': 15}
            )
#ax.set_title(f'SCC Hamming Method - Average Distance\nL={L}', fontsize=15)
ax.set_xlabel('Codebook Size', fontsize=15)
ax.set_ylabel('Chunk Size', fontsize=15)
plt.tight_layout(pad=2.0)
scc_hamming_filename = f'scc_sensitivity_heatmap_hamming_L{L}_T{TRAILS}'
plt.savefig(f'{output_dir}/{scc_hamming_filename}.pdf', bbox_inches='tight')
print(f"SCC Hamming heatmap saved as '{output_dir}/{scc_hamming_filename}.pdf'")
# plt.show()

# Create SCC Value heatmap
fig, ax = plt.subplots(1, 1, figsize=(5,2.5))
sns.heatmap(scc_value_pivot, annot=True, fmt='.3f', cmap='viridis',
            vmin=0, vmax=scc_heatmap_global_max, ax=ax, 
            #cbar_kws={'label': 'Average Distance'}, annot_kws={'size': 15}
            )
#ax.set_title(f'SCC Value Method - Average Distance\nL={L}', fontsize=15)
ax.set_xlabel('Codebook Size', fontsize=15)
ax.set_ylabel('Chunk Size', fontsize=15)
plt.tight_layout(pad=2.0)
scc_value_filename = f'scc_sensitivity_heatmap_value_L{L}_T{TRAILS}'
plt.savefig(f'{output_dir}/{scc_value_filename}.pdf', bbox_inches='tight')
print(f"SCC Value heatmap saved as '{output_dir}/{scc_value_filename}.pdf'")
# plt.show()

# Create ZCE Hamming heatmap
fig, ax = plt.subplots(1, 1, figsize=(5,2.5))
sns.heatmap(zce_hamming_pivot, annot=True, fmt='.3f', cmap='plasma',
            vmin=0, vmax=zce_heatmap_global_max, ax=ax, 
            #cbar_kws={'label': 'Average Distance'}, annot_kws={'size': 15}
            )
#ax.set_title(f'ZCE Hamming Method - Average Distance\nL={L}, Trials={TRAILS}', fontsize=15)
ax.set_xlabel('Codebook Size', fontsize=15)
ax.set_ylabel('Chunk Size', fontsize=15)
plt.tight_layout(pad=2.0)
zce_hamming_filename = f'zce_sensitivity_heatmap_hamming_L{L}_T{TRAILS}'
plt.savefig(f'{output_dir}/{zce_hamming_filename}.pdf', bbox_inches='tight')
print(f"ZCE Hamming heatmap saved as '{output_dir}/{zce_hamming_filename}.pdf'")
# plt.show()

# Create ZCE Value heatmap
fig, ax = plt.subplots(1, 1, figsize=(5,2.5))
sns.heatmap(zce_value_pivot, annot=True, fmt='.3f', cmap='plasma',
            vmin=0, vmax=zce_heatmap_global_max, ax=ax, 
            #cbar_kws={'label': 'Average Distance'}, annot_kws={'size': 15}
            )
#ax.set_title(f'ZCE Value Method - Average Distance\nL={L}, Trials={TRAILS}', fontsize=15)
ax.set_xlabel('Codebook Size', fontsize=15)
ax.set_ylabel('Chunk Size', fontsize=15)
plt.tight_layout(pad=2.0)
zce_value_filename = f'zce_sensitivity_heatmap_value_L{L}_T{TRAILS}'
plt.savefig(f'{output_dir}/{zce_value_filename}.pdf', bbox_inches='tight')
print(f"ZCE Value heatmap saved as '{output_dir}/{zce_value_filename}.pdf'")
# plt.show()

# Create AND Hamming heatmap
fig, ax = plt.subplots(1, 1, figsize=(5,2.5))
sns.heatmap(and_hamming_pivot, annot=True, fmt='.3f', cmap='coolwarm',
            vmin=0, vmax=and_heatmap_global_max, ax=ax, 
            #cbar_kws={'label': 'Average Distance'}, annot_kws={'size': 15}
            )
#ax.set_title(f'AND Hamming Method - Average Distance\nL={L}, Trials={TRAILS}', fontsize=15)
ax.set_xlabel('Codebook Size', fontsize=15)
ax.set_ylabel('Chunk Size', fontsize=15)
plt.tight_layout(pad=2.0)
and_hamming_filename = f'and_sensitivity_heatmap_hamming_L{L}_T{TRAILS}'
plt.savefig(f'{output_dir}/{and_hamming_filename}.pdf', bbox_inches='tight')
print(f"AND Hamming heatmap saved as '{output_dir}/{and_hamming_filename}.pdf'")
# plt.show()

# Create AND Value heatmap
fig, ax = plt.subplots(1, 1, figsize=(5,2.5))
sns.heatmap(and_value_pivot, annot=True, fmt='.3f', cmap='coolwarm',
            vmin=0, vmax=and_heatmap_global_max, ax=ax, 
            #cbar_kws={'label': 'Average Distance'}, annot_kws={'size': 15}
            )
#ax.set_title(f'AND Value Method - Average Distance\nL={L}, Trials={TRAILS}', fontsize=15)
ax.set_xlabel('Codebook Size', fontsize=15)
ax.set_ylabel('Chunk Size', fontsize=15)
plt.tight_layout(pad=2.0)
and_value_filename = f'and_sensitivity_heatmap_value_L{L}_T{TRAILS}'
plt.savefig(f'{output_dir}/{and_value_filename}.pdf', bbox_inches='tight')
print(f"AND Value heatmap saved as '{output_dir}/{and_value_filename}.pdf'")
# plt.show()

# Create density plots (previous functionality) - separate for Hamming and Value
hamming_codecs = [codec for codec in codec_arr if 'Hamming' in codec.get_name()]
value_codecs = [codec for codec in codec_arr if 'Value' in codec.get_name()]

# Function to create density plots for a set of codecs
def create_plots(codecs, method_name, metric_name, global_min, global_max, colormap):
    n_codecs = len(codecs)
    cols = 3  # 3 columns
    rows = (n_codecs + cols - 1) // cols  # Calculate rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    if rows == 1 and n_codecs == 1:
        axes = [axes]  # Make it iterable for single subplot
    elif rows == 1:
        axes = axes  # Keep as is for single row
    else:
        axes = axes.flatten()

    for i, codec in enumerate(codecs):
        codec_name = codec.get_name()
        ax = axes[i]
        
        if metric_name == 'SCC':
            inputs = codec_data[codec_name]['scc_input']
            outputs = codec_data[codec_name]['scc_output']
        elif metric_name == 'ZCE':
            inputs = codec_data[codec_name]['zce_input']
            outputs = codec_data[codec_name]['zce_output']
        else:  # AND
            inputs = codec_data[codec_name]['and_input']
            outputs = codec_data[codec_name]['and_output']
        
        # Calculate average vertical distance
        vertical_distances = [abs(outputs[j] - inputs[j]) for j in range(len(inputs))]
        avg_distance = np.mean(vertical_distances)
        
        if inputs and outputs:
            # Create density grid using global range
            bins_x = np.linspace(global_min, global_max, GRID_SIZE)
            bins_y = np.linspace(global_min, global_max, GRID_SIZE)
            
            density, x_edges, y_edges = np.histogram2d(inputs, outputs, bins=[bins_x, bins_y])
            
            # Set zeros to NaN for transparent/white color instead of darkest
            density_processed = np.where(density == 0, np.nan, density)
            
            # Create heatmap
            im = ax.imshow(density_processed.T, origin='lower', 
                          extent=[global_min, global_max, global_min, global_max],
                          cmap=colormap, norm=LogNorm(vmin=1), aspect='equal')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Point Density (log scale)', fontsize=12)
            
            # Add perfect correlation line (y=x)
            ax.plot([global_min, global_max], [global_min, global_max], 'r--', alpha=0.7, linewidth=2, label='y=x')
        
        ax.set_title(f'{codec_name}\nAvg Distance: {avg_distance:.6f}', fontsize=15)
        ax.set_xlabel(f'Input {metric_name}', fontsize=15)
        ax.set_ylabel(f'Output {metric_name}', fontsize=15)
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots
    for i in range(len(codecs), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    
    # Save plots
    filename_base = f'uncoinput_{metric_name.lower()}_density_{method_name.lower()}_L{L}_T{TRAILS}_parallel'
    plt.savefig(f'{output_dir}/{filename_base}.pdf', bbox_inches='tight')
    print(f"{metric_name} {method_name} density plot saved as '{output_dir}/{filename_base}.pdf'")
    # plt.show()

# Create SCC plots for Hamming and Value separately
create_plots(hamming_codecs, 'Hamming', 'SCC', global_scc_min, global_scc_max, 'viridis')
create_plots(value_codecs, 'Value', 'SCC', global_scc_min, global_scc_max, 'viridis')

# Create ZCE plots for Hamming and Value separately  
create_plots(hamming_codecs, 'Hamming', 'ZCE', global_zce_min, global_zce_max, 'plasma')
create_plots(value_codecs, 'Value', 'ZCE', global_zce_min, global_zce_max, 'plasma')

# Create AND plots for Hamming and Value separately
create_plots(hamming_codecs, 'Hamming', 'AND', global_and_min, global_and_max, 'coolwarm')
create_plots(value_codecs, 'Value', 'AND', global_and_min, global_and_max, 'coolwarm')

# Prepare data for compression ratio correlation plots
compression_ratio_data = []
for i, codec in enumerate(codec_arr):
    codec_name = codec.get_name()
    scc_inputs = codec_data[codec_name]['scc_input']
    scc_outputs = codec_data[codec_name]['scc_output']
    zce_inputs = codec_data[codec_name]['zce_input']
    zce_outputs = codec_data[codec_name]['zce_output']
    and_inputs = codec_data[codec_name]['and_input']
    and_outputs = codec_data[codec_name]['and_output']
    
    scc_distances = [abs(scc_outputs[j] - scc_inputs[j]) for j in range(len(scc_inputs))]
    zce_distances = [abs(zce_outputs[j] - zce_inputs[j]) for j in range(len(zce_inputs))]
    and_distances = [abs(and_outputs[j] - and_inputs[j]) for j in range(len(and_inputs))]
    
    avg_scc_distance = np.mean(scc_distances)
    avg_zce_distance = np.mean(zce_distances)
    avg_and_distance = np.mean(and_distances)
    
    parts = codec_name.split('_')
    method = parts[3]
    
    compression_ratio_data.append({
        'codec_name': codec_name,
        'compression_ratio': comp_ratio_arr[i],
        'method': method,
        'avg_scc_distance': avg_scc_distance,
        'avg_zce_distance': avg_zce_distance,
        'avg_and_distance': avg_and_distance
    })

# Create correlation plots
cr_df = pd.DataFrame(compression_ratio_data)
cr_hamming = cr_df[cr_df['method'] == 'Hamming']
cr_value = cr_df[cr_df['method'] == 'Value']

# SCC vs Compression Ratio - Hamming
fig, ax = plt.subplots(1, 1, figsize=(3.3115, 2))
ax.scatter(cr_hamming['compression_ratio'], cr_hamming['avg_scc_distance'], s=100, alpha=0.7, label='Hamming')
for idx, row in cr_hamming.iterrows():
    ax.annotate(row['codec_name'].replace('Dict_', '').replace('_Hamming', ''), 
                 (row['compression_ratio'], row['avg_scc_distance']), 
                 fontsize=8, alpha=0.7)
ax.set_xlabel('Compression Ratio', fontsize=15)
ax.set_ylabel('Average SCC Distance', fontsize=15)
# ax.set_title('SCC Distance vs Compression Ratio (Hamming)', fontsize=15)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/scc_vs_compression_ratio_hamming_L{L}_T{TRAILS}.pdf', bbox_inches='tight')
print(f"SCC vs Compression Ratio (Hamming) plot saved as '{output_dir}/scc_vs_compression_ratio_hamming_L{L}_T{TRAILS}.pdf'")
# plt.show()

# SCC vs Compression Ratio - Value
fig, ax = plt.subplots(1, 1, figsize=(3.3115, 2))
ax.scatter(cr_value['compression_ratio'], cr_value['avg_scc_distance'], s=100, alpha=0.7, color='orange', label='Value')
for idx, row in cr_value.iterrows():
    ax.annotate(row['codec_name'].replace('Dict_', '').replace('_Value', ''), 
                 (row['compression_ratio'], row['avg_scc_distance']), 
                 fontsize=8, alpha=0.7)
ax.set_xlabel('Compression Ratio', fontsize=15)
ax.set_ylabel('Average SCC Distance', fontsize=15)
# ax.set_title('SCC Distance vs Compression Ratio (Value)', fontsize=15)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/scc_vs_compression_ratio_value_L{L}_T{TRAILS}.pdf', bbox_inches='tight')
print(f"SCC vs Compression Ratio (Value) plot saved as '{output_dir}/scc_vs_compression_ratio_value_L{L}_T{TRAILS}.pdf'")
# plt.show()

# ZCE vs Compression Ratio - Hamming
fig, ax = plt.subplots(1, 1, figsize=(3.3115, 2))
ax.scatter(cr_hamming['compression_ratio'], cr_hamming['avg_zce_distance'], s=100, alpha=0.7, label='Hamming')
for idx, row in cr_hamming.iterrows():
    ax.annotate(row['codec_name'].replace('Dict_', '').replace('_Hamming', ''), 
                 (row['compression_ratio'], row['avg_zce_distance']), 
                 fontsize=8, alpha=0.7)
ax.set_xlabel('Compression Ratio', fontsize=15)
ax.set_ylabel('Indep. Loss', fontsize=15)
# ax.set_title('ZCE Distance vs Compression Ratio (Hamming)', fontsize=15)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/zce_vs_compression_ratio_hamming_L{L}_T{TRAILS}.pdf', bbox_inches='tight')
print(f"ZCE vs Compression Ratio (Hamming) plot saved as '{output_dir}/zce_vs_compression_ratio_hamming_L{L}_T{TRAILS}.pdf'")
# plt.show()

# ZCE vs Compression Ratio - Value
fig, ax = plt.subplots(1, 1, figsize=(3.3115, 2))
ax.scatter(cr_value['compression_ratio'], cr_value['avg_zce_distance'], s=100, alpha=0.7, color='orange', label='Value')
for idx, row in cr_value.iterrows():
    ax.annotate(row['codec_name'].replace('Dict_', '').replace('_Value', ''), 
                 (row['compression_ratio'], row['avg_zce_distance']), 
                 fontsize=8, alpha=0.7)
ax.set_xlabel('Compression Ratio', fontsize=15)
ax.set_ylabel('Indep. Loss', fontsize=15)
# ax.set_title('ZCE Distance vs Compression Ratio (Value)', fontsize=15)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/zce_vs_compression_ratio_value_L{L}_T{TRAILS}.pdf', bbox_inches='tight')
print(f"ZCE vs Compression Ratio (Value) plot saved as '{output_dir}/zce_vs_compression_ratio_value_L{L}_T{TRAILS}.pdf'")
# plt.show()

# AND vs Compression Ratio - Hamming
fig, ax = plt.subplots(1, 1, figsize=(3.3115, 2))
ax.scatter(cr_hamming['compression_ratio'], cr_hamming['avg_and_distance'], s=100, alpha=0.7, label='Hamming')
for idx, row in cr_hamming.iterrows():
    ax.annotate(row['codec_name'].replace('Dict_', '').replace('_Hamming', ''), 
                 (row['compression_ratio'], row['avg_and_distance']), 
                 fontsize=8, alpha=0.7)
ax.set_xlabel('Compression Ratio', fontsize=15)
ax.set_ylabel('Average AND Distance', fontsize=15)
# ax.set_title('AND Distance vs Compression Ratio (Hamming)', fontsize=15)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/and_vs_compression_ratio_hamming_L{L}_T{TRAILS}.pdf', bbox_inches='tight')
print(f"AND vs Compression Ratio (Hamming) plot saved as '{output_dir}/and_vs_compression_ratio_hamming_L{L}_T{TRAILS}.pdf'")
# plt.show()

# AND vs Compression Ratio - Value
fig, ax = plt.subplots(1, 1, figsize=(3.3115, 2))
ax.scatter(cr_value['compression_ratio'], cr_value['avg_and_distance'], s=100, alpha=0.7, color='orange', label='Value')
for idx, row in cr_value.iterrows():
    ax.annotate(row['codec_name'].replace('Dict_', '').replace('_Value', ''), 
                 (row['compression_ratio'], row['avg_and_distance']), 
                 fontsize=8, alpha=0.7)
ax.set_xlabel('Compression Ratio', fontsize=15)
ax.set_ylabel('Average AND Distance', fontsize=15)
# ax.set_title('AND Distance vs Compression Ratio (Value)', fontsize=15)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/and_vs_compression_ratio_value_L{L}_T{TRAILS}.pdf', bbox_inches='tight')
print(f"AND vs Compression Ratio (Value) plot saved as '{output_dir}/and_vs_compression_ratio_value_L{L}_T{TRAILS}.pdf'")
# plt.show()

# Print summary statistics
print(f"\nSummary Statistics:")
print("="*70)
print(f"SCC Range: [{scc_heatmap_global_min:.6f}, {scc_heatmap_global_max:.6f}]")
print(f"ZCE Range: [{zce_heatmap_global_min:.6f}, {zce_heatmap_global_max:.6f}]")
print(f"AND Range: [{and_heatmap_global_min:.6f}, {and_heatmap_global_max:.6f}]")
print("\nDetailed results:")
for data in compression_ratio_data:
    print(f"{data['codec_name']:20s}: CR={data['compression_ratio']:.3f}, SCC Avg Dist={data['avg_scc_distance']:.6f}, ZCE Avg Dist={data['avg_zce_distance']:.6f}, AND Avg Dist={data['avg_and_distance']:.6f}")


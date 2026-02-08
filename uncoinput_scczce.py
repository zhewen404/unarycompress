from compdecomp import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

L = 256
PRECISION = 32
TRAILS = 50
GRID_SIZE = 50  # Number of bins for density grid
CHUNK_SIZE = 500  # Process in chunks to avoid memory issues
MAX_WORKERS = min(22, mp.cpu_count())  # Limit workers to avoid memory overload

def print_dict_results(results):
    for key, value in results.items():
        print(f"{key}: {value}")

def process_chunk(chunk_data):
    """Process a chunk of (a,b) pairs and return results for all codecs"""
    codec_binarylfsr = BinaryLFSR()
    codec_binarysa = BinarySA()
    codec_dicthamming_4_8 = DictCodec(8,4,True)
    codec_dictvalue_4_8 = DictCodec(8,4,False)
    codec_samplingodd = SamplingOdd()
    codec_lp = LowerPrecision(PRECISION)
    codec_histogramrandom = HistogramRandom() #2d
    codec_arr = [codec_binarylfsr, codec_binarysa, codec_dicthamming_4_8, codec_dictvalue_4_8, codec_samplingodd, codec_lp, codec_histogramrandom]
    
    chunk_results = {}
    for codec in codec_arr:
        codec_name = codec.get_name()
        chunk_results[codec_name] = {
            'scc_input': [], 'scc_output': [],
            'zce_input': [], 'zce_output': []
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
                
                # Store SCC and ZCE data
                input_scc = results["scc"][0]
                output_scc = results["scc"][1]
                input_zce = results["zce"][0]
                output_zce = results["zce"][1]
                
                chunk_results[codec_name]['scc_input'].append(input_scc)
                chunk_results[codec_name]['scc_output'].append(output_scc)
                chunk_results[codec_name]['zce_input'].append(input_zce)
                chunk_results[codec_name]['zce_output'].append(output_zce)
    
    return chunk_results

# Create chunks of (a,b) pairs to process
def create_chunks(L, chunk_size):
    """Create chunks of (a,b) pairs"""
    all_pairs = [(a, b) for a in range(0, L+1) for b in range(0, L+1)]
    for i in range(0, len(all_pairs), chunk_size):
        yield all_pairs[i:i + chunk_size]

# Storage for all codec results
codec_binarylfsr = BinaryLFSR()
codec_binarysa = BinarySA()
codec_dicthamming_4_8 = DictCodec(8,4,True)
codec_dictvalue_4_8 = DictCodec(8,4,False)
codec_samplingodd = SamplingOdd()
codec_lp = LowerPrecision(PRECISION)
codec_histogramrandom = HistogramRandom() #2d
codec_arr = [codec_binarylfsr, codec_binarysa, codec_dicthamming_4_8, codec_dictvalue_4_8, codec_samplingodd, codec_lp, codec_histogramrandom]

codec_data = {}
for codec in codec_arr:
    codec_name = codec.get_name()
    codec_data[codec_name] = {
        'scc_input': [], 'scc_output': [],
        'zce_input': [], 'zce_output': []
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

for codec in codec_arr:
    codec_name = codec.get_name()
    scc_inputs = codec_data[codec_name]['scc_input']
    scc_outputs = codec_data[codec_name]['scc_output']
    zce_inputs = codec_data[codec_name]['zce_input']
    zce_outputs = codec_data[codec_name]['zce_output']
    
    if scc_inputs and scc_outputs:
        global_scc_min = min(global_scc_min, min(scc_inputs), min(scc_outputs))
        global_scc_max = max(global_scc_max, max(scc_inputs), max(scc_outputs))
    
    if zce_inputs and zce_outputs:
        global_zce_min = min(global_zce_min, min(zce_inputs), min(zce_outputs))
        global_zce_max = max(global_zce_max, max(zce_inputs), max(zce_outputs))

print(f"Global SCC range: [{global_scc_min:.3f}, {global_scc_max:.3f}]")
print(f"Global ZCE range: [{global_zce_min:.3f}, {global_zce_max:.3f}]")

# Create SCC plots - one subplot per codec
n_codecs = len(codec_arr)
cols = 3  # 3 columns
rows = (n_codecs + cols - 1) // cols  # Calculate rows needed

# SCC Plot
fig_scc, axes_scc = plt.subplots(rows, cols, figsize=(18, 6*rows))
if rows == 1:
    axes_scc = [axes_scc]  # Make it iterable for single row
axes_scc = axes_scc.flatten() if rows > 1 else axes_scc

for i, codec in enumerate(codec_arr):
    codec_name = codec.get_name()
    ax = axes_scc[i]
    
    scc_inputs = codec_data[codec_name]['scc_input']
    scc_outputs = codec_data[codec_name]['scc_output']
    
    # Calculate average vertical distance
    vertical_distances = [abs(scc_outputs[j] - scc_inputs[j]) for j in range(len(scc_inputs))]
    avg_distance = np.mean(vertical_distances)
    
    if scc_inputs and scc_outputs:
        # Create density grid using global range
        bins_x = np.linspace(global_scc_min, global_scc_max, GRID_SIZE)
        bins_y = np.linspace(global_scc_min, global_scc_max, GRID_SIZE)
        
        density, x_edges, y_edges = np.histogram2d(scc_inputs, scc_outputs, bins=[bins_x, bins_y])
        
        # Set zeros to NaN for transparent/white color instead of darkest
        density_processed = np.where(density == 0, np.nan, density)
        
        # Create heatmap
        im = ax.imshow(density_processed.T, origin='lower', 
                      extent=[global_scc_min, global_scc_max, global_scc_min, global_scc_max],
                      cmap='viridis', norm=LogNorm(vmin=1), aspect='equal')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Point Density (log scale)', fontsize=10)
        
        # Add perfect correlation line (y=x)
        ax.plot([global_scc_min, global_scc_max], [global_scc_min, global_scc_max], 'r--', alpha=0.7, linewidth=2, label='y=x')
    
    ax.set_title(f'{codec_name}\nAvg Distance: {avg_distance:.6f}')
    ax.set_xlabel('Input SCC')
    ax.set_ylabel('Output SCC')
    ax.grid(True, alpha=0.3)

# Hide any unused subplots
for i in range(len(codec_arr), len(axes_scc)):
    axes_scc[i].set_visible(False)

plt.tight_layout()
plt.savefig(f'uncoinput_scc_density_L{L}_T{TRAILS}_parallel.png', dpi=300, bbox_inches='tight')
plt.savefig(f'uncoinput_scc_density_L{L}_T{TRAILS}_parallel.pdf', bbox_inches='tight')
print(f"SCC density plot saved as 'uncoinput_scc_density_L{L}_T{TRAILS}_parallel.png' and 'uncoinput_scc_density_L{L}_T{TRAILS}_parallel.pdf'")
plt.show()

# ZCE Plot  
fig_zce, axes_zce = plt.subplots(rows, cols, figsize=(18, 6*rows))
if rows == 1:
    axes_zce = [axes_zce]  # Make it iterable for single row
axes_zce = axes_zce.flatten() if rows > 1 else axes_zce

for i, codec in enumerate(codec_arr):
    codec_name = codec.get_name()
    ax = axes_zce[i]
    
    zce_inputs = codec_data[codec_name]['zce_input']
    zce_outputs = codec_data[codec_name]['zce_output']
    
    # Calculate average vertical distance
    vertical_distances = [abs(zce_outputs[j] - zce_inputs[j]) for j in range(len(zce_inputs))]
    avg_distance = np.mean(vertical_distances)
    
    if zce_inputs and zce_outputs:
        # Create density grid using global range
        bins_x = np.linspace(global_zce_min, global_zce_max, GRID_SIZE)
        bins_y = np.linspace(global_zce_min, global_zce_max, GRID_SIZE)
        
        density, x_edges, y_edges = np.histogram2d(zce_inputs, zce_outputs, bins=[bins_x, bins_y])
        
        # Set zeros to NaN for transparent/white color instead of darkest
        density_processed = np.where(density == 0, np.nan, density)
        
        # Create heatmap with plasma colormap
        im = ax.imshow(density_processed.T, origin='lower', 
                      extent=[global_zce_min, global_zce_max, global_zce_min, global_zce_max],
                      cmap='plasma', norm=LogNorm(vmin=1), aspect='equal')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Point Density (log scale)', fontsize=10)
        
        # Add perfect correlation line (y=x)
        ax.plot([global_zce_min, global_zce_max], [global_zce_min, global_zce_max], 'r--', alpha=0.7, linewidth=2, label='y=x')
    
    ax.set_title(f'{codec_name}\nAvg Distance: {avg_distance:.6f}')
    ax.set_xlabel('Input ZCE')
    ax.set_ylabel('Output ZCE')
    ax.grid(True, alpha=0.3)

# Hide any unused subplots
for i in range(len(codec_arr), len(axes_zce)):
    axes_zce[i].set_visible(False)

plt.tight_layout()
plt.savefig(f'uncoinput_zce_density_L{L}_T{TRAILS}_parallel.png', dpi=300, bbox_inches='tight')
plt.savefig(f'uncoinput_zce_density_L{L}_T{TRAILS}_parallel.pdf', bbox_inches='tight')
print(f"ZCE density plot saved as 'uncoinput_zce_density_L{L}_T{TRAILS}_parallel.png' and 'uncoinput_zce_density_L{L}_T{TRAILS}_parallel.pdf'")
plt.show()

# Print summary statistics
print(f"\nSummary Statistics:")
print("="*70)
for codec in codec_arr:
    codec_name = codec.get_name()
    scc_inputs = codec_data[codec_name]['scc_input']
    scc_outputs = codec_data[codec_name]['scc_output']
    zce_inputs = codec_data[codec_name]['zce_input']
    zce_outputs = codec_data[codec_name]['zce_output']
    
    scc_distances = [abs(scc_outputs[j] - scc_inputs[j]) for j in range(len(scc_inputs))]
    zce_distances = [abs(zce_outputs[j] - zce_inputs[j]) for j in range(len(zce_inputs))]
    
    avg_scc_distance = np.mean(scc_distances)
    avg_zce_distance = np.mean(zce_distances)
    
    print(f"{codec_name:20s}: SCC Avg Dist={avg_scc_distance:.6f}, ZCE Avg Dist={avg_zce_distance:.6f}")


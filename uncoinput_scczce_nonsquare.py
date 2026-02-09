from compdecomp import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os

# Set matplotlib font sizes
plt.rcParams.update({'font.size': 15})

L = 256
PRECISION = 32
TRAILS = 10
GRID_SIZE = 50  # Number of bins for density grid
CHUNK_SIZE = 500  # Process in chunks to avoid memory issues
MAX_WORKERS = min(23, mp.cpu_count())  # Limit workers to avoid memory overload

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
    # codec_histogramrandom = HistogramRandom() #2d
    codec_arr = [codec_binarylfsr, codec_binarysa, codec_dicthamming_4_8, codec_dictvalue_4_8, 
                 codec_samplingodd, codec_lp, 
                #  codec_histogramrandom
                 ]
    
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
                
                # Store SCC, ZCE and AND data
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
codec_binarylfsr = BinaryLFSR()
codec_binarysa = BinarySA()
codec_dicthamming_4_8 = DictCodec(8,4,True)
codec_dictvalue_4_8 = DictCodec(8,4,False)
codec_samplingodd = SamplingOdd()
codec_lp = LowerPrecision(PRECISION)
# codec_histogramrandom = HistogramRandom() #2d
codec_arr = [codec_binarylfsr, codec_binarysa, codec_dicthamming_4_8, codec_dictvalue_4_8, 
             codec_samplingodd, codec_lp, 
            #  codec_histogramrandom
             ]

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

# Calculate global min/max for consistent axes - separate input/output ranges
global_scc_input_min = float('inf')
global_scc_input_max = float('-inf')
global_scc_output_min = float('inf')
global_scc_output_max = float('-inf')
global_zce_input_min = float('inf')
global_zce_input_max = float('-inf')
global_zce_output_min = float('inf')
global_zce_output_max = float('-inf')
global_and_input_min = float('inf')
global_and_input_max = float('-inf')
global_and_output_min = float('inf')
global_and_output_max = float('-inf')

for codec in codec_arr:
    codec_name = codec.get_name()
    scc_inputs = codec_data[codec_name]['scc_input']
    scc_outputs = codec_data[codec_name]['scc_output']
    zce_inputs = codec_data[codec_name]['zce_input']
    zce_outputs = codec_data[codec_name]['zce_output']
    and_inputs = codec_data[codec_name]['and_input']
    and_outputs = codec_data[codec_name]['and_output']
    
    if scc_inputs and scc_outputs:
        global_scc_input_min = min(global_scc_input_min, min(scc_inputs))
        global_scc_input_max = max(global_scc_input_max, max(scc_inputs))
        global_scc_output_min = min(global_scc_output_min, min(scc_outputs))
        global_scc_output_max = max(global_scc_output_max, max(scc_outputs))
    
    if zce_inputs and zce_outputs:
        global_zce_input_min = min(global_zce_input_min, min(zce_inputs))
        global_zce_input_max = max(global_zce_input_max, max(zce_inputs))
        global_zce_output_min = min(global_zce_output_min, min(zce_outputs))
        global_zce_output_max = max(global_zce_output_max, max(zce_outputs))
        
    if and_inputs and and_outputs:
        global_and_input_min = min(global_and_input_min, min(and_inputs))
        global_and_input_max = max(global_and_input_max, max(and_inputs))
        global_and_output_min = min(global_and_output_min, min(and_outputs))
        global_and_output_max = max(global_and_output_max, max(and_outputs))

print(f"Global SCC input range: [{global_scc_input_min:.3f}, {global_scc_input_max:.3f}]")
print(f"Global SCC output range: [{global_scc_output_min:.3f}, {global_scc_output_max:.3f}]")
print(f"Global ZCE input range: [{global_zce_input_min:.3f}, {global_zce_input_max:.3f}]")
print(f"Global ZCE output range: [{global_zce_output_min:.3f}, {global_zce_output_max:.3f}]")
print(f"Global AND input range: [{global_and_input_min:.3f}, {global_and_input_max:.3f}]")
print(f"Global AND output range: [{global_and_output_min:.3f}, {global_and_output_max:.3f}]")

# Calculate global density maximum for consistent colorbar scaling
max_density_scc = 0
max_density_zce = 0
max_density_and = 0

for codec in codec_arr:
    codec_name = codec.get_name()
    scc_inputs = codec_data[codec_name]['scc_input']
    scc_outputs = codec_data[codec_name]['scc_output']
    zce_inputs = codec_data[codec_name]['zce_input']
    zce_outputs = codec_data[codec_name]['zce_output']
    and_inputs = codec_data[codec_name]['and_input']
    and_outputs = codec_data[codec_name]['and_output']
    
    if scc_inputs and scc_outputs:
        bins_x = np.linspace(global_scc_input_min, global_scc_input_max, GRID_SIZE)
        bins_y = np.linspace(global_scc_output_min, global_scc_output_max, GRID_SIZE)
        density, _, _ = np.histogram2d(scc_inputs, scc_outputs, bins=[bins_x, bins_y])
        max_density_scc = max(max_density_scc, density.max())
    
    if zce_inputs and zce_outputs:
        bins_x = np.linspace(global_zce_input_min, global_zce_input_max, GRID_SIZE)
        bins_y = np.linspace(global_zce_output_min, global_zce_output_max, GRID_SIZE)
        density, _, _ = np.histogram2d(zce_inputs, zce_outputs, bins=[bins_x, bins_y])
        max_density_zce = max(max_density_zce, density.max())
        
    if and_inputs and and_outputs:
        bins_x = np.linspace(global_and_input_min, global_and_input_max, GRID_SIZE)
        bins_y = np.linspace(global_and_output_min, global_and_output_max, GRID_SIZE)
        density, _, _ = np.histogram2d(and_inputs, and_outputs, bins=[bins_x, bins_y])
        max_density_and = max(max_density_and, density.max())

print(f"Global SCC density max: {max_density_scc}")
print(f"Global ZCE density max: {max_density_zce}")
print(f"Global AND density max: {max_density_and}")

# Create output directory if it doesn't exist
output_dir = 'correlation_density_nonsquare'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Create individual SCC plots for each codec
for i, codec in enumerate(codec_arr):
    codec_name = codec.get_name()
    
    scc_inputs = codec_data[codec_name]['scc_input']
    scc_outputs = codec_data[codec_name]['scc_output']
    
    # Calculate individual data ranges for cropping
    if scc_inputs and scc_outputs:
        input_min = min(scc_inputs)
        input_max = max(scc_inputs)
        
        # Use individual input range but global output range
        input_range = input_max - input_min
        output_range = global_scc_output_max - global_scc_output_min
        input_padding = input_range * 0.05
        output_padding = output_range * 0.05
        
        plot_input_min = input_min - input_padding
        plot_input_max = input_max + input_padding
        plot_output_min = global_scc_output_min - output_padding
        plot_output_max = global_scc_output_max + output_padding
        
        # Calculate aspect ratio and figure dimensions
        data_aspect = (plot_output_max - plot_output_min) / (plot_input_max - plot_input_min)
        base_width = 4.5
        fig_height = base_width * data_aspect
        # Limit height to reasonable range
        fig_height = max(2.0, min(fig_height, 6.0))
        
        # Create individual figure for each codec with dynamic sizing
        fig, ax = plt.subplots(1, 1, figsize=(base_width, fig_height))
        
        # Calculate average vertical distance
        vertical_distances = [abs(scc_outputs[j] - scc_inputs[j]) for j in range(len(scc_inputs))]
        avg_distance = np.mean(vertical_distances)
        
        # Create density grid using cropped ranges
        bins_x = np.linspace(plot_input_min, plot_input_max, GRID_SIZE)
        bins_y = np.linspace(plot_output_min, plot_output_max, GRID_SIZE)
        
        density, x_edges, y_edges = np.histogram2d(scc_inputs, scc_outputs, bins=[bins_x, bins_y])
        
        # Set zeros to NaN for transparent/white color instead of darkest
        density_processed = np.where(density == 0, np.nan, density)
        
        # Create heatmap with cropped extent
        im = ax.imshow(density_processed.T, origin='lower', 
                      extent=[plot_input_min, plot_input_max, plot_output_min, plot_output_max],
                      cmap='viridis', norm=LogNorm(vmin=1, vmax=max_density_scc), aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Add perfect correlation line (y=x) within visible area
        line_min = max(plot_input_min, plot_output_min)
        line_max = min(plot_input_max, plot_output_max)
        if line_min < line_max:
            ax.plot([line_min, line_max], [line_min, line_max], 'r--', alpha=0.7, linewidth=2, label='y=x')
        
        ax.set_title(f'Avg Distance: {avg_distance:.4f}', fontsize=15)
        ax.set_xlabel('Input SCC', fontsize=15)
        ax.set_ylabel('Output SCC', fontsize=15)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save individual plot
        filename_base = f'uncoinput_scc_density_{codec_name}_L{L}_T{TRAILS}'
        plt.savefig(f'{output_dir}/{filename_base}.pdf', bbox_inches='tight')
        print(f"SCC density plot for {codec_name} saved as '{output_dir}/{filename_base}.pdf'")
        plt.show()
        plt.close()  # Close figure to free memory

# Create individual ZCE plots for each codec
for i, codec in enumerate(codec_arr):
    codec_name = codec.get_name()
    
    zce_inputs = codec_data[codec_name]['zce_input']
    zce_outputs = codec_data[codec_name]['zce_output']
    
    # Calculate individual data ranges for cropping
    if zce_inputs and zce_outputs:
        input_min = min(zce_inputs)
        input_max = max(zce_inputs)
        
        # Use individual input range but global output range
        input_range = input_max - input_min
        output_range = global_zce_output_max - global_zce_output_min
        input_padding = input_range * 0.05 if input_range > 0 else 0.01
        output_padding = output_range * 0.05 if output_range > 0 else 0.01
        
        plot_input_min = input_min - input_padding
        plot_input_max = input_max + input_padding
        plot_output_min = global_zce_output_min - output_padding
        plot_output_max = global_zce_output_max + output_padding
        
        # Calculate aspect ratio and figure dimensions
        data_aspect = (plot_output_max - plot_output_min) / (plot_input_max - plot_input_min)
        base_width = 4.5
        fig_height = base_width * data_aspect
        # Limit height to reasonable range
        fig_height = max(2.0, min(fig_height, 6.0))
        
        # Create individual figure for each codec with dynamic sizing
        fig, ax = plt.subplots(1, 1, figsize=(base_width, fig_height))
        
        # Calculate average vertical distance
        vertical_distances = [abs(zce_outputs[j] - zce_inputs[j]) for j in range(len(zce_inputs))]
        avg_distance = np.mean(vertical_distances)
        
        # Create density grid using cropped ranges
        bins_x = np.linspace(plot_input_min, plot_input_max, GRID_SIZE)
        bins_y = np.linspace(plot_output_min, plot_output_max, GRID_SIZE)
        
        density, x_edges, y_edges = np.histogram2d(zce_inputs, zce_outputs, bins=[bins_x, bins_y])
        
        # Set zeros to NaN for transparent/white color instead of darkest
        density_processed = np.where(density == 0, np.nan, density)
        
        # Create heatmap with plasma colormap and cropped extent
        im = ax.imshow(density_processed.T, origin='lower', 
                      extent=[plot_input_min, plot_input_max, plot_output_min, plot_output_max],
                      cmap='plasma', norm=LogNorm(vmin=1, vmax=max_density_zce), aspect='auto')
        
        # Add colorbar for all ZCE plots
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Add perfect correlation line (y=x) within visible area
        line_min = max(plot_input_min, plot_output_min)
        line_max = min(plot_input_max, plot_output_max)
        if line_min < line_max:
            ax.plot([line_min, line_max], [line_min, line_max], 'r--', alpha=0.7, linewidth=2, label='y=x')
        
        ax.set_title(f'Avg Distance: {avg_distance:.4f}', fontsize=15)
        ax.set_xlabel('Input ZCE', fontsize=15)
        ax.set_ylabel('Output ZCE', fontsize=15)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save individual plot
        filename_base = f'uncoinput_zce_density_{codec_name}_L{L}_T{TRAILS}'
        plt.savefig(f'{output_dir}/{filename_base}.pdf', bbox_inches='tight')
        print(f"ZCE density plot for {codec_name} saved as '{output_dir}/{filename_base}.pdf'")
        plt.show()
        plt.close()  # Close figure to free memory

# Create individual AND plots for each codec
for i, codec in enumerate(codec_arr):
    codec_name = codec.get_name()
    
    and_inputs = codec_data[codec_name]['and_input']
    and_outputs = codec_data[codec_name]['and_output']
    
    # Calculate individual data ranges for cropping
    if and_inputs and and_outputs:
        input_min = min(and_inputs)
        input_max = max(and_inputs)
        
        # Use individual input range but global output range
        input_range = input_max - input_min
        output_range = global_and_output_max - global_and_output_min
        input_padding = input_range * 0.05 if input_range > 0 else 0.001
        output_padding = output_range * 0.05 if output_range > 0 else 0.001
        
        plot_input_min = input_min - input_padding
        plot_input_max = input_max + input_padding
        plot_output_min = global_and_output_min - output_padding
        plot_output_max = global_and_output_max + output_padding
        
        # Calculate aspect ratio and figure dimensions
        data_aspect = (plot_output_max - plot_output_min) / (plot_input_max - plot_input_min)
        base_width = 4.5
        fig_height = base_width * data_aspect
        # Limit height to reasonable range
        fig_height = max(2.0, min(fig_height, 6.0))
        
        # Create individual figure for each codec with dynamic sizing
        fig, ax = plt.subplots(1, 1, figsize=(base_width, fig_height))
        
        # Calculate average vertical distance
        vertical_distances = [abs(and_outputs[j] - and_inputs[j]) for j in range(len(and_inputs))]
        avg_distance = np.mean(vertical_distances)
        
        # Create density grid using cropped ranges
        bins_x = np.linspace(plot_input_min, plot_input_max, GRID_SIZE)
        bins_y = np.linspace(plot_output_min, plot_output_max, GRID_SIZE)
        
        density, x_edges, y_edges = np.histogram2d(and_inputs, and_outputs, bins=[bins_x, bins_y])
        
        # Set zeros to NaN for transparent/white color instead of darkest
        density_processed = np.where(density == 0, np.nan, density)
        
        # Create heatmap with Greys_r colormap and cropped extent
        im = ax.imshow(density_processed.T, origin='lower', 
                      extent=[plot_input_min, plot_input_max, plot_output_min, plot_output_max],
                      cmap='Greys_r', norm=LogNorm(vmin=1, vmax=max_density_and), aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Add perfect correlation line (y=x) within visible area
        line_min = max(plot_input_min, plot_output_min)
        line_max = min(plot_input_max, plot_output_max)
        if line_min < line_max:
            ax.plot([line_min, line_max], [line_min, line_max], 'r--', alpha=0.7, linewidth=2, label='y=x')
        
        ax.set_title(f'Avg Distance: {avg_distance:.6f}', fontsize=15)
        ax.set_xlabel('Input AND', fontsize=15)
        ax.set_ylabel('Output AND', fontsize=15)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save individual plot
        filename_base = f'uncoinput_and_density_{codec_name}_L{L}_T{TRAILS}'
        plt.savefig(f'{output_dir}/{filename_base}.pdf', bbox_inches='tight')
        print(f"AND density plot for {codec_name} saved as '{output_dir}/{filename_base}.pdf'")
        plt.show()
        plt.close()  # Close figure to free memory

# Print summary statistics
print(f"\nSummary Statistics:")
print("="*70)
for codec in codec_arr:
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
    
    print(f"{codec_name:20s}: SCC Avg Dist={avg_scc_distance:.6f}, ZCE Avg Dist={avg_zce_distance:.6f}, AND Avg Dist={avg_and_distance:.6f}")


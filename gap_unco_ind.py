# quantify the SCC and ZCE gap between uncorrelated and independently generated bitstreams
from compdecomp import *
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os

L = 256
TRAILS = 10
CHUNK_SIZE = 1000  # Process in chunks to avoid memory issues
MAX_WORKERS = min(20, mp.cpu_count())  # Limit workers to avoid memory overload

def print_dict_results(results):
    for key, value in results.items():
        print(f"{key}: {value}")

def process_chunk(chunk_data):
    """Process a chunk of (a,b) pairs and return results"""
    chunk_results = {
        'scc_input': [], 'scc_output': [], 
        'zce_input': [], 'zce_output': []
    }
    
    gen_ind = BinaryLFSR() 
    gen_uncorr = BinaryUncorrelated()
    
    for a, b in chunk_data:
        for trial in range(TRAILS):
            a_in = gen_ind.gen(a, L)
            b_in = gen_ind.gen(b, L)
            results = gen_uncorr.evaluate_all(a_in, b_in, ["scc", "zce"])
            
            chunk_results['scc_input'].append(results["scc"][0])
            chunk_results['scc_output'].append(results["scc"][1])
            chunk_results['zce_input'].append(results["zce"][0])
            chunk_results['zce_output'].append(results["zce"][1])
    
    return chunk_results

# Create chunks of (a,b) pairs to process
def create_chunks(L, chunk_size):
    """Create chunks of (a,b) pairs"""
    all_pairs = [(a, b) for a in range(0, L+1) for b in range(0, L+1)]
    for i in range(0, len(all_pairs), chunk_size):
        yield all_pairs[i:i + chunk_size]

# Storage for results 
scc_input_data = []
scc_output_data = []
zce_input_data = []
zce_output_data = []

total_pairs = (L+1)**2
print(f"Running {TRAILS} trials for each (a,b) pair...")
print(f"Total pairs: {total_pairs:,}, Processing in chunks of {CHUNK_SIZE}")
print(f"Using {MAX_WORKERS} parallel workers...")

# Process chunks in parallel
with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    chunk_generator = create_chunks(L, CHUNK_SIZE)
    
    for i, chunk_results in enumerate(executor.map(process_chunk, chunk_generator)):
        # Accumulate results from this chunk
        scc_input_data.extend(chunk_results['scc_input'])
        scc_output_data.extend(chunk_results['scc_output'])
        zce_input_data.extend(chunk_results['zce_input'])
        zce_output_data.extend(chunk_results['zce_output'])
        
        # Progress update
        processed_pairs = len(scc_input_data)
        if (i + 1) % 10 == 0:
            print(f"Processed {processed_pairs:,}/{total_pairs:,} pairs ({100*processed_pairs/total_pairs:.1f}%)")

print(f"Completed processing {len(scc_input_data):,} data points")

# Create output directory if it doesn't exist
output_dir = 'gap_unco_ind'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Calculate aggregate metrics: average vertical distance to y=x line
scc_vertical_distances = [abs(scc_output_data[i] - scc_input_data[i]) for i in range(len(scc_input_data))]
zce_vertical_distances = [abs(zce_output_data[i] - zce_input_data[i]) for i in range(len(zce_input_data))]

avg_scc_vertical_distance = np.mean(scc_vertical_distances)
avg_zce_vertical_distance = np.mean(zce_vertical_distances)

print(f"\nAggregate Metrics (Average Vertical Distance to y=x):")
print("="*60)
print(f"SCC Average Vertical Distance: {avg_scc_vertical_distance:.6f}")
print(f"ZCE Average Vertical Distance: {avg_zce_vertical_distance:.6f}")

# Create scatter plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# SCC Scatter Plot
ax1.scatter(scc_input_data, scc_output_data, alpha=0.6, s=20, label='SCC Data Points')
ax1.set_title(f'SCC: Input vs Output\nAvg Distance to y=x: {avg_scc_vertical_distance:.6f}')
ax1.set_xlabel('Input SCC')
ax1.set_ylabel('Output SCC')
ax1.grid(True, alpha=0.3)

# Add perfect correlation line (y=x) for SCC
min_scc = min(min(scc_input_data), min(scc_output_data))
max_scc = max(max(scc_input_data), max(scc_output_data))
ax1.plot([min_scc, max_scc], [min_scc, max_scc], 'r--', alpha=0.7, linewidth=2, label='Perfect Correlation (y=x)')
ax1.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center')

# ZCE Scatter Plot  
ax2.scatter(zce_input_data, zce_output_data, alpha=0.6, s=20, label='ZCE Data Points', color='orange')
ax2.set_title(f'ZCE: Input vs Output\nAvg Distance to y=x: {avg_zce_vertical_distance:.6f}')
ax2.set_xlabel('Input ZCE')
ax2.set_ylabel('Output ZCE')
ax2.grid(True, alpha=0.3)

# Add perfect correlation line (y=x) for ZCE
min_zce = min(min(zce_input_data), min(zce_output_data))
max_zce = max(max(zce_input_data), max(zce_output_data))
ax2.plot([min_zce, max_zce], [min_zce, max_zce], 'r--', alpha=0.7, linewidth=2, label='Perfect Correlation (y=x)')
ax2.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center')

plt.tight_layout()
plt.savefig(f'{output_dir}/gap_unco_ind_scatter_L{L}_T{TRAILS}_parallel.pdf', bbox_inches='tight')
print(f"Scatter plots saved as '{output_dir}/gap_unco_ind_scatter_L{L}_T{TRAILS}_parallel.pdf'")
plt.show()

# Print summary statistics
print(f"\nSummary Statistics for {TRAILS} trials:")
print("="*50)
print(f"SCC - Input: mean={np.mean(scc_input_data):.4f}, std={np.std(scc_input_data):.4f}")
print(f"SCC - Output: mean={np.mean(scc_output_data):.4f}, std={np.std(scc_output_data):.4f}")
print(f"ZCE - Input: mean={np.mean(zce_input_data):.4f}, std={np.std(zce_input_data):.4f}")
print(f"ZCE - Output: mean={np.mean(zce_output_data):.4f}, std={np.std(zce_output_data):.4f}")

# Calculate correlation coefficients
scc_correlation = np.corrcoef(scc_input_data, scc_output_data)[0,1]
zce_correlation = np.corrcoef(zce_input_data, zce_output_data)[0,1]
print(f"\nCorrelation Coefficients:")
print(f"SCC Input vs Output: {scc_correlation:.4f}")
print(f"ZCE Input vs Output: {zce_correlation:.4f}")


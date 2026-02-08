import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from compdecomp import *

# Set matplotlib style
plt.rcParams.update({
    'font.size': 15,
    'axes.titlesize': 15,
    'axes.labelsize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13
}) 

###### SA ######
L=256
TRAILS=1
sa_gen = BinarySA()

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

def print_dict_results(results):
    for key, value in results.items():
        print(f"{key}: {value}")

sa_in = {}
sa_out = {}
value_in = {}
value_out = {}
for a in range(0, L+1):
    a_in = sa_gen.gen(a, L)
    for codec in codec_arr:
        # print(f"Evaluating {codec.get_name()} for a={a}...")

        # Initialize arrays for this specific codec if not already done
        codec_name = codec.get_name()
        if codec_name not in sa_in:
            sa_in[codec_name] = [0] * (L+1)
            sa_out[codec_name] = [0] * (L+1)
            value_in[codec_name] = [0] * (L+1)
            value_out[codec_name] = [0] * (L+1)

        # run independent trials
        sa_in_trials = []
        sa_out_trials = []
        value_in_trials = []
        value_out_trials = []
        
        for trial in range(TRAILS):
            results = codec.evaluate_all(a_in, a_in, ["value", "sa"])
            sa_in_trials.append(results["sa"][0][0])
            sa_out_trials.append(results["sa"][1][0])
            value_in_trials.append(results["value"][0][0])
            value_out_trials.append(results["value"][1][0])
        
        # Store mean results
        sa_in[codec_name][a] = np.mean(sa_in_trials)
        sa_out[codec_name][a] = np.mean(sa_out_trials)
        value_in[codec_name][a] = np.mean(value_in_trials)
        value_out[codec_name][a] = np.mean(value_out_trials)
print()

# Organize data for heatmaps
def parse_codec_name(codec_name):
    """Parse codec name to extract codebook_size, chunk_size, and method"""
    parts = codec_name.split('_')
    codebook_size = int(parts[1])
    chunk_size = int(parts[2])
    method = parts[3]  # 'Hamming' or 'Value'
    return codebook_size, chunk_size, method

# Calculate mean differences for each codec
results_data = []
for codec_name in sa_in.keys():
    codebook_size, chunk_size, method = parse_codec_name(codec_name)
    
    # Calculate mean SA difference
    sa_diff = [sa_in[codec_name][i] - sa_out[codec_name][i] for i in range(L+1)]
    mean_sa_diff = np.mean(sa_diff)
    
    # Calculate mean value absolute difference
    value_diff = [abs(value_out[codec_name][i] - value_in[codec_name][i]) for i in range(L+1)]
    mean_value_diff = np.mean(value_diff)
    
    results_data.append({
        'Codebook_Size': codebook_size,
        'Chunk_Size': chunk_size,
        'Method': method,
        'Mean_SA_Diff': mean_sa_diff,
        'Mean_Value_Diff': mean_value_diff
    })

# Create DataFrame
df = pd.DataFrame(results_data)

# Calculate global min/max for consistent color scales
value_min = 0
value_max = max(df['Mean_Value_Diff'].max(), abs(df['Mean_Value_Diff'].min()))
sa_min = 0  
sa_max = max(df['Mean_SA_Diff'].max(), abs(df['Mean_SA_Diff'].min()))

# Create heatmaps
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Value plots
# Hamming Value Differences
df_hamming = df[df['Method'] == 'Hamming']
pivot_hamming_value = df_hamming.pivot(index='Chunk_Size', columns='Codebook_Size', values='Mean_Value_Diff')
sns.heatmap(pivot_hamming_value, annot=True, fmt='.4f', cmap='viridis', ax=ax1, 
           cbar_kws={'shrink': 0.6}, annot_kws={'fontsize': 11}, vmin=value_min, vmax=value_max)
ax1.set_title('Hamming: Mean Value Absolute Difference')
ax1.set_xlabel('Codebook Size')
ax1.set_ylabel('Chunk Size')

# Value Value Differences  
df_value = df[df['Method'] == 'Value']
pivot_value_value = df_value.pivot(index='Chunk_Size', columns='Codebook_Size', values='Mean_Value_Diff')
sns.heatmap(pivot_value_value, annot=True, fmt='.4f', cmap='viridis', ax=ax2,
           cbar_kws={'shrink': 0.6}, annot_kws={'fontsize': 11}, vmin=value_min, vmax=value_max)
ax2.set_title('Value: Mean Value Absolute Difference')
ax2.set_xlabel('Codebook Size')
ax2.set_ylabel('Chunk Size')

# SA plots
# Hamming SA Differences
pivot_hamming_sa = df_hamming.pivot(index='Chunk_Size', columns='Codebook_Size', values='Mean_SA_Diff')
sns.heatmap(pivot_hamming_sa, annot=True, fmt='.4f', cmap='plasma', ax=ax3,
           cbar_kws={'shrink': 0.6}, annot_kws={'fontsize': 11}, vmin=sa_min, vmax=sa_max)
ax3.set_title('Hamming: Mean SA Difference')
ax3.set_xlabel('Codebook Size')
ax3.set_ylabel('Chunk Size')

# Value SA Differences
pivot_value_sa = df_value.pivot(index='Chunk_Size', columns='Codebook_Size', values='Mean_SA_Diff')
sns.heatmap(pivot_value_sa, annot=True, fmt='.4f', cmap='plasma', ax=ax4,
           cbar_kws={'shrink': 0.6}, annot_kws={'fontsize': 11}, vmin=sa_min, vmax=sa_max)
ax4.set_title('Value: Mean SA Difference')
ax4.set_xlabel('Codebook Size')
ax4.set_ylabel('Chunk Size')

plt.tight_layout(pad=2.0)
plt.savefig(f'sa_input_sensitivity_heatmaps_L{L}_T{TRAILS}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'sa_input_sensitivity_heatmaps_L{L}_T{TRAILS}.pdf', bbox_inches='tight')
print(f"Heatmaps saved as 'sa_input_sensitivity_heatmaps_L{L}_T{TRAILS}.png' and 'sa_input_sensitivity_heatmaps_L{L}_T{TRAILS}.pdf'")
plt.show()

# Print summary statistics
print("\n" + "="*80)
print("HEATMAP SUMMARY STATISTICS")
print("="*80)

print("\nValue Mean Absolute Differences:")
print("Hamming Method:")
for idx, row in df_hamming.iterrows():
    print(f"  Chunk {int(row['Chunk_Size']):2d}, Codebook {int(row['Codebook_Size']):2d}: {row['Mean_Value_Diff']:.6f}")

print("\nValue Method:")
for idx, row in df_value.iterrows():
    print(f"  Chunk {int(row['Chunk_Size']):2d}, Codebook {int(row['Codebook_Size']):2d}: {row['Mean_Value_Diff']:.6f}")

print("\nSA Mean Differences:")
print("Hamming Method:")
for idx, row in df_hamming.iterrows():
    print(f"  Chunk {int(row['Chunk_Size']):2d}, Codebook {int(row['Codebook_Size']):2d}: {row['Mean_SA_Diff']:.6f}")

print("\nValue Method:")
for idx, row in df_value.iterrows():
    print(f"  Chunk {int(row['Chunk_Size']):2d}, Codebook {int(row['Codebook_Size']):2d}: {row['Mean_SA_Diff']:.6f}")

# Find best performing configurations
print("\n" + "="*80)
print("BEST PERFORMING CONFIGURATIONS")
print("="*80)

best_value_hamming = df_hamming.loc[df_hamming['Mean_Value_Diff'].idxmin()]
best_value_value = df_value.loc[df_value['Mean_Value_Diff'].idxmin()]
best_sa_hamming = df_hamming.loc[df_hamming['Mean_SA_Diff'].idxmin()]
best_sa_value = df_value.loc[df_value['Mean_SA_Diff'].idxmin()]

print(f"Best Value Diff (Hamming): Chunk {int(best_value_hamming['Chunk_Size'])}, Codebook {int(best_value_hamming['Codebook_Size'])}, Diff: {best_value_hamming['Mean_Value_Diff']:.6f}")
print(f"Best Value Diff (Value):   Chunk {int(best_value_value['Chunk_Size'])}, Codebook {int(best_value_value['Codebook_Size'])}, Diff: {best_value_value['Mean_Value_Diff']:.6f}")
print(f"Best SA Diff (Hamming):    Chunk {int(best_sa_hamming['Chunk_Size'])}, Codebook {int(best_sa_hamming['Codebook_Size'])}, Diff: {best_sa_hamming['Mean_SA_Diff']:.6f}")
print(f"Best SA Diff (Value):      Chunk {int(best_sa_value['Chunk_Size'])}, Codebook {int(best_sa_value['Codebook_Size'])}, Diff: {best_sa_value['Mean_SA_Diff']:.6f}")



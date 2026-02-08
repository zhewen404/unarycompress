import numpy as np
import matplotlib.pyplot as plt
from compdecomp import * 

###### SA ######
L=256
TRAILS=10
sa_gen = BinarySA()

codec_binarylfsr = BinaryLFSR()
codec_binarysa = BinarySA()
codec_dicthamming_4_8 = DictCodec(8,4,True)
codec_dictvalue_4_8 = DictCodec(8,4,False)
codec_samplingodd = SamplingOdd()
codec_lp = LowerPrecision(32)
codec_histogramrandom = HistogramRandom() #2d
codec_arr = [codec_binarylfsr, codec_binarysa, codec_dicthamming_4_8, codec_dictvalue_4_8, codec_samplingodd, codec_lp, codec_histogramrandom]

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
        if codec == codec_histogramrandom:
            continue
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
# print_dict_results(sa_in)
# print_dict_results(sa_out)
# print_dict_results(value_in)
# print_dict_results(value_out)

# Calculate absolute differences and create plots
x_values = list(range(0, L+1))

# Create three subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1: SA Absolute Differences
ax1.set_title('Streaming Accuracy Differences')
ax1.set_xlabel('Input Value (a)')
ax1.set_ylabel('SA_in - SA_out')
ax1.grid(True, alpha=0.3)

for codec_name in sa_in.keys():
    sa_diff = [sa_in[codec_name][i] - sa_out[codec_name][i] for i in range(L+1)]
    ax1.plot(x_values, sa_diff, marker='o', label=codec_name, linewidth=2, markersize=4)

ax1.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center')

# Plot 2: Value Absolute Differences  
ax2.set_title('Value Absolute Differences')
ax2.set_xlabel('Input Value (a)')
ax2.set_ylabel('|Value_out - Value_in|')
ax2.grid(True, alpha=0.3)

for codec_name in value_in.keys():
    value_diff = [abs(value_out[codec_name][i] - value_in[codec_name][i]) for i in range(L+1)]
    ax2.plot(x_values, value_diff, marker='o', label=codec_name, linewidth=2, markersize=4)

ax2.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center')

# Plot 3: Input vs Output Value Scatter
ax3.set_title('Input vs Output Value')
ax3.set_xlabel('Input Value')
ax3.set_ylabel('Output Value')
ax3.grid(True, alpha=0.3)

for codec_name in value_in.keys():
    input_vals = [value_in[codec_name][i] for i in range(L+1)]
    output_vals = [value_out[codec_name][i] for i in range(L+1)]
    ax3.scatter(input_vals, output_vals, label=codec_name, alpha=0.7, s=20)

# Add perfect correlation line (y=x)
min_val = min(min(input_vals) for codec_name in value_in.keys() for input_vals in [[value_in[codec_name][i] for i in range(L+1)]])
max_val = max(max(input_vals) for codec_name in value_in.keys() for input_vals in [[value_in[codec_name][i] for i in range(L+1)]])
ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1, label='Perfect (y=x)')

ax3.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center')

plt.tight_layout()
plt.savefig(f'sainput_saval_{TRAILS}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'sainput_sa_value_{TRAILS}.pdf', bbox_inches='tight')
print(f"Plots saved as 'sainput_saval_{TRAILS}.png' and 'sainput_sa_value_{TRAILS}.pdf'")
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print("\nSA Difference (SA_in - SA_out):")
for codec_name in sa_in.keys():
    sa_diff = [sa_in[codec_name][i] - sa_out[codec_name][i] for i in range(L+1)]
    print(f"{codec_name:20s}: Mean={np.mean(sa_diff):.4f}, Max={np.max(sa_diff):.4f}, Sum={np.sum(sa_diff):.4f}")

print("\nValue Absolute Difference (|Value_out - Value_in|):")
for codec_name in value_in.keys():
    value_diff = [abs(value_out[codec_name][i] - value_in[codec_name][i]) for i in range(L+1)]
    print(f"{codec_name:20s}: Mean={np.mean(value_diff):.4f}, Max={np.max(value_diff):.4f}, Sum={np.sum(value_diff):.4f}")


# print input value and output value for odd bits
print("\n" + "="*60)
print("ODD BITS INPUT/OUTPUT")
print("="*60)
codec_name = "SamplingOdd"
print(f"\n{codec_name} - Odd Bits Input/Output:")
for a in range(0, L+1):
    print(f"a={a:3d}: Value_in={value_in[codec_name][a]:.4f}, Value_out={value_out[codec_name][a]:.4f}")
print(f"\nLower Precision (32) - Input/Output:")
codec_name = "LowerPrecision_32"
for a in range(0, L+1):
    print(f"a={a:3d}: Value_in={value_in[codec_name][a]:.4f}, Value_out={value_out[codec_name][a]:.4f}")
        

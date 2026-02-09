from abc import ABC, abstractmethod

import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import torch
from pylfsr import LFSR
import pandas as pd
import math

class AbstractCodec(ABC):

    ############### ZCE helper functions ###############
    def zce(self,stream_a, stream_b):
        a = np.mean(stream_a)
        b = np.mean(stream_b)
        delta_naught = 1/len(stream_a) * ((a*b/(1/len(stream_a))) + 1/2) - (a*b)
        delta = np.mean(np.bitwise_and(stream_a, stream_b)) - (a*b)
        if delta == 0:
            zce = 0
        else:
            zce = delta*(1-abs(delta_naught/delta))
        return zce
    
    ############### SCC helper functions ###############
    def scc(self, stream_a, stream_b): 
        #https://ieeexplore.ieee.org/document/6657023
        unary_mult = np.mean(np.bitwise_and(stream_a, stream_b))
        binary_mult = np.mean(stream_a)* np.mean(stream_b)
        if unary_mult > binary_mult:
            denominator = (min(np.mean(stream_a), np.mean(stream_b)) - binary_mult)
        else:
            denominator = (binary_mult - max(np.mean(stream_a) + np.mean(stream_b) -1, 0))
        if denominator == 0:
            scc = 0
        else:
            scc = (unary_mult - binary_mult) / denominator
        return scc

    ############### SA helper functions ###############
    def calculate_raw_streaming_accuracy(self,bitstream):
        """
        Calculates the raw streaming accuracy (Phi) of a stochastic bitstream.

        Based on Equation (1) from 'Streaming Accuracy: Characterizing Early Termination'.

        Args:
            bitstream (list of int or str): A sequence of 0s and 1s (e.g., [0, 1, 0, 1] or "0101").

        Returns:
            float: The raw streaming accuracy value.
        """
        # Ensure input is a list of integers
        if isinstance(bitstream, str):
            bits = [int(b) for b in bitstream]
        else:
            bits = bitstream

        L = len(bits)
        if L == 0:
            return 0.0

        # P_X: The value represented by the full bitstream
        # "P_X represents the value of the full bitstream"
        total_ones = sum(bits)
        P_X = total_ones / L

        cumulative_error_sum = 0.0
        current_ones_count = 0

        # Iterate through every possible termination point i from 1 to L
        for i in range(1, L + 1):
            # Update partial count (bit at index i-1)
            current_bit = bits[i-1]
            current_ones_count += current_bit

            # P_Xi: Value of the partial bitstream up to bit i
            # "value represented by its initial partial bitstream value up to the i-th bit"
            P_Xi = current_ones_count / i

            # Add the absolute error |P_Xi - P_X| to the sum
            cumulative_error_sum += abs(P_Xi - P_X)

        # Calculate Raw Streaming Accuracy (Phi)
        # Formula: 1 - (Sum of errors / L)
        raw_streaming_accuracy = 1 - (cumulative_error_sum / L)

        return raw_streaming_accuracy
    def get_best_and_worst_bitstreams(self,length, num_ones):
        target_value = num_ones / length

        # --- Construct Best (Greedy) ---
        best_bits = []
        current_ones = 0
        for i in range(1, length + 1):
            # Calculate error if we add 0 vs if we add 1
            # Option 0:
            val_0 = current_ones / i
            err_0 = abs(val_0 - target_value)

            # Option 1:
            val_1 = (current_ones + 1) / i
            err_1 = abs(val_1 - target_value)

            if err_1 < err_0:
                best_bits.append(1)
                current_ones += 1
            elif err_0 < err_1:
                best_bits.append(0)
            else:
                # Tie-breaker (can prefer 1 or 0, usually keeps distribution balanced)
                # Simple heuristic: pick 1 if we are "under" the target
                if (current_ones / i) < target_value:
                        best_bits.append(1)
                        current_ones += 1
                else:
                        best_bits.append(0)

        # --- Construct Worst (Clustered) ---
        if target_value <= 0.5:
            # Worst case: All 1s at the FRONT (starts with max error 1.0 vs small target)
            worst_bits = [1]*num_ones + [0]*(length - num_ones)
        else:
            # Worst case: All 1s at the BACK (starts with max error 0.0 vs large target)
            worst_bits = [0]*(length - num_ones) + [1]*num_ones

        return best_bits, worst_bits
    def get_streaming_accuracy(self, bitstream, length, num_ones):
        # print(f"length: {length}, num_ones: {num_ones}")
        raw = self.calculate_raw_streaming_accuracy(bitstream)
        best_bits, worst_bits = self.get_best_and_worst_bitstreams(length, num_ones)
        best = self.calculate_raw_streaming_accuracy(best_bits)
        worst = self.calculate_raw_streaming_accuracy(worst_bits)
        if best == worst:
            streaming_accuracy = 1.0
        else:
            streaming_accuracy = (raw - worst) / (best - worst)
        return streaming_accuracy

class Codec2D(AbstractCodec):
    """
    Abstract base class for 2D compression/decompression codecs.
    All concrete codec implementations must inherit from this class
    and implement the compress and decompress methods.
    """
    @abstractmethod
    def gen(self, num1s1, num1s2, length):
        pass
    
    @abstractmethod
    def compress(self, dataA, dataB):
        pass
    
    @abstractmethod
    def decompress(self, compressed_a, compressed_b, shape):
        pass

    def transform(self, stream_a, stream_b):
        compressed_a, compressed_b = self.compress(stream_a, stream_b)
        decompressed_a, decompressed_b = self.decompress(compressed_a, compressed_b, len(stream_a))
        return decompressed_a, decompressed_b
    
    def evaluate_all(self, input_stream_a, input_stream_b, flags=['value', 'sa', 'scc', 'zce', 'and', 'or']):
        results = {}
        output_stream_a, output_stream_b = self.transform(input_stream_a, input_stream_b)
        flag_count = len(flags)
        if 'value' in flags:
            input_value_a = np.mean(input_stream_a)
            input_value_b = np.mean(input_stream_b)
            output_value_a = np.mean(output_stream_a)
            output_value_b = np.mean(output_stream_b)
            results['value'] = (input_value_a, input_value_b), (output_value_a, output_value_b)
            flag_count -= 1
        if 'sa' in flags:
            input_sa_a = self.get_streaming_accuracy(input_stream_a, len(input_stream_a), sum(input_stream_a))
            input_sa_b = self.get_streaming_accuracy(input_stream_b, len(input_stream_b), sum(input_stream_b))
            output_sa_a = self.get_streaming_accuracy(output_stream_a, len(output_stream_a), sum(output_stream_a))
            output_sa_b = self.get_streaming_accuracy(output_stream_b, len(output_stream_b), sum(output_stream_b))
            results['sa'] = (input_sa_a, input_sa_b), (output_sa_a, output_sa_b)
            flag_count -= 1
        if 'scc' in flags:
            input_scc = self.scc(input_stream_a, input_stream_b)
            output_scc = self.scc(output_stream_a, output_stream_b)
            results['scc'] = input_scc, output_scc
            flag_count -= 1
        if 'zce' in flags:
            input_zce = self.zce(input_stream_a, input_stream_b)
            output_zce = self.zce(output_stream_a, output_stream_b)
            results['zce'] = input_zce, output_zce
            flag_count -= 1
        if 'and' in flags:
            input_and = np.mean(np.bitwise_and(input_stream_a, input_stream_b))
            output_and = np.mean(np.bitwise_and(output_stream_a, output_stream_b))
            results['and'] = input_and, output_and
            flag_count -= 1
        if 'or' in flags:
            input_or = np.mean(np.bitwise_or(input_stream_a, input_stream_b))
            output_or = np.mean(np.bitwise_or(output_stream_a, output_stream_b))
            results['or'] = input_or, output_or
            flag_count -= 1
        if flag_count > 0:
            print(f"Warning: {flag_count} unrecognized flags were ignored.")
        return results
    
    def evaluate_value(self, input_stream_a, input_stream_b):
        output_stream_a, output_stream_b = self.transform(input_stream_a, input_stream_b)
        input_value_a = np.mean(input_stream_a)
        input_value_b = np.mean(input_stream_b)
        output_value_a = np.mean(output_stream_a)
        output_value_b = np.mean(output_stream_b)
        return (input_value_a, input_value_b), (output_value_a, output_value_b)
    def evaluate_sa(self, input_stream_a, input_stream_b): # Streaming Accuracy
        output_stream_a, output_stream_b = self.transform(input_stream_a, input_stream_b)
        input_sa_a = self.get_streaming_accuracy(input_stream_a, len(input_stream_a), sum(input_stream_a))
        input_sa_b = self.get_streaming_accuracy(input_stream_b, len(input_stream_b), sum(input_stream_b))
        output_sa_a = self.get_streaming_accuracy(output_stream_a, len(output_stream_a), sum(output_stream_a))
        output_sa_b = self.get_streaming_accuracy(output_stream_b, len(output_stream_b), sum(output_stream_b))
        return (input_sa_a, input_sa_b), (output_sa_a, output_sa_b)
    
    def evaluate_scc(self, input_stream_a, input_stream_b):
        output_stream_a, output_stream_b = self.transform(input_stream_a, input_stream_b)
        input_scc = self.scc(input_stream_a, input_stream_b)
        output_scc = self.scc(output_stream_a, output_stream_b)
        return input_scc, output_scc
    def evaluate_zce(self, input_stream_a, input_stream_b):
        output_stream_a, output_stream_b = self.transform(input_stream_a, input_stream_b)
        input_zce = self.zce(input_stream_a, input_stream_b)
        output_zce = self.zce(output_stream_a, output_stream_b)
        return input_zce, output_zce
    
    def evaluate_and_result(self, input_stream_a, input_stream_b):
        output_stream_a, output_stream_b = self.transform(input_stream_a, input_stream_b)
        input_and = np.mean(np.bitwise_and(input_stream_a, input_stream_b))
        output_and = np.mean(np.bitwise_and(output_stream_a, output_stream_b))
        return input_and, output_and
    def evaluate_or_result(self, input_stream_a, input_stream_b):
        output_stream_a, output_stream_b = self.transform(input_stream_a, input_stream_b)
        input_or = np.mean(np.bitwise_or(input_stream_a, input_stream_b))
        output_or = np.mean(np.bitwise_or(output_stream_a, output_stream_b))
        return input_or, output_or
    
class Codec(AbstractCodec):
    """
    Abstract base class for compression/decompression codecs.
    All concrete codec implementations must inherit from this class
    and implement the compress and decompress methods.
    """
    @abstractmethod
    def gen(self, num1s, length):
        pass
    
    @abstractmethod
    def compress(self, data):
        pass
    
    @abstractmethod
    def decompress(self, compressed_data, length):
        pass

    def transform(self, stream):
        compressed = self.compress(stream)
        decompressed = self.decompress(compressed, len(stream))
        return decompressed
    
    def evaluate_all(self, input_stream_a, input_stream_b, flags=['value', 'sa', 'scc', 'zce', 'and', 'or']):
        results = {}
        flag_count = len(flags)
        output_stream_a = self.transform(input_stream_a)
        output_stream_b = self.transform(input_stream_b)
        # print(f"input_stream_a: {input_stream_a}")
        # print(f"input_stream_b: {input_stream_b}")
        # print(f"output_stream_a: {output_stream_a}")
        # print(f"output_stream_b: {output_stream_b}")
        if 'value' in flags:
            input_value_a = np.mean(input_stream_a)
            input_value_b = np.mean(input_stream_b)
            output_value_a = np.mean(output_stream_a)
            output_value_b = np.mean(output_stream_b)
            results['value'] = (input_value_a, input_value_b), (output_value_a, output_value_b)
            flag_count -= 1
        if 'sa' in flags:
            input_sa_a = self.get_streaming_accuracy(input_stream_a, len(input_stream_a), sum(input_stream_a))
            input_sa_b = self.get_streaming_accuracy(input_stream_b, len(input_stream_b), sum(input_stream_b))
            output_sa_a = self.get_streaming_accuracy(output_stream_a, len(output_stream_a), sum(output_stream_a))
            output_sa_b = self.get_streaming_accuracy(output_stream_b, len(output_stream_b), sum(output_stream_b))
            results['sa'] = (input_sa_a, input_sa_b), (output_sa_a, output_sa_b)
            flag_count -= 1
        if 'scc' in flags:
            input_scc = self.scc(input_stream_a, input_stream_b)
            output_scc = self.scc(output_stream_a, output_stream_b)
            results['scc'] = input_scc, output_scc
            flag_count -= 1
        if 'zce' in flags:
            input_zce = self.zce(input_stream_a, input_stream_b)
            output_zce = self.zce(output_stream_a, output_stream_b)
            results['zce'] = input_zce, output_zce
            flag_count -= 1
        if 'and' in flags:
            input_and = np.mean(np.bitwise_and(input_stream_a, input_stream_b))
            output_and = np.mean(np.bitwise_and(output_stream_a, output_stream_b))
            results['and'] = input_and, output_and
            flag_count -= 1
        if 'or' in flags:
            input_or = np.mean(np.bitwise_or(input_stream_a, input_stream_b))
            output_or = np.mean(np.bitwise_or(output_stream_a, output_stream_b))
            results['or'] = input_or, output_or
            flag_count -= 1
        if flag_count > 0:
            print(f"Warning: {flag_count} unrecognized flags were ignored.")
        return results
    
    def evaluate_value(self, input_stream):
        output_stream = self.transform(input_stream)
        input_value = np.mean(input_stream)
        output_value = np.mean(output_stream)
        return input_value, output_value
    def evaluate_sa(self, input_stream): # Streaming Accuracy
        output_stream = self.transform(input_stream)
        input_sa = self.get_streaming_accuracy(input_stream, len(input_stream), sum(input_stream))
        output_sa = self.get_streaming_accuracy(output_stream, len(output_stream), sum(output_stream))
        return input_sa, output_sa
    
    def evaluate_scc(self, input_stream_a, input_stream_b):
        output_stream_a = self.transform(input_stream_a)
        output_stream_b = self.transform(input_stream_b)
        input_scc = self.scc(input_stream_a, input_stream_b)
        output_scc = self.scc(output_stream_a, output_stream_b)
        return input_scc, output_scc
    def evaluate_autoscc(self, input_stream):
        output_stream = self.transform(input_stream)
        autoscc = self.scc(input_stream, output_stream)
        return autoscc
    
    def evaluate_zce(self, input_stream_a, input_stream_b):
        output_stream_a = self.transform(input_stream_a)
        output_stream_b = self.transform(input_stream_b)
        input_zce = self.zce(input_stream_a, input_stream_b)
        output_zce = self.zce(output_stream_a, output_stream_b)
        return input_zce, output_zce
    def evaluate_autozce(self, input_stream):
        output_stream = self.transform(input_stream)
        autozce = self.zce(input_stream, output_stream)
        return autozce
    
    def evaluate_and_result(self, input_stream_a, input_stream_b):
        # prefer perfectly uncorrelated streams
        output_stream_a = self.transform(input_stream_a)
        output_stream_b = self.transform(input_stream_b)
        input_and = np.mean(np.bitwise_and(input_stream_a, input_stream_b))
        output_and = np.mean(np.bitwise_and(output_stream_a, output_stream_b))
        return input_and, output_and
    def evaluate_or_result(self, input_stream_a, input_stream_b):
        # prefer perfectly negatively correlated streams
        output_stream_a = self.transform(input_stream_a)
        output_stream_b = self.transform(input_stream_b)
        input_or = np.mean(np.bitwise_or(input_stream_a, input_stream_b))
        output_or = np.mean(np.bitwise_or(output_stream_a, output_stream_b))
        return input_or, output_or
    
class BinaryLFSR(Codec):
    """
    Binary LFSR-based compression codec.
    """
    def get_name(self):
        return "BinaryLFSR"
    
    def __init__(self):
        pass
    
    def compress(self, stream):
        if len(stream) == 0: return 0
        return np.sum(stream)
    
    def decompress(self, compressed_data, length):
        decompress = self.lsfr_rng(compressed_data, length)
        return decompress
    
    def gen(self, num1s, length):
        return self.lsfr_rng(num1s, length)
        
    def get_lfsr_seq(self, width=8):
        polylist = LFSR().get_fpolyList(m=width)
        if polylist is None:
            raise ValueError(f"No polynomials found for width {width}. Try a value between 3 and 31.")
        poly = polylist[np.random.randint(0, len(polylist), 1)[0]]
        while True:
            init_state = np.random.randint(0, 2, width).tolist()
            if sum(init_state) > 0: # Valid if it contains at least one '1'
                break
        L = LFSR(fpoly=poly, initstate=init_state)
        lfsr_seq = []
        for i in range(2**width):
            value = 0
            for j in range(width):
                value = value + L.state[j]*2**(width-1-j)
            lfsr_seq.append(value)
            L.next()
        return lfsr_seq

    def lsfr_rng(self, binary, length):
        bit_width = int(np.log2(length))
        lfsr_rng = self.get_lfsr_seq(width=bit_width)
        bitstream = []
        for cycle in range(length):
            # If our target is greater than the random number at this cycle, output 1
            bit = 1 if binary > lfsr_rng[cycle] else 0
            bitstream.append(bit)
        return bitstream
    
class BinarySA(Codec):
    """
    Binary SA-based compression codec.
    """
    
    def __init__(self):
        pass
    def get_name(self):
        return "BinarySA"
    
    def compress(self, stream):
        if len(stream) == 0: return 0
        return np.sum(stream)
    
    def decompress(self, compressed_data, length):
        decompress = self.streaming_accurate_bitstream_generator(length, compressed_data)
        return decompress

    def gen(self, num1s, length):
        return self.streaming_accurate_bitstream_generator(length, num1s)
    
    def streaming_accurate_bitstream_generator(self, length, num_ones): 
        #"streaming accuracy:charecterising early termination in stochsatic computing"
        stream = []
        sum = length/2
        for i in range(length):
            sum = sum + num_ones
            if sum > length:
                stream.append(1)
                sum = sum - length
            else:
                stream.append(0)
        return stream
    
class DictCodec(Codec):
    """
    Dictionary-based compression codec.
    """
    
    def __init__(self, codebook_size, chunk_size, use_hamming):
        self.codebook_size = codebook_size
        self.chunk_size = chunk_size
        # self.codebook = self.generate_rand_codebook()
        self.codebook = self.get_deterministic_codebook()
        self.use_hamming = use_hamming

    def get_name(self):
        return f"Dict_{self.codebook_size}_{self.chunk_size}_{'Hamming' if self.use_hamming else 'Value'}"
    def get_codebook(self):
        return self.codebook

    def get_deterministic_codebook(self):
        if self.chunk_size == 4 and self.codebook_size == 8:
            return np.array([[0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 1, 0, 1],
                            [1, 0, 1, 1],
                            [1, 1, 1, 1],
                            [0, 1, 0, 0],
                            [1, 0, 0, 1],
                            [1, 1, 0, 1]])
        elif self.chunk_size == 4 and self.codebook_size == 16:
            return np.array([[0, 0, 0, 0],
                [0, 0, 0, 1],
                [1, 0, 1, 0],
                [1, 1, 0, 1],
                [1, 1, 1, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 1],
                [0, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 1, 1, 1],
                [0, 0, 1, 0],
                [0, 0, 1, 1],
                [1, 1, 1, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1]])

        elif self.chunk_size == 8 and self.codebook_size == 8:
            return np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 1, 0],
                [1, 1, 0, 0, 0, 0, 1, 1],
                [1, 1, 0, 0, 0, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [1, 0, 1, 1, 1, 1, 1, 1]])
        elif self.chunk_size == 8 and self.codebook_size == 16:
            return np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 1, 0, 1, 0, 1, 1],
                [0, 0, 1, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 0, 1],
                [1, 1, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 1, 0, 0, 0],
                [0, 1, 0, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 1, 0, 0],
                [0, 1, 1, 1, 0, 1, 1, 1],
                [1, 0, 1, 1, 1, 1, 1, 1]])
        elif self.chunk_size == 8 and self.codebook_size == 32:
            return np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 1],
                [1, 1, 0, 0, 0, 1, 1, 0],
                [1, 1, 0, 1, 1, 0, 1, 0],
                [1, 1, 1, 1, 0, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0],
                [1, 0, 1, 1, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 1],
                [1, 1, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 0, 1, 1, 1, 1],
                [1, 1, 1, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [1, 1, 0, 1, 0, 0, 0, 0],
                [1, 1, 0, 1, 1, 0, 0, 0],
                [0, 1, 0, 0, 1, 1, 1, 1],
                [1, 0, 1, 1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 1, 1],
                [1, 0, 1, 0, 1, 0, 1, 1],
                [1, 1, 0, 1, 0, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 1, 0]])
        
        elif self.chunk_size == 16 and self.codebook_size == 16:
            return np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0],
                [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
                [1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0],
                [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
                [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1],
                [0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]])
        elif self.chunk_size == 16 and self.codebook_size == 32:
            return np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
                [1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
                [1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0],
                [1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1],
                [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
                [1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
                [1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
                [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]])
        elif self.chunk_size == 16 and self.codebook_size == 64:
            return np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                [1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1],
                [0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
                [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
                [0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1],
                [1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1],
                [1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
                [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0],
                [1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1],
                [1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0],
                [0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                [0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
                [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
                [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0],
                [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0],
                [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
                [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                [1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])

        else:
            raise NotImplementedError("Deterministic codebook is only implemented for chunk_size=4 and codebook_size=8. Please generate a random codebook for other configurations.")
        
    def get_compression_ratio(self):
        return float(self.chunk_size) / int(np.log2(self.codebook_size))

    def generate_rand_codebook(self):
        """
        Generates a codebook that is:
        1. Unique: No duplicate keywords.
        2. Balanced: Rotates through weights (0 ones, 1 one... N ones) to ensure coverage.
        """
        codebook = []
        seen_keywords = set()

        # Create a list of weights to cycle through: [0, 1, 2, ..., 8, 0, 1...]
        # We prioritize the "edges" (0 and max) because they are hardest to hit randomly.
        possible_weights = list(range(self.chunk_size + 1))

        # Iterator that cycles endlessly through weights
        import itertools
        weight_cycler = itertools.cycle(possible_weights)

        attempts = 0
        max_attempts = self.codebook_size * 50  # Safety break

        while len(codebook) < self.codebook_size and attempts < max_attempts:
            target_weight = next(weight_cycler)

            # Try to find a unique keyword with this specific weight
            # We give it a few tries (limited because some weights like 0 have only 1 option)
            for _ in range(20):
                attempts += 1

                # Generate candidate
                candidate = np.zeros(self.chunk_size, dtype=int)
                candidate[:target_weight] = 1
                np.random.shuffle(candidate)

                # Check Uniqueness (convert to tuple to make it hashable for set)
                candidate_tuple = tuple(candidate)

                if candidate_tuple not in seen_keywords:
                    seen_keywords.add(candidate_tuple)
                    codebook.append(candidate)
                    break # Move to next weight in the cycle

        # If we exited due to lack of options (e.g. we wanted 5 "zeros" but only 1 exists),
        # we might be short. Fill the rest with random unique ones from the middle weights.
        while len(codebook) < self.codebook_size:
            # Generate random weight between 1 and chunk_size-1
            w = np.random.randint(1, self.chunk_size)
            candidate = np.zeros(self.chunk_size, dtype=int)
            candidate[:w] = 1
            np.random.shuffle(candidate)
            candidate_tuple = tuple(candidate)

            if candidate_tuple not in seen_keywords:
                seen_keywords.add(candidate_tuple)
                codebook.append(candidate)

        return np.array(codebook)
    
    def compress_hamming_based(self, stream):
        #compress, decompress together
        num_chunks = len(stream) // self.chunk_size
        reconstructed_chunks = []

        for i in range(num_chunks):
            chunk = stream[i*self.chunk_size : (i+1)*self.chunk_size]
            distances = np.sum(np.abs(self.codebook - chunk), axis=1)
            best_idx = np.argmin(distances) # In case of a tie this consitently picks the same keyword which could be problamatic?

            reconstructed_chunks.append(self.codebook[best_idx])

        return np.concatenate(reconstructed_chunks)
    def dictionary_compression_value_based(self, stream):
        num_chunks = len(stream) // self.chunk_size
        reconstructed_chunks = []

        # Pre-calculate codebook weights (number of 1s) outside the loop for speed
        # Shape: (num_keywords,)
        codebook_weights = np.sum(self.codebook, axis=1)

        # Define a scaling factor.
        # It must be larger than the max possible Hamming distance (chunk_size)
        # to ensure Value Error strictly dominates the score.
        priority_factor = self.chunk_size + 1

        for i in range(num_chunks):
            chunk = stream[i*self.chunk_size : (i+1)*self.chunk_size]

            # 1. Calculate the weight (value) of the current chunk
            chunk_weight = np.sum(chunk)

            # 2. Primary Metric: Absolute Value Error
            # How far is the chunk's value from the keyword's value?
            value_errors = np.abs(codebook_weights - chunk_weight)

            # 3. Secondary Metric: Hamming Distance
            # How many bits are physically different?
            hamming_dists = np.sum(np.abs(self.codebook - chunk), axis=1)

            # 4. Combined Score (Scalarization)
            # Low score is better. Because of the factor, a keyword with
            # Value Error 0 and Hamming Dist 4 will beat a keyword with
            # Value Error 1 and Hamming Dist 0.
            final_scores = (value_errors * priority_factor) + hamming_dists

            best_idx = np.argmin(final_scores)
            reconstructed_chunks.append(self.codebook[best_idx])

        if not reconstructed_chunks:
            return np.array([])
        return np.concatenate(reconstructed_chunks)

    def compress(self, stream):
        #compress, decompress together
        if self.use_hamming:
            return self.compress_hamming_based(stream)
        else:
            return self.dictionary_compression_value_based(stream)
    def decompress(self, compressed_data, length):
        return compressed_data
    
    def gen(self, num1s, length):
        raise NotImplementedError("Dictionary codec does not support direct generation. Use a separate method to create streams and then compress them.")
    
class SamplingOdd(Codec):
    
    def __init__(self):
        pass
    def get_name(self):
        return "SamplingOdd"
    def compress(self,arr):
        # decompress together
        """
        Takes a numpy array of bits.
        Returns a compressed array containing: [Odd Bits ... Flag]
        """
        # 1. Extract bits at Odd indices (1, 3, 5...)
        odd_bits = arr[1::2]

        # 2. Extract bits at Even indices (0, 2, 4...) to calculate flag
        even_bits = arr[0::2]

        # 3. Calculate Flag
        # If the array is empty, handle gracefully
        if len(even_bits) == 0:
            return np.array([0])

        # Check if majority of even bits are 1
        if np.mean(even_bits) > 0.5:
            flag = 1
        else:
            flag = 0

        # 3. Determine Output Size
        target_length = len(arr)

        reconstructed = np.zeros(target_length, dtype=int)

        # 4. Fill Even Positions (0, 2, 4...) with Flag
        reconstructed[0::2] = flag

        # 5. Fill Odd Positions (1, 3, 5...) with stored bits
        # We calculate how many spots we actually have in the target array
        slots_available = len(reconstructed[1::2])

        # Slice stored_odds to fit the available slots (prevents length mismatch errors)
        reconstructed[1::2] = odd_bits[:slots_available]

        return reconstructed
    def decompress(self, compressed_data, length):
        # decompress together
        return compressed_data
    
    def gen(self, num1s, length):
        raise NotImplementedError("SamplingOdd codec does not support direct generation. Use a separate method to create streams and then compress them.")
    
class XORFold(Codec):
    def __init__(self, num_folds):
        self.num_folds = num_folds
    def get_name(self):
        return f"XORFold_{self.num_folds}"
    def compress(self,stream):
        #compress
        chunk_size = int(len(stream) / (self.num_folds + 1))
        compressed = []
        for i in range(self.num_folds + 1):
            start = i * chunk_size
            chunk = stream[start : start + chunk_size]
            if i == 0:
                compressed = chunk
            else:
                compressed = np.bitwise_xor(compressed, chunk).tolist()

        #decompress
        decompressed = []
        for i in range(self.num_folds):
            for j in range(chunk_size):
                decompressed.append(random.randint(0, 1))
                compressed[j] = compressed[j] ^ decompressed[-1]
        decompressed.extend(compressed)
        return decompressed
    def decompress(self, compressed_data, length):
        #decompress
        return compressed_data
    def gen(self, num1s, length):
        raise NotImplementedError("XORFold codec does not support direct generation. Use a separate method to create streams and then compress them.")

class LowerPrecision(Codec):
    def __init__(self, precision):
        self.precision = precision
    def get_name(self):
        return f"LowerPrecision_{self.precision}"
    def compress(self, stream):
        if len(stream) < self.precision:
            raise ValueError(f"Stream length {len(stream)} is less than the specified precision {self.precision}.")
        compress = stream[:self.precision]
        n = int(len(stream)/self.precision)
        decompress = np.tile(compress, n)
        return decompress
    def decompress(self, compressed_data, length):
        return compressed_data
    def gen(self, num1s, length):
        raise NotImplementedError("LowerPrecision codec does not support direct generation. Use a separate method to create streams and then compress them.")

class HistogramRandom(Codec2D):
    def __init__(self):
        pass
    def get_name(self):
        return "HistogramRandom"
    def compress(self, stream_a, stream_b):
        # --- COMPRESSION (Counting) ---
        zero_zero = 0
        zero_one = 0
        one_zero = 0
        one_one = 0

        # Using zip is cleaner and faster than iterating by index
        for a, b in zip(stream_a, stream_b):
            if a == 0 and b == 0:
                zero_zero += 1
            elif a == 0 and b == 1:
                zero_one += 1
            elif a == 1 and b == 0:
                one_zero += 1
            elif a == 1 and b == 1:
                one_one += 1

        # --- DECOMPRESSION (Randomized Reconstruction) ---

        # 1. Create a list containing all the pairs we counted
        all_pairs = []
        all_pairs.extend([(0, 0)] * zero_zero)
        all_pairs.extend([(0, 1)] * zero_one)
        all_pairs.extend([(1, 0)] * one_zero)
        all_pairs.extend([(1, 1)] * one_one)

        # 2. Shuffle the list to randomize the order
        random.shuffle(all_pairs)

        # 3. Unzip the pairs back into two separate streams
        # (Handle edge case where streams might be empty)
        if all_pairs:
            decompressed_a, decompressed_b = map(list, zip(*all_pairs))
        else:
            decompressed_a, decompressed_b = [], []

        return decompressed_a, decompressed_b
    def decompress(self, compressed_a, compressed_b, shape):
        return compressed_a, compressed_b
    def gen(self, num1s1, num1s2, length):
        raise NotImplementedError("HistogramRandom codec does not support direct generation. Use a separate method to create streams and then compress them.")

class BinaryUncorrelated(Codec2D):
    def __init__(self):
        pass
    def get_name(self):
        return "BinaryUncorrelated"
    def compress(self, stream_a, stream_b):
        """
        Reorders stream_b to ensure the number of overlapping '1's
        matches the theoretical expectation for independent streams.
        """
        stream_a = np.array(stream_a)
        stream_b = np.array(stream_b)
        n = len(stream_a)
        count_a = np.sum(stream_a)
        count_b = np.sum(stream_b)

        # 1. Calculate the target number of overlaps for ZCE = 0
        # Expected Overlaps = P(A) * P(B) * N
        target_overlaps = int(round((count_a * count_b) / n))

        # 2. Identify available slots in Stream A
        # Indices where A is 1
        assert len(np.where(stream_a == 1))>0, "Stream A must contain at least one '1' for this codec to work."
        slots_where_A_is_1 = np.where(stream_a == 1)[0]
        # Indices where A is 0
        assert len(np.where(stream_a == 0))>0, "Stream A must contain at least one '0' for this codec to work."
        slots_where_A_is_0 = np.where(stream_a == 0)[0]

        # 3. Construct new Stream B
        new_b = np.zeros(n, dtype=int)

        # Place '1's where A is '1' (to satisfy the overlap requirement)
        # We randomly pick 'target_overlaps' indices from the slots where A is 1
        chosen_overlaps = np.random.choice(slots_where_A_is_1, target_overlaps, replace=False)
        new_b[chosen_overlaps] = 1

        # Place the remaining '1's where A is '0' (non-overlapping)
        remaining_ones = count_b - target_overlaps

        # Safety check: ensure we have enough zeros in A to hold the rest of B
        if remaining_ones > len(slots_where_A_is_0):
            # This happens if streams are very dense; perfect decorrelation is impossible.
            # We fill all 0-slots and dump the rest back into 1-slots.
            new_b[slots_where_A_is_0] = 1
            leftover = remaining_ones - len(slots_where_A_is_0)
            # Put remaining ones in the empty spots of A-is-1
            remaining_A1_slots = list(set(slots_where_A_is_1) - set(chosen_overlaps))
            extra_fill = np.random.choice(remaining_A1_slots, leftover, replace=False)
            new_b[extra_fill] = 1
        else:
            chosen_non_overlaps = np.random.choice(slots_where_A_is_0, remaining_ones, replace=False)
            new_b[chosen_non_overlaps] = 1

        return stream_a, new_b
    def decompress(self, compressed_a, compressed_b, shape):
        return compressed_a, compressed_b
    def gen(self, num1s1, num1s2, length):
        stream_a = [1]*num1s1 + [0]*(length - num1s1)
        stream_b = [1]*num1s2 + [0]*(length - num1s2)
        compressed_a, compressed_b = self.compress(stream_a, stream_b)
        return compressed_a, compressed_b

class BinaryPositivelyCorrelated(Codec2D):
    def __init__(self):
        pass
    def get_name(self):
        return "BinaryPositivelyCorrelated"
    def compress(self, stream_a, stream_b):
        return np.sum(stream_a), np.sum(stream_b)
    def decompress(self, compressed_a, compressed_b, shape):
        pos_a = np.ones(compressed_a, dtype=int)
        pos_b = np.ones(compressed_b, dtype=int)
        print(f"Decompressing with positive correlation: compressed_a={compressed_a}, compressed_b={compressed_b}, shape={shape}")
        # Fill the rest with zeros
        if compressed_a < shape:
            pos_a = np.concatenate([pos_a, np.zeros(shape - compressed_a, dtype=int)])
        if compressed_b < shape:
            pos_b = np.concatenate([pos_b, np.zeros(shape - compressed_b, dtype=int)])
        return pos_a, pos_b
    def gen(self, num1s1, num1s2, length):
        return self.decompress(num1s1, num1s2, length)
class BinaryNegativelyCorrelated(Codec2D):
    def __init__(self):
        pass
    def get_name(self):
        return "BinaryNegativelyCorrelated"
    def compress(self, stream_a, stream_b):
        return np.sum(stream_a), np.sum(stream_b)
    def decompress(self, compressed_a, compressed_b, shape):
        neg_a = np.ones(compressed_a, dtype=int)
        neg_b = np.zeros(shape-compressed_b, dtype=int)
        # Fill the rest with zeros
        if compressed_a < shape:
            neg_a_ = np.concatenate([neg_a, np.zeros(shape - compressed_a, dtype=int)])
        if compressed_b < shape:
            neg_b_ = np.concatenate([neg_b, np.ones(compressed_b, dtype=int)])
        return neg_a_, neg_b_
    def gen(self, num1s1, num1s2, length):
        return self.decompress(num1s1, num1s2, length)
 
# dictcodec = DictCodec(codebook_size=8, chunk_size=8, use_hamming=False)
# np.set_printoptions(threshold=np.inf, linewidth=200)
# print(f"Codebook for {dictcodec.get_name()}:")
# print(dictcodec.get_codebook())

# dictcodec2 = DictCodec(codebook_size=16, chunk_size=16, use_hamming=False)
# np.set_printoptions(threshold=np.inf, linewidth=200)
# print(f"Codebook for {dictcodec2.get_name()}:")
# print(dictcodec2.get_codebook())

input_stream_a = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
bin_rand  = BinaryLFSR()
sa_in = bin_rand.get_streaming_accuracy(input_stream_a, len(input_stream_a), sum(input_stream_a))

input_stream_b = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
sa_out = bin_rand.get_streaming_accuracy(input_stream_b, len(input_stream_b), sum(input_stream_b))
print(sa_in, sa_out)

best, worst = bin_rand.get_best_and_worst_bitstreams(32, 15)
print(f"Best bitstream: {best}, Worst bitstream: {worst}")
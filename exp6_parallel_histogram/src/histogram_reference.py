# Experiment 6: Parallel Histogram — data generation + CPU correctness oracle
# hps_gpu_cuda_lab

import time

import numpy as np
import torch


def generate_uniform(N, num_bins, seed=None):
    g = torch.Generator(device='cuda').manual_seed(seed) if seed is not None else None
    return torch.randint(0, num_bins, (N,), dtype=torch.int32, device='cuda', generator=g)


def generate_skewed(N, num_bins, num_hot=1, hot_mass=0.9, seed=None):
    g = torch.Generator(device='cuda').manual_seed(seed) if seed is not None else None
    num_hot = max(1, min(num_hot, num_bins))
    num_cold = num_bins - num_hot

    probs = torch.empty(num_bins, device='cuda', dtype=torch.float32)
    probs[:num_hot] = hot_mass / num_hot
    probs[num_hot:] = (1.0 - hot_mass) / num_cold if num_cold > 0 else 0.0

    bins = torch.multinomial(probs, N, replacement=True, generator=g)
    return bins.to(torch.int32)


def cpu_histogram(data, num_bins):
    data_cpu = data.cpu().numpy()
    return np.bincount(data_cpu, minlength=num_bins)


def cpu_bincount_timer(data, num_bins, warmup=3, runs=20):
    # .cpu() transfer happens outside the timed region — isolates pure
    # np.bincount compute time, matching what a CPU-only pipeline pays.
    data_cpu = data.cpu().numpy()

    for _ in range(warmup):
        np.bincount(data_cpu, minlength=num_bins)

    start = time.perf_counter()
    for _ in range(runs):
        np.bincount(data_cpu, minlength=num_bins)
    elapsed = time.perf_counter() - start

    return elapsed / runs * 1000.0  # ms, same units as gpu_timer


def verify(gpu_result, cpu_result, kernel_name="naive", max_print=20):
    gpu_result_c = gpu_result.cpu().numpy()

    if np.array_equal(gpu_result_c, cpu_result):
        print(f"Kernel: {kernel_name}, verified!")
        return True

    print(f"Kernel: {kernel_name}, failed!")
    indices = np.where(cpu_result != gpu_result_c)[0]
    values_cpu = cpu_result[indices]
    values_gpu = gpu_result_c[indices]

    print(f"------------------- Mismatches ({len(indices)} total) --------------------")
    for index, cpu_val, gpu_val in list(zip(indices, values_cpu, values_gpu))[:max_print]:
        print(f"Index: {index}, cpu_val={cpu_val}, gpu_val={gpu_val}")
    if len(indices) > max_print:
        print(f"... and {len(indices) - max_print} more")
    return False

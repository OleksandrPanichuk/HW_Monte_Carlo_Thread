import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np

CHUNK_SIZE = 5_000_000
N_VALUES = [100_000_000_000]
M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
RESULTS_DIR = 'results'
FILE_PATH = os.path.join(RESULTS_DIR, 'pi_monte_carlo_parallel_results.csv')


def split_batches(total_points, workers):
    base_batch = total_points // workers
    remainder = total_points % workers
    return [base_batch + (1 if index < remainder else 0) for index in range(workers)]


def monte_carlo_batch(batch_size):
    rng = np.random.default_rng()
    inside_circle = 0
    remaining = batch_size

    while remaining > 0:
        current_batch = min(CHUNK_SIZE, remaining)
        xy = rng.random((current_batch, 2), dtype=np.float32)
        inside_circle += int(np.sum(xy[:, 0] * xy[:, 0] + xy[:, 1] * xy[:, 1] <= 1.0))
        remaining -= current_batch

    return inside_circle


def estimate_pi_parallel(total_points, workers):
    batches = split_batches(total_points, workers)
    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        counts = list(executor.map(monte_carlo_batch, batches))

    inside_circle = sum(counts)
    pi_estimate = 4 * inside_circle / total_points
    execution_time = time.perf_counter() - start_time
    return pi_estimate, execution_time


def run_benchmark():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(FILE_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['N \\ M', *M_VALUES])

        for total_points in N_VALUES:
            row: list[int | float] = [total_points]
            for workers in M_VALUES:
                print(f'Running N={total_points:,}, M={workers}...', flush=True)
                _, execution_time = estimate_pi_parallel(total_points, workers)
                row.append(round(execution_time, 4))
                print(f'  done in {execution_time:.4f}s', flush=True)
            writer.writerow(row)
            file.flush()

    print(f'Results saved to {FILE_PATH}')


if __name__ == '__main__':
    run_benchmark()

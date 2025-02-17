import platform
import psutil
import timeit
import numpy as np
import time
import multiprocessing


# 1. Tipul și frecvența procesorului
def get_cpu_info():
    cpu_name = platform.processor()
    cpu_freq = psutil.cpu_freq().current
    return cpu_name, cpu_freq


# 2. Dimensiunea memoriei RAM
# 1 kilobyte (KB) = 1024 bytes.
# 1 megabyte (MB) = 1024 KB.
# 1 gigabyte (GB) = 1024 MB.
def get_memory_info():
    total_memory = psutil.virtual_memory().total / (1024 ** 3)  # Convert to GB
    return total_memory


# 3. Viteza de transfer a unui bloc de date
def test_transfer_speed():
    main_block = []
    block_size = 10000000  # 1MB block   aici poate fi cu inca un 0 pt 10MB
    transfer_time = 0.0

    for i in range(4):
        data_block = np.random.rand(block_size)
        main_block.append(data_block)

    for block in main_block:
        start_time = time.time()
        transfer = block.copy()
        end_time = time.time()
        transfer_time += end_time - start_time

    return transfer_time


def test_transfer_speed_multiple_blocks_with_gaps(block_size=10000000, gap_size=1000000):
    main_blocks = []

    # 4 main data blocks with gaps in between
    for i in range(4):
        # main memory block
        main_blocks.append(np.random.rand(block_size))

        # Add a gap block
        if i < 3:  # Only add gaps between main blocks
            main_blocks.append(np.zeros(gap_size))

    # Measure time
    total_transfer_time = 0.0

    i = 0
    for block in main_blocks:
        if i % 2 == 0:
            start_time = time.time()
            copied_block = block.copy()  # Copy operation
            end_time = time.time()
            total_transfer_time += (end_time - start_time)
        i += 1

        # Accumulate the time taken for copying each main block
        # total_transfer_time += (end_time - start_time)

    return total_transfer_time


# 4. Viteza de execuție a operațiilor aritmetice și logice
def test_arithmetic_operations():
    setup_code = """
import numpy as np
a = np.random.rand(1000000)
"""
    test_code = """
b = a * 2
"""
    exec_time = timeit.timeit(stmt=test_code, setup=setup_code, number=100)
    return exec_time


# TFLOPS:
def test_tflops():
    """
    Measure the system's floating-point operations per second (FLOPS) performance.
    Returns the calculated performance in TFLOPS.
    """
    n = 10 ** 7  # Size of arrays (10 million elements)
    iterations = 100  # Number of iterations

    # Initialize large arrays with random values
    a = np.random.rand(n)
    b = np.random.rand(n)
    c = np.zeros(n)  # Output array

    total_operations = 2 * n * iterations  # Each iteration does 2 operations: multiply and add

    # Start timing
    start_time = time.time()
    for _ in range(iterations):
        c = a * b + c  # Floating-point operations: multiplication and addition
    end_time = time.time()

    # Calculate FLOPS
    execution_time = end_time - start_time
    flops = total_operations / execution_time
    tflops = flops / 1e12  # 10^12

    return tflops, execution_time


# TFLOPS for a single process
def compute_max_tflops_optimized(process_id, duration, result_dict):
    """
    Simulate maximum TFLOP computation using optimized vectorized operations.
    Args:
        process_id (int): ID of the process.
        duration (float): Duration in seconds for the computation.
        result_dict (dict): Dictionary to store results from processes.
    """
    n = 10 ** 8  # Increase array size for larger workload
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)

    total_operations = 0
    start_time = time.time()

    while time.time() - start_time < duration:
        # Perform multiple operations to maximize FLOPS
        c = a * b + c * a - b * a + c * b
        total_operations += 6 * n  # 6 operations per element per iteration

    execution_time = time.time() - start_time
    flops = total_operations / execution_time
    tflops = flops / 1e12  # Convert FLOPS to TFLOPS
    timp = execution_time / total_operations  # Compute `timp`
    frequency_hz = 1.0 / timp if timp > 0 else 0  # Compute frequency in Hz
    frequency_mhz = frequency_hz / 1e6  # Convert Hz to MHz

    print(f"Process {process_id} executed with timp={timp:.10f} seconds and frequency={frequency_mhz:.2f} MHz.")

    # Store the results in the shared dictionary
    result_dict[process_id] = (tflops, execution_time, timp, frequency_mhz)




# Funcție principală
def main():
    cpu_name, cpu_freq = get_cpu_info()
    total_memory = get_memory_info()

    transfer_time = test_transfer_speed()
    arithmetic_time = test_arithmetic_operations()
    tflops, exec_time = test_tflops()

    print(f"Procesor: {cpu_name}")
    print(f"Frecvența procesorului: {cpu_freq} MHz")
    print(f"Memorie RAM totală: {total_memory:.2f} GB")
    print(f"Timp transfer bloc date: {transfer_time:.6f} secunde")
    print(f"Timp transfer bloc date cu goluri: {test_transfer_speed_multiple_blocks_with_gaps():.6f} secunde")
    print(f"Timp execuție operații aritmetice: {arithmetic_time:.6f} secunde")
    print(f"TFLOPS measured: {tflops:.4f} TFLOPS")
    print(f"Execution time: {exec_time:.6f} seconds")

    # Simulate 4 processes performing TFLOP computations
    num_processes = 4  # Number of processes to spawn
    computation_duration = 5  # Duration of computation in seconds
    manager = multiprocessing.Manager()
    result_dict = manager.dict()  # Shared dictionary to store results

    processes = []
    for i in range(num_processes):
        p = multiprocessing.Process(target=compute_max_tflops_optimized, args=(i, computation_duration, result_dict))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Display results
    print("\nMaximized TFLOP Results for Each Process:")
    total_tflops = 0
    for process_id, (tflops, exec_time, timp, frequency_mhz) in result_dict.items():
        print(f"Process {process_id}: TFLOPS={tflops:.4f}, Execution Time={exec_time:.6f} seconds, "
              f"Timp={timp:.10f} seconds, Frequency={frequency_mhz:.2f} MHz")
        total_tflops += tflops

    print(f"\nTotal TFLOPS across all processes: {total_tflops:.4f} TFLOPS")


if __name__ == "__main__":
    main()

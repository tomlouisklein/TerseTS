"""
Benchmarking script for TerseTS Mix-Piece and Sim-Piece algorithms.
This version works with the actual TerseTS Python API.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import tersets
from tersets import Method

# After you fix the bindings, you can use Method.SimPiece and Method.MixPiece
# For now, as a workaround, you could modify the compress function

def benchmark_compression(
    data: np.ndarray, 
    method: Method, 
    error_bound: float
) -> Dict[str, float]:
    """Benchmark a single compression method"""
    
    # Convert numpy array to list of floats
    values = data.astype(float).tolist()
    
    # Time the compression
    start_time = time.time()
    compressed = tersets.compress(values, method, error_bound)
    compression_time = time.time() - start_time
    
    # Time the decompression
    start_time = time.time()
    decompressed = tersets.decompress(compressed)
    decompression_time = time.time() - start_time
    
    # Convert decompressed back to numpy for analysis
    decompressed_array = np.array(decompressed)
    
    # Calculate metrics
    original_size = data.nbytes
    compressed_size = len(compressed)
    compression_ratio = original_size / compressed_size
    
    # Calculate approximation error
    mae = np.mean(np.abs(data - decompressed_array))
    rmse = np.sqrt(np.mean((data - decompressed_array) ** 2))
    max_error = np.max(np.abs(data - decompressed_array))
    
    return {
        'method': method.name,
        'compression_time': compression_time,
        'decompression_time': decompression_time,
        'compression_ratio': compression_ratio,
        'compressed_size': compressed_size,
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'error_bound': error_bound
    }


def generate_test_datasets() -> Dict[str, np.ndarray]:
    """Generate various test datasets"""
    datasets = {}
    
    # 1. Clean sine wave
    t = np.linspace(0, 100, 10000)
    datasets['sine_clean'] = np.sin(t)
    
    # 2. Noisy sine wave
    np.random.seed(42)
    datasets['sine_noisy'] = np.sin(t) + 0.1 * np.random.randn(len(t))
    
    # 3. Multiple frequencies
    datasets['multi_freq'] = (np.sin(t) + 
                             0.5 * np.sin(10*t) + 
                             0.25 * np.sin(50*t))
    
    # 4. Random walk (trending data)
    datasets['random_walk'] = np.cumsum(np.random.randn(10000))
    
    # 5. Seasonal pattern with trend
    datasets['seasonal_trend'] = np.sin(2 * np.pi * t / 10) + 0.01 * t
    
    # 6. Step function
    datasets['step_function'] = np.repeat(np.random.randn(100), 100)
    
    return datasets


def run_benchmark_suite():
    """Run complete benchmark suite"""
    
    # Methods to benchmark
    methods_to_test = [
        Method.SwingFilter,
        Method.SlideFilter,
        Method.SimPiece,
        Method.MixPiece  
    ]
    
    # Generate datasets
    print("Generating test datasets...")
    datasets = generate_test_datasets()
    
    # Results storage
    all_results = []
    
    # For each dataset
    for dataset_name, data in datasets.items():
        print(f"\nBenchmarking dataset: {dataset_name}")
        print(f"  Shape: {data.shape}, Range: [{np.min(data):.3f}, {np.max(data):.3f}]")
        
        # Calculate error bounds as percentage of range
        data_range = np.max(data) - np.min(data)
        error_percents = [0.5, 1.0, 2.0, 5.0]  # Percentage of range
        
        for error_percent in error_percents:
            error_bound = (error_percent / 100) * data_range
            print(f"\n  Error bound: {error_percent}% ({error_bound:.3f})")
            
            for method in methods_to_test:
                print(f"    Testing {method.name}...", end=" ")
                try:
                    result = benchmark_compression(data, method, error_bound)
                    result['dataset'] = dataset_name
                    result['error_percent'] = error_percent
                    all_results.append(result)
                    print(f"CR: {result['compression_ratio']:.1f}")
                except Exception as e:
                    print(f"Failed: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Save results
    df.to_csv('tersets_benchmark_results.csv', index=False)
    print("\nResults saved to 'tersets_benchmark_results.csv'")
    
    return df


def plot_results(df: pd.DataFrame):
    """Create visualizations of benchmark results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('TerseTS Algorithm Benchmarking Results')
    
    # 1. Compression Ratio vs Error Bound
    ax = axes[0, 0]
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        avg_by_error = method_data.groupby('error_percent')['compression_ratio'].mean()
        ax.plot(avg_by_error.index, avg_by_error.values, 
                marker='o', label=method, linewidth=2)
    
    ax.set_xlabel('Error Bound (% of range)')
    ax.set_ylabel('Compression Ratio')
    ax.set_title('Compression Ratio vs Error Bound')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Compression Time Comparison
    ax = axes[0, 1]
    avg_times = df.groupby('method')['compression_time'].mean()
    avg_times.plot(kind='bar', ax=ax)
    ax.set_xlabel('Method')
    ax.set_ylabel('Average Compression Time (s)')
    ax.set_title('Compression Speed Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. MAE vs Compression Ratio
    ax = axes[1, 0]
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        ax.scatter(method_data['compression_ratio'], 
                  method_data['mae'],
                  label=method, alpha=0.6, s=50)
    
    ax.set_xlabel('Compression Ratio')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Accuracy vs Compression Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Dataset-specific comparison
    ax = axes[1, 1]
    # Use 2% error bound for comparison
    subset = df[df['error_percent'] == 2.0]
    pivot = subset.pivot_table(
        index='dataset', 
        columns='method', 
        values='compression_ratio',
        aggfunc='mean'
    )
    
    pivot.plot(kind='bar', ax=ax)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Compression Ratio')
    ax.set_title('Dataset-specific Performance (2% error bound)')
    ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('tersets_benchmark_plots.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_summary(df: pd.DataFrame):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    # Average compression ratios
    print("\nAverage Compression Ratios:")
    avg_cr = df.groupby('method')['compression_ratio'].mean().sort_values(ascending=False)
    for method, cr in avg_cr.items():
        print(f"  {method:<25}: {cr:>8.1f}")
    
    # Average compression times
    print("\nAverage Compression Times (ms):")
    avg_time = df.groupby('method')['compression_time'].mean() * 1000
    for method, time_ms in avg_time.sort_values().items():
        print(f"  {method:<25}: {time_ms:>8.1f} ms")
    
    # Average MAE
    print("\nAverage Mean Absolute Error:")
    avg_mae = df.groupby('method')['mae'].mean().sort_values()
    for method, mae in avg_mae.items():
        print(f"  {method:<25}: {mae:>8.4f}")
    
    # If MixPiece is available, show relative performance
    if 'MixPiece' in avg_cr.index and 'SimPiece' in avg_cr.index:
        print(f"\nMix-Piece vs Sim-Piece:")
        improvement = (avg_cr['MixPiece'] / avg_cr['SimPiece'] - 1) * 100
        print(f"  Compression ratio improvement: {improvement:.1f}%")


def main():
    """Main execution"""
    print("TerseTS Mix-Piece and Sim-Piece Benchmarking")
    print("=" * 60)
    
    # Check available methods
    print("\nAvailable compression methods:")
    for method in Method:
        print(f"  {method.value}: {method.name}")
    
    # Run benchmarks
    df = run_benchmark_suite()
    
    # Create plots
    print("\nGenerating plots...")
    plot_results(df)
    
    # Print summary
    print_summary(df)
    
    print("\nBenchmarking complete!")


# Workaround if you can't modify the bindings
def compress_with_method_index(values: List[float], method_index: int, error_bound: float) -> bytes:
    """
    Workaround to use method indices directly.
    This creates a fake Method enum value.
    """
    # Create a mock configuration
    from ctypes import Structure, c_byte, c_float, byref
    from tersets import _TersetsPython__library as library  # Access private library
    from tersets import _TersetsPython__UncompressedValues as UncompressedValues
    from tersets import _TersetsPython__CompressedValues as CompressedValues
    from tersets import _TersetsPython__Configuration as Configuration
    
    uncompressed_values = UncompressedValues()
    uncompressed_values.data = (c_double * len(values))(*values)
    uncompressed_values.len = len(values)
    
    compressed_values = CompressedValues()
    configuration = Configuration(method_index, error_bound)
    
    error = library.compress(
        uncompressed_values, byref(compressed_values), configuration
    )
    
    if error != 0:
        raise ValueError(f"Compression error: {error}")
    
    return compressed_values.data[:compressed_values.len]


if __name__ == "__main__":
    main()
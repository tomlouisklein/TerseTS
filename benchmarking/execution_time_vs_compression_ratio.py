"""
TerseTS Execution Time vs Compression Ratio Benchmark
This script measures execution time and plots it against compression ratio for TerseTS algorithms.
Based on the Mix-Piece paper's methodology (Figure 13).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from typing import Dict, List, Tuple
import tersets
from tersets import Method
from ucr_dataloader import prepare_ucr_datasets

def benchmark_algorithm(data: np.ndarray, method: Method, error_bound: float, 
                       num_runs: int = 25) -> Tuple[float, float]:
    """
    Benchmark a single algorithm, returning compression ratio and amortized execution time.
    
    Returns:
        Tuple of (compression_ratio, amortized_time_per_record_ms)
    """
    try:
        # Convert numpy array to list of floats
        values = data.astype(float).tolist()
        data_length = len(data)
        
        # Measure execution time over multiple runs
        execution_times = []
        compressed_size = 0
        
        for _ in range(num_runs):
            start_time = time.perf_counter()
            compressed = tersets.compress(values, method, error_bound)
            end_time = time.perf_counter()
            
            execution_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            compressed_size = len(compressed)
        
        # Calculate average execution time in milliseconds
        avg_execution_time_ms = np.mean(execution_times)
        
        # Calculate amortized time per record
        amortized_time_per_record = avg_execution_time_ms / data_length
        
        # Calculate compression ratio
        original_size = data.nbytes
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        return compression_ratio, amortized_time_per_record
        
    except Exception as e:
        print(f"Error with method {method.name} and error bound {error_bound}: {e}")
        return 0, 0

def plot_execution_time_vs_compression(datasets: Dict[str, np.ndarray], 
                                      error_percentages: List[float]):
    """
    Create execution time vs compression ratio plots for each dataset.
    """
    
    # Methods to test (consistent with compression ratio benchmark)
    methods_to_test = [
        Method.MixPiece,
        Method.SimPiece,
        Method.PoorMansCompressionMidrange,
        Method.SwingFilter,
        Method.SlideFilter,
        Method.VisvalingamWhyatt,
    ]
    
    # Colors and markers for each method
    method_styles = {
        Method.MixPiece: {'color': 'green', 'marker': 'o', 'markersize': 10, 'label': 'Mix-Piece'},
        Method.SimPiece: {'color': 'blue', 'marker': 's', 'markersize': 8, 'label': 'Sim-Piece'},
        Method.PoorMansCompressionMidrange: {'color': 'orange', 'marker': '^', 'markersize': 8, 'label': 'PMC-MR'},
        Method.SwingFilter: {'color': 'red', 'marker': 'v', 'markersize': 8, 'label': 'Swing'},
        Method.SlideFilter: {'color': 'purple', 'marker': 'd', 'markersize': 8, 'label': 'Slide'},
        Method.VisvalingamWhyatt: {'color': 'brown', 'marker': 'x', 'markersize': 10, 'label': 'Visvalingam-Whyatt'},
    }
    
    # Process each dataset
    for dataset_name, data in datasets.items():
        print(f"\nProcessing {dataset_name}...")
        print(f"  Data length: {len(data):,} points")
        
        # Calculate data range
        data_range = np.max(data) - np.min(data)
        
        # Create figure with subplots for different error bounds
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{dataset_name}: Execution Time vs Compression Ratio Trade-off', 
                     fontsize=14, fontweight='bold')
        
        for idx, error_percent in enumerate(error_percentages):
            ax = axes[idx]
            error_bound = (error_percent / 100) * data_range
            
            print(f"\n  Testing with ε = {error_percent}% of range...")
            
            # Collect results for each method
            results = []
            for method in methods_to_test:
                print(f"    Benchmarking {method.name}...")
                
                # Get compression ratio and amortized execution time
                compression_ratio, amortized_time = benchmark_algorithm(
                    data, method, error_bound, num_runs=25
                )
                
                if compression_ratio > 0:  # Only plot valid results
                    results.append((method, compression_ratio, amortized_time))
                    print(f"      CR: {compression_ratio:.1f}, Amortized time: {amortized_time:.6f} ms/record")
            
            # Plot all results
            for method, cr, amortized_time in results:
                style = method_styles[method]
                ax.scatter(cr, amortized_time, 
                          color=style['color'], marker=style['marker'], 
                          s=style['markersize']**2, label=style['label'],
                          edgecolors='black', linewidth=0.5, zorder=5)
            
            # Customize subplot
            ax.set_xlabel('Compression Ratio', fontsize=12)
            ax.set_ylabel('Amortized Execution Time (ms)', fontsize=12)
            ax.set_title(f'ε = {error_percent}%', fontsize=12)
            ax.grid(True, alpha=0.3, which='both')
            
            # Set y-axis to log scale
            ax.set_yscale('log')
            
            # Format axes properly
            from matplotlib.ticker import ScalarFormatter, LogLocator
            
            # Format y-axis for log scale without scientific notation
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.yaxis.get_major_formatter().set_useOffset(False)
            
            # Format x-axis to avoid scientific notation
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.get_major_formatter().set_scientific(False)
            ax.xaxis.get_major_formatter().set_useOffset(False)
            
            # Set reasonable x-axis limits based on data
            if results:
                min_cr = min(r[1] for r in results) * 0.8
                max_cr = max(r[1] for r in results) * 1.2
                ax.set_xlim(max(0, min_cr), max_cr)
            
            # Add legend only to first subplot
            if idx == 0:
                ax.legend(loc='upper left', fontsize=10)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save the plot
        filename = f'{dataset_name.lower()}_execution_time_vs_compression.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n  Saved plot: {filename}")
        plt.show()
        plt.close()

def create_summary_plot(datasets: Dict[str, np.ndarray], error_percent: float):
    """
    Create a summary plot showing all datasets at a specific error threshold.
    Similar to Figure 13 in the Mix-Piece paper.
    """
    methods_to_test = [
        Method.MixPiece,
        Method.SimPiece,
        Method.SwingFilter,
        Method.SlideFilter,
    ]
    
    # Method styles
    method_styles = {
        Method.MixPiece: {'color': 'green', 'marker': 'o', 'markersize': 10, 'label': 'Mix-Piece'},
        Method.SimPiece: {'color': 'blue', 'marker': 's', 'markersize': 8, 'label': 'Sim-Piece'},
        Method.SwingFilter: {'color': 'red', 'marker': 'v', 'markersize': 8, 'label': 'Swing'},
        Method.SlideFilter: {'color': 'purple', 'marker': 'd', 'markersize': 8, 'label': 'Slide'},
    }
    
    plt.figure(figsize=(10, 8))
    
    print(f"\nCreating summary plot for ε = {error_percent}%...")
    
    # Collect all results first to determine axis limits
    all_results = []
    
    for dataset_name, data in datasets.items():
        data_range = np.max(data) - np.min(data)
        error_bound = (error_percent / 100) * data_range
        
        for method in methods_to_test:
            compression_ratio, amortized_time = benchmark_algorithm(
                data, method, error_bound, num_runs=10  # Fewer runs for summary
            )
            
            if compression_ratio > 0:
                all_results.append((dataset_name, method, compression_ratio, amortized_time))
    
    # Plot all results
    for dataset_name, method, cr, amortized_time in all_results:
        style = method_styles[method]
        plt.scatter(cr, amortized_time,
                   color=style['color'], marker=style['marker'],
                   s=style['markersize']**2, alpha=0.7,
                   edgecolors='black', linewidth=0.5)
        
        # Add dataset label near Mix-Piece point
        if method == Method.MixPiece:
            plt.annotate(dataset_name, 
                       (cr, amortized_time),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
    
    # Create legend with unique entries
    handles = []
    labels = []
    for method in methods_to_test:
        style = method_styles[method]
        handle = plt.scatter([], [], color=style['color'], marker=style['marker'],
                           s=style['markersize']**2, edgecolors='black', linewidth=0.5)
        handles.append(handle)
        labels.append(style['label'])
    
    plt.legend(handles, labels, loc='upper left', fontsize=10)
    plt.xlabel('Compression Ratio', fontsize=12)
    plt.ylabel('Amortized Execution Time (ms)', fontsize=12)
    plt.title(f'Execution Time vs Compression Ratio Trade-off (ε = {error_percent}%)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, which='both')
    
    # Use log scale for both axes in summary plot
    plt.xscale('log')
    plt.yscale('log')
    
    # Format axes properly
    from matplotlib.ticker import ScalarFormatter
    ax = plt.gca()
    
    # Format both axes to avoid scientific notation
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
        axis.get_major_formatter().set_scientific(False)
        axis.get_major_formatter().set_useOffset(False)
    
    plt.tight_layout()
    plt.savefig(f'summary_execution_time_eps_{error_percent}pct.png', 
                dpi=300, bbox_inches='tight')
    print(f"  Saved summary plot: summary_execution_time_eps_{error_percent}pct.png")
    plt.show()
    plt.close()

def print_interpretation_guide():
    """Print a guide on how to interpret the plots."""
    print("\n" + "="*60)
    print("HOW TO INTERPRET THE PLOTS:")
    print("="*60)
    print()
    print("1. X-axis (Compression Ratio):")
    print("   - Higher values = better compression")
    print("   - A ratio of 20 means the compressed data is 20x smaller")
    print()
    print("2. Y-axis (Amortized Execution Time in ms):")
    print("   - Shows time per dataset record")
    print("   - Lower values = faster algorithm")
    print("   - Log scale is used to show wide range of times")
    print("   - Values like 0.0001 ms mean very fast per-record processing")
    print()
    print("3. Ideal Position:")
    print("   - TOP-RIGHT: High compression but slow (good for storage)")
    print("   - BOTTOM-LEFT: Fast but poor compression (good for real-time)")
    print("   - BOTTOM-RIGHT: Fast AND good compression (BEST overall)")
    print()
    print("4. Algorithm Characteristics:")
    print("   - Mix-Piece: Should be rightmost (best compression)")
    print("   - Swing: Should be leftmost and lowest (fastest, worst compression)")
    print("   - Sim-Piece: Should be between Mix-Piece and others")
    print("   - Slide: Traditional optimal algorithm (good balance)")
    print()
    print("5. Effect of Error Threshold (ε):")
    print("   - Larger ε (5%) → Better compression ratios, faster execution")
    print("   - Smaller ε (0.5%) → Lower compression ratios, slower execution")
    print("="*60)

def main():
    """Main execution function"""
    print("TerseTS Execution Time vs Compression Ratio Benchmark")
    print("=" * 60)
    
    # Load UCR datasets
    print("\nLoading UCR datasets...")
    datasets = prepare_ucr_datasets()
    
    if not datasets:
        print("No datasets were loaded. Please check the UCR dataloader setup.")
        return
    
    # Test with two error percentages
    error_percentages = [0.5, 5.0]
    
    # Create individual plots for each dataset
    plot_execution_time_vs_compression(datasets, error_percentages)
    
    # Create summary plots
    for error_percent in error_percentages:
        create_summary_plot(datasets, error_percent)
    
    # Print interpretation guide
    print_interpretation_guide()
    
    print("\n" + "="*60)
    print("Benchmark completed successfully!")
    print("Generated plots:")
    print("  - Individual dataset plots: {dataset_name}_execution_time_vs_compression.png")
    print("  - Summary plots: summary_execution_time_eps_{0.5,5.0}pct.png")

if __name__ == "__main__":
    main()
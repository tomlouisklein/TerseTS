"""
UCR Dataset Compression Ratio vs Error Bound Plots - Real Data
This script creates compression ratio plots for real UCR datasets using TerseTS algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List
import tersets
from tersets import Method
from ucr_dataloader import prepare_ucr_datasets

def benchmark_compression_ratio(data: np.ndarray, method: Method, error_bound: float) -> float:
    """Benchmark compression ratio for a single method and error bound"""
    try:
        # Convert numpy array to list of floats
        values = data.astype(float).tolist()
        
        # Compress the data
        compressed = tersets.compress(values, method, error_bound)
        
        # Calculate compression ratio
        original_size = data.nbytes
        compressed_size = len(compressed)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        return compression_ratio
    except Exception as e:
        print(f"Error with method {method.name} and error bound {error_bound}: {e}")
        return 0

def plot_ucr_compression_ratios(datasets: Dict[str, np.ndarray]):
    """Create compression ratio plots for UCR datasets"""
 
    # Methods to test (using available methods from TerseTS)
    methods_to_test = [
        Method.MixPiece,
        Method.SimPiece,
        Method.PoorMansCompressionMidrange,
        Method.PoorMansCompressionMean,
        Method.SwingFilter,
        Method.SwingFilterDisconnected,
        Method.SlideFilter,
        Method.VisvalingamWhyatt,        
        #Method.PiecewiseConstantHistogram,
        #Method.PiecewiseLinearHistogram,
        #Method.ABCLinearApproximation,
    ]
    
    # Error bounds as percentage of data range
    error_percentages = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    # Create individual plots for each dataset
    for dataset_name, data in datasets.items():
        plt.figure(figsize=(10, 6))
        
        print(f"\nProcessing {dataset_name}...")
        print(f"  Data length: {len(data):,}")
        print(f"  Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
        
        # Calculate data range
        data_range = np.max(data) - np.min(data)
        
        # Test each method
        for method in methods_to_test:
            compression_ratios = []
            
            print(f"    Testing {method.name}...")
            for error_percent in error_percentages:
                error_bound = (error_percent / 100) * data_range
                
                compression_ratio = benchmark_compression_ratio(data, method, error_bound)
                compression_ratios.append(compression_ratio)
                print(f"      {error_percent}%: CR = {compression_ratio:.1f}")
            
            # Plot the results
            plt.plot(error_percentages, compression_ratios, 
                    marker='o', linewidth=2, label=method.name, markersize=6)
        
        plt.xlabel('Error Bound (% of range)', fontsize=12)
        plt.ylabel('Compression Ratio', fontsize=12)
        plt.title(f'{dataset_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(0.4, 5.1)
        
        # Save individual plot
        plt.savefig(f'{dataset_name.lower()}_compression_ratio.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

def print_dataset_summary(datasets: Dict[str, np.ndarray]):
    """Print summary of loaded datasets"""
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    for name, data in datasets.items():
        data_range = np.max(data) - np.min(data)
        print(f"\n{name}:")
        print(f"  Length: {len(data):,} points")
        print(f"  Range: [{np.min(data):.3f}, {np.max(data):.3f}]")
        print(f"  Mean: {np.mean(data):.3f}")
        print(f"  Std: {np.std(data):.3f}")
        print(f"  Data range: {data_range:.3f}")

def main():
    """Main execution function"""
    print("UCR Dataset Compression Ratio Analysis")
    print("=" * 50)
    
    # Check available methods
    print("\nAvailable compression methods:")
    for method in Method:
        print(f"  {method.name}")
    
    # Load UCR datasets
    print("\nLoading UCR datasets...")
    datasets = prepare_ucr_datasets()
    
    if not datasets:
        print("No datasets were loaded. Please check the UCR dataloader setup.")
        return
    
    # Print dataset summary
    print_dataset_summary(datasets)
    
    # Create plots
    print("\nGenerating compression ratio plots...")
    
    # Create individual plots for each dataset
    plot_ucr_compression_ratios(datasets)
    
    print("\nAll plots generated successfully!")
    print("Individual plots saved as: {dataset_name}_compression_ratio.png")
    print("Combined plot saved as: ucr_all_datasets_compression_ratios.png")

if __name__ == "__main__":
    main()
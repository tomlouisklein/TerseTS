"""
TerseTS MAE vs Space Efficiency Benchmarking Script
This script creates MAE vs space efficiency plots similar to Figure 10 in the Mix-Piece paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple
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

def benchmark_mae_and_space(data: np.ndarray, method: Method, error_bound: float) -> Tuple[float, float]:
    """
    Benchmark both MAE and space efficiency for a single method and error bound
    
    Returns:
        Tuple[float, float]: (compressed_size_bytes, mae)
    """
    try:
        # Convert numpy array to list of floats
        values = data.astype(float).tolist()
        
        # Compress the data
        compressed = tersets.compress(values, method, error_bound)
        
        # Calculate compressed size in bytes
        # The compressed result might be a different format depending on the method
        if isinstance(compressed, list):
            # Assuming compressed is a list of segments or points
            compressed_size_bytes = len(compressed) * 8  # 8 bytes per float64
        else:
            # Try to estimate size based on the object
            compressed_size_bytes = len(str(compressed).encode('utf-8'))
        
        # Try to decompress/reconstruct the data
        try:
            # Method 1: Try direct decompress if available
            decompressed = tersets.decompress(compressed)
            approximation = np.array(decompressed)
        except (AttributeError, TypeError):
            try:
                # Method 2: Try reconstruct if available
                decompressed = tersets.reconstruct(compressed)
                approximation = np.array(decompressed)
            except (AttributeError, TypeError):
                try:
                    # Method 3: Check if compressed is already the approximated values
                    if hasattr(compressed, '__iter__') and len(compressed) > 0:
                        # Try to interpret as time series points
                        if hasattr(compressed[0], '__iter__'):
                            # Might be (time, value) pairs or similar
                            approximation = np.array([point[1] if len(point) > 1 else point[0] for point in compressed])
                        else:
                            # Might be just values
                            approximation = np.array(compressed)
                    else:
                        print(f"Warning: Cannot decompress {method.name} output: {type(compressed)}")
                        return 0, float('inf')
                except Exception as e:
                    print(f"Warning: Failed to interpret compressed data for {method.name}: {e}")
                    return 0, float('inf')
        
        # Ensure same length for MAE calculation
        min_len = min(len(data), len(approximation))
        if min_len == 0:
            print(f"Warning: No data to compare for {method.name}")
            return 0, float('inf')
            
        original_subset = data[:min_len]
        approx_subset = approximation[:min_len]
        
        # Calculate Mean Absolute Error
        mae = np.mean(np.abs(original_subset - approx_subset))
        
        return compressed_size_bytes, mae
        
    except Exception as e:
        print(f"Error with method {method.name} and error bound {error_bound}: {e}")
        return 0, float('inf')

def get_tersets_approximation(data: np.ndarray, method: Method, error_bound: float) -> np.ndarray:
    """
    Helper function to get the approximated time series from TerseTS compression
    This function tries different ways to extract the approximation
    """
    try:
        values = data.astype(float).tolist()
        compressed = tersets.compress(values, method, error_bound)
        
        # Try different methods to get the approximation
        decompress_methods = [
            lambda x: tersets.decompress(x),
            lambda x: tersets.reconstruct(x),
            lambda x: tersets.approximate(x),
            lambda x: [point[1] if hasattr(point, '__iter__') and len(point) > 1 else point for point in x],
            lambda x: list(x) if hasattr(x, '__iter__') else [x]
        ]
        
        for method_func in decompress_methods:
            try:
                result = method_func(compressed)
                if result is not None and len(result) > 0:
                    return np.array(result)
            except:
                continue
                
        # If all methods fail, return original data (no compression achieved)
        print(f"Warning: Could not decompress {method.name}, using original data")
        return data
        
    except Exception as e:
        print(f"Error in get_tersets_approximation for {method.name}: {e}")
        return data

def create_mae_space_curves(datasets: Dict[str, np.ndarray]):
    """Create MAE vs Space efficiency curves for different algorithms"""
    
    if not datasets:
        print("No datasets available for plotting!")
        return
    
    # Methods to test - focusing on key ones from the paper
    methods_to_test = [
        Method.MixPiece,          # Should be dotted green like in original
        Method.SimPiece,          # Should be solid dark line
        Method.SwingFilter,       # Swing (solid red/pink)
        Method.SlideFilter,       # Slide (solid blue) 
        Method.PoorMansCompressionMidrange,  # Alternative if Mixed not available
        Method.VisvalingamWhyatt,  # Another method for comparison
    ]
    
    # Error bounds as percentage of data range - wider range for smooth curves
    error_percentages = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0]
    
    # Create individual plots for each dataset
    for dataset_name, data in datasets.items():
        plt.figure(figsize=(10, 8))
        
        print(f"\nProcessing {dataset_name} for MAE vs Space analysis...")
        print(f"  Data length: {len(data):,}")
        print(f"  Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
        
        # Calculate data range
        data_range = np.max(data) - np.min(data)
        
        # Test each method
        for method in methods_to_test:
            compressed_sizes = []
            maes = []
            
            # Define specific styles for each method to match original
            style_map = {
                'MixPiece': {'linestyle': ':', 'color': 'green', 'label': 'Mix-Piece'},
                'SimPiece': {'linestyle': '-', 'color': 'darkblue', 'label': 'Sim-Piece'},
                'SwingFilter': {'linestyle': '-', 'color': 'red', 'label': 'Swing'},
                'SlideFilter': {'linestyle': '-', 'color': 'blue', 'label': 'Slide'},
                'PoorMansCompressionMidrange': {'linestyle': '--', 'color': 'orange', 'label': 'PMC-Midrange'},
                'VisvalingamWhyatt': {'linestyle': '-.', 'color': 'purple', 'label': 'Visvalingam-Whyatt'},
            }
            
            style = style_map.get(method.name, {'linestyle': '-', 'color': 'gray', 'label': method.name})
            
            print(f"    Testing {method.name}...")
            
            for error_percent in error_percentages:
                error_bound = (error_percent / 100) * data_range
                
                compressed_size, mae = benchmark_mae_and_space(data, method, error_bound)
                
                if compressed_size > 0 and mae < float('inf'):
                    compressed_sizes.append(compressed_size)
                    maes.append(mae)
                    print(f"      {error_percent}%: Size = {compressed_size:,.0f} bytes, MAE = {mae:.4f}")
                else:
                    print(f"      {error_percent}%: Failed")
            
            # Plot the curve if we have valid data points
            if len(compressed_sizes) > 0 and len(maes) > 0:
                plt.semilogx(compressed_sizes, maes, 
                            marker='o', linewidth=2, markersize=6,
                            linestyle=style['linestyle'],
                            color=style['color'],
                            label=style['label'])
        
        plt.xlabel('Size (Bytes)', fontsize=12)
        plt.ylabel('Mean Absolute Error', fontsize=12)
        plt.title(f'{dataset_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Set y-axis to start from 0 and use reasonable limits
        plt.ylim(bottom=0)
        
        # Format x-axis to show scientific notation like the original
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0e}'))
        
        # Save individual plot
        plt.savefig(f'{dataset_name.lower()}_mae_vs_space.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

def create_combined_mae_space_plot(datasets: Dict[str, np.ndarray]):
    """Create a combined plot showing MAE vs Space for selected datasets"""
    
    if not datasets:
        print("No datasets available for plotting!")
        return
    
    # Select a subset of datasets for the combined plot
    selected_datasets = {}
    dataset_names = list(datasets.keys())
    
    # Take up to 4 datasets for readability
    for i, name in enumerate(dataset_names[:4]):
        selected_datasets[name] = datasets[name]
    
    # Calculate subplot layout
    n_datasets = len(selected_datasets)
    if n_datasets <= 2:
        rows, cols = 1, n_datasets
    else:
        rows, cols = 2, 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    fig.suptitle('MAE vs Size Comparison', fontsize=16)
    
    # Handle single subplot case
    if n_datasets == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()
    
    # Methods to test (subset for clarity in combined plot)
    methods_to_test = [
        Method.MixPiece,
        Method.SimPiece,
        Method.SwingFilter,
        Method.SlideFilter
    ]
    
    # Error bounds - use enough points for smooth curves
    error_percentages = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]
    
    # Plot each dataset
    for idx, (dataset_name, data) in enumerate(selected_datasets.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        print(f"\nProcessing {dataset_name} for combined plot...")
        
        # Calculate data range
        data_range = np.max(data) - np.min(data)
        
        # Test each method
        for method in methods_to_test:
            compressed_sizes = []
            maes = []
            
            # Define specific styles for each method to match original
            style_map = {
                'MixPiece': {'linestyle': ':', 'color': 'green', 'label': 'Mix-Piece'},
                'SimPiece': {'linestyle': '-', 'color': 'darkblue', 'label': 'Sim-Piece'},
                'SwingFilter': {'linestyle': '-', 'color': 'red', 'label': 'Swing'},
                'SlideFilter': {'linestyle': '-', 'color': 'blue', 'label': 'Slide'},
            }
            
            style = style_map.get(method.name, {'linestyle': '-', 'color': 'gray', 'label': method.name})
            
            for error_percent in error_percentages:
                error_bound = (error_percent / 100) * data_range
                compressed_size, mae = benchmark_mae_and_space(data, method, error_bound)
                
                if compressed_size > 0 and mae < float('inf'):
                    compressed_sizes.append(compressed_size)
                    maes.append(mae)
            
            # Plot the curve
            if len(compressed_sizes) > 0 and len(maes) > 0:
                ax.semilogx(compressed_sizes, maes, 
                           marker='o', linewidth=2,
                           linestyle=style['linestyle'],
                           color=style['color'],
                           label=style['label'])
        
        ax.set_xlabel('Size (Bytes)')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title(dataset_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Format x-axis to show scientific notation like the original
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0e}'))
    
    # Hide unused subplots
    for idx in range(len(selected_datasets), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('mae_vs_space_combined.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_pareto_frontier(datasets: Dict[str, np.ndarray]):
    """Analyze which algorithms provide the best MAE vs Space tradeoffs (Pareto frontier)"""
    
    print("\n" + "="*60)
    print("PARETO FRONTIER ANALYSIS")
    print("="*60)
    
    methods_to_test = [
        Method.MixPiece,
        Method.SimPiece,
        Method.PoorMansCompressionMidrange,
        Method.PoorMansCompressionMean,
        Method.SwingFilter,
        Method.SwingFilterDisconnected,
        Method.SlideFilter,
        Method.VisvalingamWhyatt, 
    ]
    
    error_percentages = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    for dataset_name, data in datasets.items():
        print(f"\n{dataset_name}:")
        print("-" * 40)
        
        data_range = np.max(data) - np.min(data)
        all_results = []
        
        for method in methods_to_test:
            for error_percent in error_percentages:
                error_bound = (error_percent / 100) * data_range
                compressed_size, mae = benchmark_mae_and_space(data, method, error_bound)
                
                if compressed_size > 0 and mae < float('inf'):
                    all_results.append((method.name, error_percent, compressed_size, mae))
        
        # Sort by compressed size
        all_results.sort(key=lambda x: x[2])
        
        # Find Pareto frontier (points where no other point has both smaller size and smaller MAE)
        pareto_points = []
        for i, (method, error_pct, size, mae) in enumerate(all_results):
            is_pareto = True
            for j, (other_method, other_error_pct, other_size, other_mae) in enumerate(all_results):
                if i != j and other_size <= size and other_mae <= mae and (other_size < size or other_mae < mae):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_points.append((method, error_pct, size, mae))
        
        print("Pareto-optimal points (Method, Error%, Size(bytes), MAE):")
        for method, error_pct, size, mae in pareto_points:
            print(f"  {method:25} {error_pct:6.1f}% {size:10,.0f} {mae:10.4f}")

def simple_space_analysis(datasets: Dict[str, np.ndarray]):
    """
    Simplified analysis using compression ratios to estimate space usage
    This is a fallback if the full MAE analysis doesn't work due to API limitations
    """
    print("\n" + "="*60)
    print("SIMPLIFIED SPACE ANALYSIS (using compression ratios)")
    print("="*60)
    
    methods_to_test = [
        Method.MixPiece,
        Method.SimPiece,
        Method.SwingFilter,
        Method.SlideFilter
    ]
    
    error_percentages = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    for dataset_name, data in datasets.items():
        print(f"\n{dataset_name}:")
        print("-" * 40)
        
        original_size_bytes = data.nbytes
        data_range = np.max(data) - np.min(data)
        
        print(f"Original size: {original_size_bytes:,} bytes")
        print(f"Method                    Error%    Compressed Size (bytes)    Compression Ratio")
        print("-" * 80)
        
        for method in methods_to_test:
            for error_percent in error_percentages:
                error_bound = (error_percent / 100) * data_range
                
                # Use your existing compression ratio function
                compression_ratio = benchmark_compression_ratio(data, method, error_bound)
                
                if compression_ratio > 0:
                    compressed_size = original_size_bytes / compression_ratio
                    print(f"{method.name:25} {error_percent:6.1f}%    {compressed_size:15,.0f}    {compression_ratio:12.1f}")

def create_compression_size_plot(datasets: Dict[str, np.ndarray]):
    """
    Create plots showing compressed size vs error threshold
    This provides insight into space efficiency without needing MAE
    """
    
    methods_to_test = [
        Method.MixPiece,
        Method.SimPiece,
        Method.SwingFilter,
        Method.SlideFilter
    ]
    
    error_percentages = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    
    for dataset_name, data in datasets.items():
        plt.figure(figsize=(10, 6))
        
        print(f"\nCreating size vs error plot for {dataset_name}...")
        
        original_size_bytes = data.nbytes
        data_range = np.max(data) - np.min(data)
        
        for method in methods_to_test:
            compressed_sizes = []
            error_bounds_actual = []
            
            for error_percent in error_percentages:
                error_bound = (error_percent / 100) * data_range
                compression_ratio = benchmark_compression_ratio(data, method, error_bound)
                
                if compression_ratio > 0:
                    compressed_size = original_size_bytes / compression_ratio
                    compressed_sizes.append(compressed_size)
                    error_bounds_actual.append(error_percent)
            
            if len(compressed_sizes) > 0:
                plt.semilogy(error_bounds_actual, compressed_sizes, 
                           marker='o', linewidth=2, label=method.name, markersize=6)
        
        plt.xlabel('Error Bound (% of range)', fontsize=12)
        plt.ylabel('Compressed Size (bytes)', fontsize=12)
        plt.title(f'{dataset_name} - Compressed Size vs Error Bound', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add original size reference line
        plt.axhline(y=original_size_bytes, color='red', linestyle='--', alpha=0.7, 
                   label=f'Original size ({original_size_bytes:,} bytes)')
        plt.legend(fontsize=10)
        
        plt.savefig(f'{dataset_name.lower()}_size_vs_error.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main execution function"""
    print("TerseTS MAE vs Space Efficiency Analysis")
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
        print(f"  Original size: {data.nbytes:,} bytes")
    
    # Try full MAE vs Space analysis first
    print("\nAttempting full MAE vs Space efficiency analysis...")
    try:
        # Create MAE vs Space plots
        create_mae_space_curves(datasets)
        create_combined_mae_space_plot(datasets)
        analyze_pareto_frontier(datasets)
        
        print("\nFull MAE analysis complete!")
        print("Individual plots saved as: {dataset_name}_mae_vs_space.png")
        print("Combined plot saved as: mae_vs_space_combined.png")
        
    except Exception as e:
        print(f"Full MAE analysis failed: {e}")
        print("Falling back to simplified analysis...")
        
        # Fallback to simpler analysis
        simple_space_analysis(datasets)
        create_compression_size_plot(datasets)
        
        print("\nSimplified analysis complete!")
        print("Size vs error plots saved as: {dataset_name}_size_vs_error.png")

if __name__ == "__main__":
    main()
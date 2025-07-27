"""
Anomaly Detection Benchmarking for Time Series Compression
Compares anomaly detection performance across different compression methods
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import tersets
from tersets import Method
from anomaly_detection_analytics.ucr_dataloader import prepare_ucr_datasets
from typing import Dict, List, Tuple
import time

# Configuration
METHODS_TO_TEST = [Method.MixPiece, Method.SimPiece, Method.SlideFilter]
ERROR_BOUNDS = np.logspace(-3, -1, 10)  # 0.1% to 10% of range
CONTAMINATION = 0.1  # 10% expected anomalies
RANDOM_STATE = 42
TOP_K_PERCENT = 0.1  # Top 10% for overlap metric

def compress_decompress(data: np.ndarray, method: Method, error_bound: float) -> Tuple[np.ndarray, float]:
    """Compress and decompress data, returning decompressed values and compression ratio"""
    try:
        values = data.astype(float).tolist()
        
        # Compress
        compressed = tersets.compress(values, method, error_bound)
        
        # Calculate compression ratio
        original_size = data.nbytes
        compressed_size = len(compressed)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        # Decompress (method is encoded in the last byte)
        decompressed = tersets.decompress(compressed)
        
        return np.array(decompressed), compression_ratio
    except Exception as e:
        print(f"Error with {method.name}, error_bound={error_bound}: {e}")
        return data.copy(), 1.0

def detect_anomalies(data: np.ndarray, contamination: float = CONTAMINATION) -> np.ndarray:
    """Run Isolation Forest and return anomaly scores"""
    # Reshape for sklearn
    X = data.reshape(-1, 1)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Detect anomalies
    iso_forest = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    iso_forest.fit(X_scaled)
    
    # Get anomaly scores (lower = more anomalous)
    scores = iso_forest.score_samples(X_scaled)
    return scores

def calculate_spearman_correlation(scores1: np.ndarray, scores2: np.ndarray) -> float:
    """Calculate Spearman correlation between two sets of anomaly scores"""
    correlation, _ = spearmanr(scores1, scores2)
    return correlation

def calculate_top_k_overlap(scores1: np.ndarray, scores2: np.ndarray, k_percent: float = TOP_K_PERCENT) -> float:
    """Calculate overlap in top-k anomalies"""
    k = int(len(scores1) * k_percent)
    
    # Get indices of top-k anomalies (lowest scores)
    top_k_1 = set(np.argsort(scores1)[:k])
    top_k_2 = set(np.argsort(scores2)[:k])
    
    # Calculate overlap
    overlap = len(top_k_1.intersection(top_k_2)) / k
    return overlap

def benchmark_dataset(data: np.ndarray, dataset_name: str) -> Dict:
    """Benchmark anomaly detection for a single dataset"""
    print(f"\nBenchmarking {dataset_name}...")
    
    # Calculate data range for error bounds
    data_range = np.max(data) - np.min(data)
    
    # Get ground truth anomaly scores
    ground_truth_scores = detect_anomalies(data)
    
    results = {method: {'compression_ratios': [], 'spearman_correlations': [], 'top_k_overlaps': []} 
               for method in METHODS_TO_TEST}
    
    for error_percent in ERROR_BOUNDS * 100:  # Convert to percentage
        error_bound = (error_percent / 100) * data_range
        
        for method in METHODS_TO_TEST:
            # Compress and decompress
            decompressed, compression_ratio = compress_decompress(data, method, error_bound)
            
            # Detect anomalies on decompressed data
            decompressed_scores = detect_anomalies(decompressed)
            
            # Calculate metrics
            spearman_corr = calculate_spearman_correlation(ground_truth_scores, decompressed_scores)
            top_k_overlap = calculate_top_k_overlap(ground_truth_scores, decompressed_scores)
            
            # Store results
            results[method]['compression_ratios'].append(compression_ratio)
            results[method]['spearman_correlations'].append(spearman_corr)
            results[method]['top_k_overlaps'].append(top_k_overlap)
            
            print(f"  {method.name} - ε={error_percent:.1f}%: CR={compression_ratio:.1f}, "
                  f"Spearman={spearman_corr:.3f}, Top-K={top_k_overlap:.3f}")
    
    return results

def plot_individual_results(all_results: Dict[str, Dict]):
    """Plot performance vs compression ratio for each dataset individually"""
    colors = {'MixPiece': '#1f77b4', 'SimPiece': '#ff7f0e', 'SlideFilter': '#2ca02c'}
    
    for dataset_name, results in all_results.items():
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        for method in METHODS_TO_TEST:
            compression_ratios = results[method]['compression_ratios']
            spearman_corrs = results[method]['spearman_correlations']
            top_k_overlaps = results[method]['top_k_overlaps']
            
            # Sort by compression ratio for smooth lines
            sorted_indices = np.argsort(compression_ratios)
            cr_sorted = np.array(compression_ratios)[sorted_indices]
            spearman_sorted = np.array(spearman_corrs)[sorted_indices]
            topk_sorted = np.array(top_k_overlaps)[sorted_indices]
            
            # Plot Spearman correlation
            ax1.plot(cr_sorted, spearman_sorted, 
                    marker='o', label=method.name,
                    color=colors[method.name], linewidth=2.5, markersize=7)
            
            # Plot Top-K overlap
            ax2.plot(cr_sorted, topk_sorted, 
                    marker='o', label=method.name,
                    color=colors[method.name], linewidth=2.5, markersize=7)
        
        # Configure Spearman subplot
        ax1.set_xlabel('Compression Ratio', fontsize=12)
        ax1.set_ylabel('Spearman Correlation', fontsize=12)
        ax1.set_title(f'{dataset_name} - Score Correlation', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)
        
        # Configure Top-K subplot
        ax2.set_xlabel('Compression Ratio', fontsize=12)
        ax2.set_ylabel('Top-K Anomaly Overlap', fontsize=12)
        ax2.set_title(f'{dataset_name} - Top-10% Overlap', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        filename = f'anomaly_{dataset_name.lower()}_performance.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved plot: {filename}")

def plot_aggregated_results(all_results: Dict[str, Dict]):
    """Plot aggregated performance across all datasets"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {'MixPiece': '#1f77b4', 'SimPiece': '#ff7f0e', 'SlideFilter': '#2ca02c'}
    
    for method in METHODS_TO_TEST:
        # Aggregate data across datasets
        all_cr = []
        all_spearman = []
        all_topk = []
        
        for dataset_results in all_results.values():
            all_cr.extend(dataset_results[method]['compression_ratios'])
            all_spearman.extend(dataset_results[method]['spearman_correlations'])
            all_topk.extend(dataset_results[method]['top_k_overlaps'])
        
        # Create bins for compression ratio
        cr_bins = np.logspace(0, 2, 20)  # 1 to 100
        spearman_means = []
        topk_means = []
        cr_centers = []
        
        for i in range(len(cr_bins) - 1):
            mask = (np.array(all_cr) >= cr_bins[i]) & (np.array(all_cr) < cr_bins[i+1])
            if np.any(mask):
                spearman_means.append(np.mean(np.array(all_spearman)[mask]))
                topk_means.append(np.mean(np.array(all_topk)[mask]))
                cr_centers.append(np.sqrt(cr_bins[i] * cr_bins[i+1]))  # Geometric mean
        
        # Plot Spearman correlation
        ax1.plot(cr_centers, spearman_means, 
                marker='o', label=method.name, color=colors[method.name], 
                linewidth=3, markersize=8)
        
        # Plot Top-K overlap
        ax2.plot(cr_centers, topk_means, 
                marker='o', label=method.name, color=colors[method.name], 
                linewidth=3, markersize=8)
    
    # Configure Spearman plot
    ax1.set_xlabel('Compression Ratio', fontsize=12)
    ax1.set_ylabel('Spearman Correlation', fontsize=12)
    ax1.set_title('Score Correlation vs Compression Ratio', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.set_xlim(1, 100)
    ax1.set_ylim(0, 1.05)
    
    # Configure Top-K plot
    ax2.set_xlabel('Compression Ratio', fontsize=12)
    ax2.set_ylabel('Top-K Anomaly Overlap', fontsize=12)
    ax2.set_title('Top-K Overlap vs Compression Ratio', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    ax2.set_xlim(1, 100)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig('anomaly_detection_aggregated.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved aggregated plot: anomaly_detection_aggregated.png")

def main():
    """Main execution function"""
    print("Anomaly Detection Benchmarking")
    print("=" * 50)
    
    # Load datasets
    print("\nLoading UCR datasets...")
    datasets = prepare_ucr_datasets()
    
    if not datasets:
        print("No datasets loaded. Please check UCR dataloader setup.")
        return
    
    # Benchmark each dataset
    all_results = {}
    for dataset_name, data in datasets.items():
        results = benchmark_dataset(data, dataset_name)
        all_results[dataset_name] = results
    
    # Generate plots
    print("\nGenerating plots...")
    plot_individual_results(all_results)
    plot_aggregated_results(all_results)
    
    print("\nBenchmarking complete!")

if __name__ == "__main__":
    main()
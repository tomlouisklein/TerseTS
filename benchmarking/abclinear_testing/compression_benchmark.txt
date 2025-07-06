"""
UCR Dataset Compression Benchmarking
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import Dict, Optional
import tersets
from tersets import Method

class UCRLoader:
    def __init__(self, ucr_path: str = "/home/tom/uni/large-scale_data_engineering/UCRArchive_2018"):
        self.ucr_path = ucr_path
    
    def load_dataset(self, dataset_name: str) -> Optional[np.ndarray]:
        dataset_dir = os.path.join(self.ucr_path, dataset_name)
        train_file = os.path.join(dataset_dir, f"{dataset_name}_TRAIN.tsv")
        test_file = os.path.join(dataset_dir, f"{dataset_name}_TEST.tsv")
        
        try:
            data_parts = []
            
            if os.path.exists(train_file):
                train_data = pd.read_csv(train_file, sep='\t', header=None)
                if train_data.shape[1] > 1:
                    train_series = train_data.iloc[:, 1:].values.flatten()
                    train_series = train_series[~np.isnan(train_series)]
                    data_parts.append(train_series)
            
            if os.path.exists(test_file):
                test_data = pd.read_csv(test_file, sep='\t', header=None)
                if test_data.shape[1] > 1:
                    test_series = test_data.iloc[:, 1:].values.flatten()
                    test_series = test_series[~np.isnan(test_series)]
                    data_parts.append(test_series)
            
            if data_parts:
                combined_data = np.concatenate(data_parts)
                print(f"Loaded {dataset_name}: {len(combined_data):,} points")
                return combined_data
                
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
        return None
    
    def load_benchmark_datasets(self) -> Dict[str, np.ndarray]:
        # NOTE: Change the arrangements of datasets if needed.
        dataset_names = ['FaceFour', 'Lightning7', 'MoteStrain', 'Wafer', 'CricketX']
        datasets = {}
        
        for name in dataset_names:
            data = self.load_dataset(name)
            if data is not None:
                datasets[name] = data
        
        return datasets

def benchmark_compression(data: np.ndarray, method: Method, error_bound: float) -> float:
    try:
        values = data.astype(float).tolist()
        compressed = tersets.compress(values, method, error_bound)
        
        original_size = data.nbytes
        compressed_size = len(compressed)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        return compression_ratio
    except Exception:
        return 0

def create_compression_plots(datasets: Dict[str, np.ndarray]):
    methods_to_test = [
        Method.MixPiece,
        Method.SimPiece,
        Method.PoorMansCompressionMidrange,
        Method.PoorMansCompressionMean,
        Method.SwingFilter,
        Method.SwingFilterDisconnected,
        Method.SlideFilter,
        Method.VisvalingamWhyatt,
        Method.ABCLinearApproximation,
    ]
    
    error_percentages = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    # Plot 1: Individual dataset plots
    for dataset_name, data in datasets.items():
        print(f"Processing {dataset_name}...")
        plt.figure(figsize=(10, 6))
        data_range = np.max(data) - np.min(data)
        
        for method in methods_to_test:
            print(f"  Testing {method.name}...")
            compression_ratios = []
            
            for error_percent in error_percentages:
                error_bound = (error_percent / 100) * data_range
                compression_ratio = benchmark_compression(data, method, error_bound)
                compression_ratios.append(compression_ratio)
            
            plt.plot(error_percentages, compression_ratios, 
                    marker='o', linewidth=2, label=method.name, markersize=4)
        
        plt.xlabel('Error Bound (% of range)')
        plt.ylabel('Compression Ratio')
        plt.title(f'{dataset_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0.4, 5.1)
        plt.savefig(f'{dataset_name.lower()}_compression.png', dpi=300, bbox_inches='tight')
        plt.show()
    
def main():
    loader = UCRLoader()
    datasets = loader.load_benchmark_datasets()
    
    if not datasets:
        print("No datasets loaded")
        return
    
    print(f"Loaded {len(datasets)} datasets")
    for name, data in datasets.items():
        print(f"  {name}: {len(data):,} points")
    
    create_compression_plots(datasets)

if __name__ == "__main__":
    main()
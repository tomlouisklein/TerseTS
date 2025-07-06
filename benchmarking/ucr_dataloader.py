import numpy as np
import pandas as pd
import os
from typing import Dict, Optional

class UCRDatasetLoader:
    """Load UCR Time Series datasets from local UCRArchive_2018 directory"""
    
    def __init__(self, ucr_path: str = "/home/tom/uni/large-scale_data_engineering/UCRArchive_2018"):
        self.ucr_path = ucr_path
        if not os.path.exists(ucr_path):
            raise FileNotFoundError(f"UCR Archive path not found: {ucr_path}")
    
    def load_ucr_dataset(self, dataset_name: str, combine_train_test: bool = True) -> Optional[np.ndarray]:
        """
        Load UCR dataset from local files
        
        Args:
            dataset_name: Name of the UCR dataset directory
            combine_train_test: If True, combine training and test sets into one series
            
        Returns:
            numpy array containing the time series data
        """
        dataset_dir = os.path.join(self.ucr_path, dataset_name)
        
        if not os.path.exists(dataset_dir):
            print(f"Dataset directory not found: {dataset_dir}")
            return None
        
        train_file = os.path.join(dataset_dir, f"{dataset_name}_TRAIN.tsv")
        test_file = os.path.join(dataset_dir, f"{dataset_name}_TEST.tsv")
        
        try:
            data_parts = []
            
            # Load training data
            if os.path.exists(train_file):
                print(f"Loading {train_file}")
                train_data = pd.read_csv(train_file, sep='\t', header=None)
                
                # Remove class labels (first column) and flatten all series
                if train_data.shape[1] > 1:
                    train_series = train_data.iloc[:, 1:].values.flatten()
                    # Remove NaN values that might exist
                    train_series = train_series[~np.isnan(train_series)]
                    data_parts.append(train_series)
                    print(f"  Train data shape: {train_data.shape}")
            
            # Load test data if combining
            if combine_train_test and os.path.exists(test_file):
                print(f"Loading {test_file}")
                test_data = pd.read_csv(test_file, sep='\t', header=None)
                
                if test_data.shape[1] > 1:
                    test_series = test_data.iloc[:, 1:].values.flatten()
                    test_series = test_series[~np.isnan(test_series)]
                    data_parts.append(test_series)
                    print(f"  Test data shape: {test_data.shape}")
            
            if data_parts:
                combined_data = np.concatenate(data_parts)
                print(f"  Combined dataset length: {len(combined_data):,}")
                return combined_data
            else:
                print(f"No data found for {dataset_name}")
                return None
                
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            return None
    
    def load_compression_benchmark_datasets(self, include_all_cricket: bool = False) -> Dict[str, np.ndarray]:
        """
        Load the specific UCR datasets commonly used in compression benchmarking
        
        Args:
            include_all_cricket: If True, load CricketX, CricketY, CricketZ separately
        """
        
        if include_all_cricket:
            # Load all Cricket variants separately
            dataset_mapping = {
                'CricketX': 'CricketX',
                'CricketY': 'CricketY', 
                'CricketZ': 'CricketZ',
                'FaceFour': 'FaceFour',
                'Lightning': 'Lightning7',
                'MoteStrain': 'MoteStrain',
                'Wafer': 'Wafer'
            }
            alternatives = {
                'Lightning': ['Lightning2']
            }
        else:
            # Use single Cricket representative
            dataset_mapping = {
                'Cricket': 'CricketX',        # Using CricketX as representative
                'FaceFour': 'FaceFour',
                'Lightning': 'Lightning7',    # Using Lightning7 as it's more commonly used
                'MoteStrain': 'MoteStrain',
                'Wafer': 'Wafer'
            }
            # Alternative mappings to try if primary choice fails
            alternatives = {
                'Cricket': ['CricketY', 'CricketZ'],
                'Lightning': ['Lightning2']
            }
        
        datasets = {}
        
        print("Loading UCR datasets for compression benchmarking...")
        print("=" * 60)
        
        for paper_name, ucr_name in dataset_mapping.items():
            print(f"\nLoading {paper_name} (UCR: {ucr_name})...")
            data = self.load_ucr_dataset(ucr_name)
            
            # Try alternatives if primary choice failed
            if data is None and paper_name in alternatives:
                for alt_name in alternatives[paper_name]:
                    print(f"  Trying alternative: {alt_name}...")
                    data = self.load_ucr_dataset(alt_name)
                    if data is not None:
                        print(f"  Using {alt_name} instead of {ucr_name}")
                        break
            
            if data is not None:
                datasets[paper_name] = data
                stats = self.get_dataset_stats(data)
                print(f"  ✓ Loaded successfully")
                print(f"  Length: {stats['length']:,}")
                print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
                print(f"  Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
            else:
                print(f"  ✗ Failed to load {paper_name}")
        
        return datasets
    
    def list_available_datasets(self) -> list:
        """List all available datasets in the UCR archive"""
        datasets = []
        for item in os.listdir(self.ucr_path):
            item_path = os.path.join(self.ucr_path, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Check if it has the expected train/test files
                train_file = os.path.join(item_path, f"{item}_TRAIN.tsv")
                test_file = os.path.join(item_path, f"{item}_TEST.tsv")
                if os.path.exists(train_file) or os.path.exists(test_file):
                    datasets.append(item)
        return sorted(datasets)
    
    @staticmethod
    def get_dataset_stats(data: np.ndarray) -> Dict[str, float]:
        """Calculate statistics about the dataset"""
        return {
            'length': len(data),
            'min': np.min(data),
            'max': np.max(data),
            'range': np.max(data) - np.min(data),
            'mean': np.mean(data),
            'std': np.std(data),
            'median': np.median(data)
        }

def prepare_ucr_datasets(ucr_path: str = "/home/tom/uni/large-scale_data_engineering/UCRArchive_2018", include_all_cricket: bool = False) -> Dict[str, np.ndarray]:
    """
    Prepare UCR datasets for compression benchmarking
    
    Args:
        ucr_path: Path to the UCRArchive_2018 directory
        
    Returns:
        Dictionary mapping dataset names to numpy arrays
    """
    try:
        loader = UCRDatasetLoader(ucr_path)
        
        print("UCR Time Series Datasets for Compression Benchmarking")
        print("=" * 60)
        
        # Load the datasets
        datasets = loader.load_compression_benchmark_datasets(include_all_cricket)
        
        if not datasets:
            print("\nNo datasets were successfully loaded.")
            print("Available datasets in your UCR archive:")
            available = loader.list_available_datasets()
            for i, dataset in enumerate(available[:10]):  # Show first 10
                print(f"  {dataset}")
            if len(available) > 10:
                print(f"  ... and {len(available) - 10} more")
            return {}
        
        print(f"\nSuccessfully loaded {len(datasets)} datasets:")
        for name in datasets.keys():
            print(f"  ✓ {name}")
        
        return datasets
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return {}

if __name__ == "__main__":
    # Load datasets for compression benchmarking
    datasets = prepare_ucr_datasets()
    
    # Example: Access individual datasets
    if 'Cricket' in datasets:
        cricket_data = datasets['Cricket']
        print(f"\nCricket dataset: {len(cricket_data):,} points")
        print(f"Range: {np.min(cricket_data):.3f} to {np.max(cricket_data):.3f}")
    
    # Show what's available
    if datasets:
        print(f"\nDatasets ready for compression benchmarking:")
        for name, data in datasets.items():
            print(f"  {name}: {len(data):,} data points")
    
    # Optional: List all available datasets
    loader = UCRDatasetLoader()
    print(f"\nAll available datasets in UCR archive: {len(loader.list_available_datasets())}")
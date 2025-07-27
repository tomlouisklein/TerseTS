"""
UCR Dataset Loader with Label Analysis for Contamination Rate Estimation
Loads UCR datasets with labels and provides contamination rate recommendations
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, Tuple, Optional
from collections import Counter
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score


class UCRContaminationAnalyzer:
    """Load UCR datasets and analyze class distributions for anomaly detection"""
    
    def __init__(self, ucr_path: str = "/home/tom/uni/large-scale_data_engineering/UCRArchive_2018"):
        self.ucr_path = ucr_path
        if not os.path.exists(ucr_path):
            raise FileNotFoundError(f"UCR Archive path not found: {ucr_path}")
        
        # Binary datasets well-suited for anomaly detection
        self.binary_anomaly_datasets = [
            'Earthquakes',          # 7.91% contamination
            'Lightning2',           # 16.39% contamination
            'Chinatown',           # 5.51% contamination
            'SonyAIBORobotSurface1',  # 16.81% contamination
            'Wafer',               # 10.45% contamination
            'ECG200',              # 33% contamination
            'ItalyPowerDemand',    # 49.54% contamination (nearly balanced)
            'ToeSegmentation2'     # 27.69% contamination
        ]
        
        # Domain knowledge for specific datasets (for multi-class)
        self.anomaly_class_hints = {
            'Lightning7': ['O'],  # Off-record events are anomalous
            'Lightning2': ['negative'],  # If binary, negative class might be anomalous
            'Wafer': ['abnormal', '-1', '2'],  # Abnormal wafers
            'ECG200': ['abnormal', '-1', '2'],  # Abnormal heartbeats
            'ECG5000': ['abnormal', '-1', '2', '3', '4', '5'],  # Various arrhythmias
            'MoteStrain': ['2'],  # Often the minority class
            'FaceFour': None,  # No clear anomaly interpretation
            'Cricket': None,  # Classification task, no natural anomalies
        }
    
    def load_dataset_with_labels(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Load UCR dataset with labels and return data, labels, and recommended contamination rate
        
        Returns:
            data: Combined train+test time series data
            labels: Combined train+test labels
            contamination: Recommended contamination rate based on class distribution
        """
        dataset_dir = os.path.join(self.ucr_path, dataset_name)
        
        if not os.path.exists(dataset_dir):
            print(f"Dataset directory not found: {dataset_dir}")
            return None, None, 0.1  # Default fallback
        
        train_file = os.path.join(dataset_dir, f"{dataset_name}_TRAIN.tsv")
        test_file = os.path.join(dataset_dir, f"{dataset_name}_TEST.tsv")
        
        try:
            # Load train and test data
            train_data = pd.read_csv(train_file, sep='\t', header=None)
            test_data = pd.read_csv(test_file, sep='\t', header=None)
            
            # Extract labels (first column) and time series data
            train_labels = train_data.iloc[:, 0].values
            train_series = train_data.iloc[:, 1:].values
            
            test_labels = test_data.iloc[:, 0].values
            test_series = test_data.iloc[:, 1:].values
            
            # Combine train and test
            all_labels = np.concatenate([train_labels, test_labels])
            all_series = np.concatenate([train_series, test_series])
            
            # Flatten time series for anomaly detection
            all_data = all_series.flatten()
            
            # Calculate contamination rate
            contamination = self._calculate_contamination(dataset_name, all_labels, 
                                                         train_labels, test_labels)
            
            print(f"\n{dataset_name} Dataset Analysis:")
            print(f"  Total samples: {len(all_labels):,}")
            print(f"  Time series length: {all_series.shape[1]}")
            print(f"  Unique classes: {np.unique(all_labels)}")
            print(f"  Class distribution: {Counter(all_labels)}")
            print(f"  Recommended contamination: {contamination:.3f}")
            
            return all_data, all_labels, contamination
            
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            return None, None, 0.1
    
    def _calculate_contamination(self, dataset_name: str, all_labels: np.ndarray,
                               train_labels: np.ndarray, test_labels: np.ndarray) -> float:
        """Calculate appropriate contamination rate based on class distribution"""
        
        unique_classes = np.unique(all_labels)
        class_counts = Counter(all_labels)
        
        # For binary classification - simple minority class proportion
        if len(unique_classes) == 2:
            total_count = len(all_labels)
            minority_class = min(class_counts, key=class_counts.get)
            minority_count = class_counts[minority_class]
            contamination = minority_count / total_count
            
            print(f"  Binary classification detected")
            print(f"  Classes: {unique_classes}")
            print(f"  Class distribution: {dict(class_counts)}")
            print(f"  Minority class: {minority_class} ({minority_count}/{total_count} samples)")
            
            # Verify consistency between train and test
            train_minority_rate = np.sum(train_labels == minority_class) / len(train_labels)
            test_minority_rate = np.sum(test_labels == minority_class) / len(test_labels)
            print(f"  Train minority rate: {train_minority_rate:.3f}")
            print(f"  Test minority rate: {test_minority_rate:.3f}")
            
            return contamination
        
        # For multi-class with domain knowledge
        elif dataset_name in self.anomaly_class_hints and self.anomaly_class_hints[dataset_name]:
            anomaly_classes = self.anomaly_class_hints[dataset_name]
            anomaly_count = sum(class_counts[c] for c in anomaly_classes if c in class_counts)
            contamination = anomaly_count / len(all_labels) if anomaly_count > 0 else 0.05
            
            print(f"  Using domain knowledge for anomaly classes: {anomaly_classes}")
            print(f"  Anomaly samples: {anomaly_count}")
            
            return contamination
        
        # For other multi-class: use smallest class(es)
        else:
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
            
            # Option 1: Use only the smallest class
            smallest_count = sorted_classes[0][1]
            contamination = smallest_count / len(all_labels)
            
            # Option 2: Use bottom 20% of classes as anomalies
            # cumulative_count = 0
            # for cls, count in sorted_classes:
            #     cumulative_count += count
            #     if cumulative_count / len(all_labels) > 0.2:
            #         break
            # contamination = cumulative_count / len(all_labels)
            
            print(f"  Multi-class: using smallest class as anomaly")
            print(f"  Smallest class: {sorted_classes[0][0]} ({smallest_count} samples)")
            
            return max(contamination, 0.01)  # At least 1% contamination
    
    def grid_search_contamination(self, data: np.ndarray, 
                                 contamination_range: list = None) -> float:
        """
        Grid search for optimal contamination parameter using unsupervised metrics
        
        Args:
            data: Time series data
            contamination_range: List of contamination values to test
            
        Returns:
            Best contamination value based on silhouette score
        """
        if contamination_range is None:
            contamination_range = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
        
        # Prepare data
        X = data.reshape(-1, 1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        best_score = -1
        best_contamination = 0.1
        
        print("\nGrid Search Results:")
        print("Contamination | Silhouette Score | CH Score")
        print("-" * 45)
        
        for contamination in contamination_range:
            try:
                # Fit Isolation Forest
                iso_forest = IsolationForest(
                    contamination=contamination,
                    random_state=42,
                    n_estimators=100
                )
                labels = iso_forest.fit_predict(X_scaled)
                
                # Skip if all points are in one class
                if len(np.unique(labels)) < 2:
                    continue
                
                # Calculate metrics
                silhouette = silhouette_score(X_scaled, labels)
                ch_score = calinski_harabasz_score(X_scaled, labels)
                
                print(f"{contamination:12.3f} | {silhouette:16.3f} | {ch_score:8.1f}")
                
                if silhouette > best_score:
                    best_score = silhouette
                    best_contamination = contamination
                    
            except Exception as e:
                print(f"{contamination:12.3f} | Error: {str(e)}")
        
        print(f"\nBest contamination: {best_contamination} (silhouette={best_score:.3f})")
        return best_contamination
    
    def analyze_all_datasets(self, dataset_names: list) -> Dict[str, float]:
        """Analyze contamination rates for multiple datasets"""
        contamination_rates = {}
        
        for dataset_name in dataset_names:
            _, _, contamination = self.load_dataset_with_labels(dataset_name)
            contamination_rates[dataset_name] = contamination
        
        return contamination_rates

def get_dataset_contamination_rates() -> Dict[str, float]:
    """
    Get recommended contamination rates for standard UCR datasets
    
    Returns:
        Dictionary mapping dataset names to contamination rates
    """
    analyzer = UCRContaminationAnalyzer()
    
    # Use binary datasets that are well-suited for anomaly detection
    datasets = analyzer.binary_anomaly_datasets
    
    contamination_rates = analyzer.analyze_all_datasets(datasets)
    
    # Print summary
    print("\n" + "="*60)
    print("CONTAMINATION RATE SUMMARY")
    print("="*60)
    print(f"{'Dataset':<25} {'Contamination':>10} {'Percentage':>10}")
    print("-"*60)
    for dataset, rate in contamination_rates.items():
        print(f"{dataset:<25} {rate:>10.4f} {rate*100:>9.2f}%")
    
    return contamination_rates

def prepare_ucr_datasets_with_contamination() -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Load UCR datasets and return both data and contamination rates
    
    Returns:
        datasets: Dictionary of dataset_name -> time series data
        contamination_rates: Dictionary of dataset_name -> contamination rate
    """
    analyzer = UCRContaminationAnalyzer()
    
    # Use binary datasets for anomaly detection
    dataset_names = analyzer.binary_anomaly_datasets
    
    datasets = {}
    contamination_rates = {}
    
    for dataset_name in dataset_names:
        data, labels, contamination = analyzer.load_dataset_with_labels(dataset_name)
        if data is not None:
            datasets[dataset_name] = data
            contamination_rates[dataset_name] = contamination
    
    return datasets, contamination_rates

if __name__ == "__main__":
    # Example usage
    print("UCR Binary Dataset Contamination Analysis for Anomaly Detection")
    print("="*65)
    
    # Analyze individual dataset
    analyzer = UCRContaminationAnalyzer()
    
    # Show expected contamination rates
    print("\nExpected contamination rates from external analysis:")
    print("-"*65)
    expected_rates = {
        'Earthquakes': 0.0791,
        'Lightning2': 0.1639,
        'Chinatown': 0.0551,
        'SonyAIBORobotSurface1': 0.1681,
        'Wafer': 0.1045,
        'ECG200': 0.33,
        'ItalyPowerDemand': 0.4954,
        'ToeSegmentation2': 0.2769
    }
    for dataset, rate in expected_rates.items():
        print(f"{dataset:<25} {rate:.4f} ({rate*100:.2f}%)")
    
    # Test with Wafer dataset
    print("\n" + "="*65)
    data, labels, contamination = analyzer.load_dataset_with_labels('Wafer')
    
    if data is not None:
        print(f"\nWafer dataset loaded successfully")
        print(f"Calculated contamination: {contamination:.4f} ({contamination*100:.2f}%)")
        print(f"Expected contamination: {expected_rates['Wafer']:.4f} ({expected_rates['Wafer']*100:.2f}%)")
    
    # Get all contamination rates
    print("\n" + "="*65)
    contamination_rates = get_dataset_contamination_rates()
    
    # Compare with expected rates
    print("\n" + "="*65)
    print("COMPARISON WITH EXPECTED RATES")
    print("="*65)
    print(f"{'Dataset':<25} {'Calculated':>12} {'Expected':>12} {'Match':>8}")
    print("-"*65)
    for dataset in analyzer.binary_anomaly_datasets:
        if dataset in contamination_rates and dataset in expected_rates:
            calc_rate = contamination_rates[dataset]
            exp_rate = expected_rates[dataset]
            match = "✓" if abs(calc_rate - exp_rate) < 0.01 else "✗"
            print(f"{dataset:<25} {calc_rate:>12.4f} {exp_rate:>12.4f} {match:>8}")
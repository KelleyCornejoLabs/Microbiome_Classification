#!/usr/bin/env python3
"""
estimating centroids from multivariate labeled data

Usage:
    python centroid_estimator.py input_file.csv --label-column class
    python centroid_estimator.py data.csv -l target -o centroids.csv
"""

import numpy as np
import pandas as pd
import argparse
import sys
from pathlib import Path


def load_data(filepath, label_column, non_data_columns = []):
    """Load data from CSV or Excel file."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Load file based on extension
    if filepath.suffix.lower() == '.csv':
        df = pd.read_csv(filepath)
    elif filepath.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
    else:
        # Try CSV as default
        df = pd.read_csv(filepath)
    
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in {list(df.columns)}")
    
    # Extract features and labels
    y = df[label_column]
    X = df.drop(columns=[label_column]+non_data_columns)
    # X = X.drop(columns=["sampleID", "read_count","subCST","Pct I-A","Pct I-B","Pct II","Pct III-A","Pct III-B","Pct IV-A","Pct IV-B","Pct IV-C0","Pct IV-C1","Pct IV-C2","Pct IV-C3","Pct IV-C4","Pct V"])
    
    return X, y


def calculate_centroids(X, y, method='mean'):
    """Calculate centroids for each class."""
    centroids = {}
    unique_labels = np.unique(y)
    
    for label in unique_labels:
        mask = y == label
        class_data = X[mask].values
        
        if method == 'mean':
            centroid = np.mean(class_data, axis=0)
        elif method == 'median':
            centroid = np.median(class_data, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        centroids[label] = centroid
    
    return centroids


def save_centroids(centroids, feature_names, output_path):
    """Save centroids to CSV file."""
    centroids_data = []
    
    for label, centroid in centroids.items():
        row = {'sub_CST': label}
        for i, feature_name in enumerate(feature_names):
            row[feature_name] = centroid[i]
        centroids_data.append(row)
    
    df = pd.DataFrame(centroids_data)
    df.to_csv(output_path, index=False)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Estimate centroids from labeled data")
    
    parser.add_argument('input_file', help='Input CSV or Excel file')
    parser.add_argument('-l', '--label-column', required=True,
                       help='Name of the column containing class labels')
    parser.add_argument('-ndc', '--non-data-columns', help='Comma seperated list of columns which should not be used to calculate centroids, excluding label column')
    parser.add_argument('-o', '--output', help='Output CSV file (default: centroids.csv)')
    parser.add_argument('-m', '--method', choices=['mean', 'median'], 
                       default='mean', help='Centroid calculation method')
    
    args = parser.parse_args()
    
    try:
        # Load data
        X, y = load_data(args.input_file, args.label_column, args.non_data_columns.split(","))
        print(f"Loaded {len(X)} samples with {len(X.columns)} features")
        
        # Calculate centroids
        centroids = calculate_centroids(X, y, args.method)
        print(f"Calculated centroids for {len(centroids)} classes")
        
        # Save results
        output_path = args.output or 'centroids.csv'
        centroids_df = save_centroids(centroids, X.columns.tolist(), output_path)
        
        print(f"Centroids saved to: {output_path}")
        print("\nCentroids:")
        print(centroids_df)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Compare Results
Detailed comparison of experimental results with reference values.
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}\n")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓{Colors.END} {text}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗{Colors.END} {text}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")

def load_results(file_path: Path) -> pd.DataFrame:
    """Load results from CSV."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print_error(f"Failed to load {file_path}: {str(e)}")
        return None

def compute_statistics(df: pd.DataFrame) -> Dict:
    """Compute statistics for numeric columns."""
    stats = {}
    
    for col in df.select_dtypes(include=[np.number]).columns:
        stats[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median(),
        }
    
    return stats

def compare_dataframes(
    actual_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    tolerance: float
) -> Dict:
    """Compare two dataframes with tolerance."""
    
    comparison = {
        'passed': True,
        'details': []
    }
    
    # Get common numeric columns
    actual_numeric = set(actual_df.select_dtypes(include=[np.number]).columns)
    reference_numeric = set(reference_df.select_dtypes(include=[np.number]).columns)
    common_cols = actual_numeric & reference_numeric
    
    for col in common_cols:
        actual_mean = actual_df[col].mean()
        reference_mean = reference_df[col].mean()
        
        if reference_mean != 0:
            relative_diff = abs(actual_mean - reference_mean) / abs(reference_mean)
        else:
            relative_diff = abs(actual_mean - reference_mean)
        
        passed = relative_diff <= tolerance
        
        comparison['details'].append({
            'column': col,
            'actual_mean': actual_mean,
            'reference_mean': reference_mean,
            'relative_diff': relative_diff * 100,
            'passed': passed
        })
        
        if not passed:
            comparison['passed'] = False
    
    return comparison

def print_comparison_table(comparison: Dict):
    """Print formatted comparison table."""
    
    # Header
    print(f"\n{'Column':<30} {'Actual':<15} {'Reference':<15} {'Diff (%)':<12} {'Status':<10}")
    print('-' * 85)
    
    # Rows
    for detail in comparison['details']:
        col = detail['column'][:28]
        actual = f"{detail['actual_mean']:.2f}"
        reference = f"{detail['reference_mean']:.2f}"
        diff = f"{detail['relative_diff']:.2f}"
        
        if detail['passed']:
            status = f"{Colors.GREEN}PASS{Colors.END}"
        else:
            status = f"{Colors.RED}FAIL{Colors.END}"
        
        print(f"{col:<30} {actual:<15} {reference:<15} {diff:<12} {status:<10}")

def main():
    """Main comparison routine."""
    parser = argparse.ArgumentParser(description="Compare experimental results")
    parser.add_argument('--result_file', type=str, required=True,
                       help='Result CSV file')
    parser.add_argument('--reference_file', type=str, required=True,
                       help='Reference CSV file')
    parser.add_argument('--tolerance', type=str, default='5%',
                       help='Tolerance percentage (default: 5%%)')
    parser.add_argument('--show_stats', action='store_true',
                       help='Show detailed statistics')
    
    args = parser.parse_args()
    
    # Parse tolerance
    tolerance_str = args.tolerance.rstrip('%')
    tolerance = float(tolerance_str) / 100.0
    
    print_header("Results Comparison")
    
    print(f"Result file: {args.result_file}")
    print(f"Reference file: {args.reference_file}")
    print(f"Tolerance: {tolerance*100:.1f}%\n")
    
    # Load data
    actual_df = load_results(Path(args.result_file))
    reference_df = load_results(Path(args.reference_file))
    
    if actual_df is None or reference_df is None:
        print_error("Failed to load data files")
        return 1
    
    print(f"Actual data shape: {actual_df.shape}")
    print(f"Reference data shape: {reference_df.shape}")
    
    # Show statistics if requested
    if args.show_stats:
        print(f"\n{Colors.BOLD}Actual Results Statistics:{Colors.END}")
        actual_stats = compute_statistics(actual_df)
        for col, stats in actual_stats.items():
            print(f"\n{col}:")
            for stat_name, stat_val in stats.items():
                print(f"  {stat_name}: {stat_val:.4f}")
        
        print(f"\n{Colors.BOLD}Reference Results Statistics:{Colors.END}")
        reference_stats = compute_statistics(reference_df)
        for col, stats in reference_stats.items():
            print(f"\n{col}:")
            for stat_name, stat_val in stats.items():
                print(f"  {stat_name}: {stat_val:.4f}")
    
    # Compare
    print(f"\n{Colors.BOLD}Detailed Comparison:{Colors.END}")
    comparison = compare_dataframes(actual_df, reference_df, tolerance)
    print_comparison_table(comparison)
    
    # Summary
    print_header("Comparison Summary")
    
    passed_count = sum(1 for d in comparison['details'] if d['passed'])
    failed_count = sum(1 for d in comparison['details'] if not d['passed'])
    total_count = len(comparison['details'])
    
    print(f"Passed: {passed_count}/{total_count}")
    print(f"Failed: {failed_count}/{total_count}")
    print(f"Pass rate: {passed_count/total_count*100:.1f}%")
    
    if comparison['passed']:
        print(f"\n{Colors.GREEN}✓ All metrics within {tolerance*100:.1f}% tolerance{Colors.END}\n")
        return 0
    else:
        print(f"\n{Colors.RED}✗ Some metrics exceed {tolerance*100:.1f}% tolerance{Colors.END}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())

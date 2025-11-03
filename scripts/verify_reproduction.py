#!/usr/bin/env python3
"""
Verify Reproduction Results
Compares reproduced results against reference values with tolerance.
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

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
    """Load results from CSV file."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print_error(f"Failed to load {file_path}: {str(e)}")
        return None

def compare_values(actual: float, expected: float, tolerance: float) -> Tuple[bool, float]:
    """
    Compare actual vs expected with tolerance.
    
    Returns:
        (passed, relative_error)
    """
    if expected == 0:
        relative_error = abs(actual - expected)
    else:
        relative_error = abs(actual - expected) / abs(expected)
    
    passed = relative_error <= tolerance
    return passed, relative_error * 100  # Convert to percentage

def verify_table(
    actual_file: Path,
    reference_file: Path,
    table_name: str,
    tolerance: float
) -> bool:
    """Verify a single table against reference."""
    
    print(f"\n{Colors.BOLD}Verifying {table_name}...{Colors.END}")
    
    # Load data
    actual_df = load_results(actual_file)
    reference_df = load_results(reference_file)
    
    if actual_df is None or reference_df is None:
        print_error(f"{table_name}: Failed to load data")
        return False
    
    # Check dimensions
    if actual_df.shape != reference_df.shape:
        print_warning(f"{table_name}: Different dimensions")
        print(f"  Actual: {actual_df.shape}, Reference: {reference_df.shape}")
    
    all_passed = True
    
    # Compare numeric columns
    numeric_cols = actual_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col not in reference_df.columns:
            print_warning(f"{table_name}: Column '{col}' not in reference")
            continue
        
        actual_vals = actual_df[col].values
        reference_vals = reference_df[col].values
        
        # Element-wise comparison
        passed_count = 0
        failed_count = 0
        max_error = 0
        
        for i, (actual_val, ref_val) in enumerate(zip(actual_vals, reference_vals)):
            if pd.isna(actual_val) or pd.isna(ref_val):
                continue
            
            passed, error = compare_values(actual_val, ref_val, tolerance)
            
            if passed:
                passed_count += 1
            else:
                failed_count += 1
                all_passed = False
            
            max_error = max(max_error, error)
        
        # Report column results
        if failed_count == 0:
            print_success(f"  {col}: All values within {tolerance*100:.1f}% tolerance (max error: {max_error:.2f}%)")
        else:
            print_error(f"  {col}: {failed_count}/{len(actual_vals)} values exceed tolerance (max error: {max_error:.2f}%)")
    
    return all_passed

def main():
    """Main verification routine."""
    parser = argparse.ArgumentParser(description="Verify reproduction results")
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing reproduced results')
    parser.add_argument('--reference_dir', type=str, required=True,
                       help='Directory containing reference results')
    parser.add_argument('--tolerance', type=float, default=5.0,
                       help='Tolerance percentage (default: 5.0%%)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    reference_dir = Path(args.reference_dir)
    tolerance = args.tolerance / 100.0  # Convert to decimal
    
    print_header("PPO-GNN Results Verification")
    
    print(f"Results directory: {results_dir}")
    print(f"Reference directory: {reference_dir}")
    print(f"Tolerance: {args.tolerance}%\n")
    
    # Tables to verify
    tables = {
        'Table 5.2 (Ablation Study)': (
            results_dir / 'ablation' / 'table_5_2.csv',
            reference_dir / 'ablation' / 'table_5_2.csv'
        ),
        'Table 5.3 (Generalization)': (
            results_dir / 'ablation' / 'table_5_3.csv',
            reference_dir / 'ablation' / 'table_5_3.csv'
        ),
        'Table 5.4 (Benchmark)': (
            results_dir / 'benchmark' / 'table_5_4.csv',
            reference_dir / 'benchmark' / 'table_5_4.csv'
        ),
        'Table 5.5 (Gurobi Multi-Scale)': (
            results_dir / 'gaps' / 'table_5_5.csv',
            reference_dir / 'gaps' / 'table_5_5.csv'
        ),
        'Table 5.6 (LP Bounds)': (
            results_dir / 'lp_bounds' / 'table_5_6.csv',
            reference_dir / 'lp_bounds' / 'table_5_6.csv'
        ),
    }
    
    results = {}
    
    for table_name, (actual_file, reference_file) in tables.items():
        # Check if files exist
        if not actual_file.exists():
            print_warning(f"{table_name}: Actual file not found")
            results[table_name] = None
            continue
        
        if not reference_file.exists():
            print_warning(f"{table_name}: Reference file not found (skipping)")
            results[table_name] = None
            continue
        
        # Verify table
        passed = verify_table(actual_file, reference_file, table_name, tolerance)
        results[table_name] = passed
    
    # Summary
    print_header("Verification Summary")
    
    verified_count = sum(1 for v in results.values() if v is True)
    failed_count = sum(1 for v in results.values() if v is False)
    skipped_count = sum(1 for v in results.values() if v is None)
    
    for table_name, passed in results.items():
        if passed is True:
            print_success(f"{table_name}: PASSED")
        elif passed is False:
            print_error(f"{table_name}: FAILED")
        else:
            print_warning(f"{table_name}: SKIPPED")
    
    print(f"\nSummary:")
    print(f"  Passed: {verified_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Total: {len(results)}")
    
    if failed_count == 0 and verified_count > 0:
        print(f"\n{Colors.GREEN}✓ All results verified successfully!{Colors.END}\n")
        return 0
    elif failed_count > 0:
        print(f"\n{Colors.RED}✗ Some results differ from reference{Colors.END}")
        print(f"Check individual table comparisons above for details.\n")
        return 1
    else:
        print(f"\n{Colors.YELLOW}⚠ No results could be verified{Colors.END}\n")
        return 2

if __name__ == "__main__":
    sys.exit(main())

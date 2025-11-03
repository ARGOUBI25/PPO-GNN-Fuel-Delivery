#!/usr/bin/env python3
"""
Verify Installation Script
Checks all dependencies and system configuration for PPO-GNN framework.
"""

import sys
import subprocess
import importlib
from typing import Dict, Tuple, List

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

def check_python_version() -> bool:
    """Check Python version (>= 3.8)."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor >= 8:
        print_success(f"Python version: {version_str}")
        return True
    else:
        print_error(f"Python version: {version_str} (requires >= 3.8)")
        return False

def check_package(package_name: str, min_version: str = None) -> bool:
    """Check if package is installed with optional version check."""
    try:
        module = importlib.import_module(package_name)
        
        # Get version
        if hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'VERSION'):
            version = module.VERSION
        else:
            version = "unknown"
        
        # Version check if specified
        if min_version and version != "unknown":
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                print_error(f"{package_name}: {version} (requires >= {min_version})")
                return False
        
        print_success(f"{package_name}: {version}")
        return True
        
    except ImportError:
        print_error(f"{package_name}: not installed")
        return False

def check_cuda() -> Tuple[bool, str]:
    """Check CUDA availability."""
    try:
        import torch
        
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            
            print_success(f"CUDA available: {cuda_version}")
            print_success(f"GPU device: {device_name}")
            print_success(f"GPU count: {device_count}")
            
            # Test GPU computation
            try:
                x = torch.randn(100, 100).cuda()
                y = torch.randn(100, 100).cuda()
                z = torch.mm(x, y)
                print_success("GPU computation test: passed")
                return True, device_name
            except Exception as e:
                print_error(f"GPU computation test: failed ({str(e)})")
                return False, "N/A"
        else:
            print_warning("CUDA not available (CPU-only mode)")
            return False, "N/A"
            
    except ImportError:
        print_error("PyTorch not installed")
        return False, "N/A"

def check_pytorch_geometric() -> bool:
    """Check PyTorch Geometric installation."""
    try:
        import torch_geometric
        from torch_geometric.data import Data
        import torch
        
        # Test basic functionality
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        x = torch.randn(3, 16)
        data = Data(x=x, edge_index=edge_index)
        
        print_success(f"PyTorch Geometric: {torch_geometric.__version__}")
        print_success("PyTorch Geometric test: passed")
        return True
        
    except ImportError as e:
        print_error(f"PyTorch Geometric: not installed ({str(e)})")
        return False
    except Exception as e:
        print_error(f"PyTorch Geometric test failed: {str(e)}")
        return False

def check_gurobi() -> bool:
    """Check Gurobi installation (optional)."""
    try:
        import gurobipy
        
        # Try to get version
        try:
            version = gurobipy.gurobi.version()
            version_str = f"{version[0]}.{version[1]}.{version[2]}"
            print_success(f"Gurobi: {version_str}")
            
            # Test license
            try:
                env = gurobipy.Env()
                env.dispose()
                print_success("Gurobi license: valid")
                return True
            except Exception as e:
                print_warning(f"Gurobi license: invalid ({str(e)})")
                return False
                
        except Exception as e:
            print_warning(f"Gurobi version check failed: {str(e)}")
            return False
            
    except ImportError:
        print_warning("Gurobi: not installed (optional)")
        return False

def check_system_info():
    """Display system information."""
    import platform
    import os
    
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # Check available memory
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"RAM: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available")
    except ImportError:
        print_warning("psutil not installed (can't check RAM)")

def main():
    """Main verification routine."""
    print_header("PPO-GNN Installation Verification")
    
    all_passed = True
    
    # System info
    print(f"\n{Colors.BOLD}System Information:{Colors.END}")
    check_system_info()
    
    # Python version
    print(f"\n{Colors.BOLD}Python Environment:{Colors.END}")
    all_passed &= check_python_version()
    
    # Core dependencies
    print(f"\n{Colors.BOLD}Core Deep Learning Frameworks:{Colors.END}")
    
    required_packages = {
        'torch': '1.12.0',
        'torchvision': '0.13.0',
        'numpy': '1.21.0',
    }
    
    for package, min_version in required_packages.items():
        all_passed &= check_package(package, min_version)
    
    # CUDA check
    print(f"\n{Colors.BOLD}CUDA/GPU Support:{Colors.END}")
    cuda_available, gpu_name = check_cuda()
    
    # PyTorch Geometric
    print(f"\n{Colors.BOLD}Graph Neural Networks:{Colors.END}")
    all_passed &= check_pytorch_geometric()
    
    # Additional packages
    print(f"\n{Colors.BOLD}Additional Dependencies:{Colors.END}")
    
    additional_packages = [
        'pandas',
        'matplotlib',
        'seaborn',
        'networkx',
        'yaml',
        'tqdm',
        'tensorboard',
        'scipy',
    ]
    
    for package in additional_packages:
        check_package(package)
    
    # Optional: Gurobi
    print(f"\n{Colors.BOLD}Optional Dependencies:{Colors.END}")
    gurobi_available = check_gurobi()
    
    # Summary
    print_header("Installation Summary")
    
    if all_passed:
        print_success("All required dependencies are installed correctly!")
        
        if cuda_available:
            print_success(f"GPU acceleration enabled ({gpu_name})")
        else:
            print_warning("GPU not available - training will be slower on CPU")
        
        if not gurobi_available:
            print_warning("Gurobi not available - exact solver comparison will be skipped")
        
        print(f"\n{Colors.GREEN}Ready to train PPO-GNN models!{Colors.END}\n")
        return 0
    else:
        print_error("Some dependencies are missing or have incorrect versions")
        print(f"\n{Colors.RED}Please fix the issues above before proceeding.{Colors.END}")
        print("\nInstallation guide: docs/INSTALLATION.md")
        print("Run: pip install -r requirements.txt\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())

#!/bin/bash

# ============================================================================
# Complete Reproduction Pipeline
# Reproduces all tables and figures from the paper
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DEVICE="cuda"
USE_GUROBI=true
NUM_SEEDS=5
OUTPUT_DIR="results/reproduction"
PARALLEL_JOBS=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --no-gurobi)
            USE_GUROBI=false
            shift
            ;;
        --seeds)
            NUM_SEEDS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash scripts/reproduce_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --device <cuda|cpu>    Device to use (default: cuda)"
            echo "  --no-gurobi           Skip Gurobi experiments"
            echo "  --seeds <N>           Number of random seeds (default: 5)"
            echo "  --output <DIR>        Output directory (default: results/reproduction)"
            echo "  --parallel <N>        Number of parallel jobs (default: 1)"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/ablation"
mkdir -p "$OUTPUT_DIR/benchmark"
mkdir -p "$OUTPUT_DIR/gaps"
mkdir -p "$OUTPUT_DIR/lp_bounds"
mkdir -p "$OUTPUT_DIR/figures"

# Log file
LOG_FILE="$OUTPUT_DIR/reproduction.log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}PPO-GNN Paper Results Reproduction Pipeline${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo "Configuration:"
echo "  Device: $DEVICE"
echo "  Use Gurobi: $USE_GUROBI"
echo "  Number of seeds: $NUM_SEEDS"
echo "  Output directory: $OUTPUT_DIR"
echo "  Parallel jobs: $PARALLEL_JOBS"
echo "  Log file: $LOG_FILE"
echo ""

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
python scripts/verify_installation.py
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Dependency check failed. Please run installation first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ All dependencies satisfied${NC}"
echo ""

# Start timer
START_TIME=$(date +%s)

# ============================================================================
# Section 5.2: Ablation Study (Table 5.2)
# ============================================================================
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}Section 5.2: Ablation Study (Table 5.2)${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

SEEDS="42 123 456 789 1011"
ABLATION_METHODS="ppo_flat ppo_mlp ppo_gnn"

echo -e "${YELLOW}Training models with $NUM_SEEDS seeds...${NC}"
echo "Methods: $ABLATION_METHODS"
echo "Dataset: data/synthetic_networks/medium_50_nodes"
echo "Episodes: 50,000"
echo ""

python experiments/ablation_study.py \
    --methods $ABLATION_METHODS \
    --dataset data/synthetic_networks/medium_50_nodes \
    --num_episodes 50000 \
    --num_test_instances 20 \
    --seeds $SEEDS \
    --device $DEVICE \
    --output "$OUTPUT_DIR/ablation/table_5_2.csv"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Table 5.2 (Ablation Study) completed${NC}"
else
    echo -e "${RED}✗ Table 5.2 failed${NC}"
    exit 1
fi
echo ""

# ============================================================================
# Section 5.2: Generalization Test (Table 5.3)
# ============================================================================
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}Section 5.2: Generalization Test (Table 5.3)${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

echo -e "${YELLOW}Testing generalization across network sizes...${NC}"
echo "Trained size: 50 nodes"
echo "Test sizes: 30, 50, 70 nodes"
echo ""

python experiments/generalization_test.py \
    --trained_size 50 \
    --test_sizes 30 50 70 \
    --methods ppo_flat ppo_mlp ppo_gnn \
    --num_test_instances 10 \
    --checkpoint_dir checkpoints/ \
    --device $DEVICE \
    --output "$OUTPUT_DIR/ablation/table_5_3.csv"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Table 5.3 (Generalization) completed${NC}"
else
    echo -e "${RED}✗ Table 5.3 failed${NC}"
    exit 1
fi
echo ""

# ============================================================================
# Section 5.3: Overall Performance Comparison (Table 5.4)
# ============================================================================
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}Section 5.3: Performance Comparison (Table 5.4)${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

BENCHMARK_METHODS="ppo_gnn classical_ppo clarke_wright"
if [ "$USE_GUROBI" = true ]; then
    BENCHMARK_METHODS="gurobi $BENCHMARK_METHODS"
    echo -e "${YELLOW}Including Gurobi in benchmark...${NC}"
else
    echo -e "${YELLOW}Skipping Gurobi (use --no-gurobi flag was set)${NC}"
fi

echo "Methods: $BENCHMARK_METHODS"
echo "Dataset: data/synthetic_networks/large_100_nodes"
echo "Test instances: 20"
echo ""

python experiments/benchmark_comparison.py \
    --methods $BENCHMARK_METHODS \
    --dataset data/synthetic_networks/large_100_nodes \
    --num_test_instances 20 \
    --gurobi_time_limit 7200 \
    --gurobi_threads 8 \
    --device $DEVICE \
    --output "$OUTPUT_DIR/benchmark/table_5_4.csv"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Table 5.4 (Benchmark) completed${NC}"
else
    echo -e "${RED}✗ Table 5.4 failed${NC}"
    exit 1
fi
echo ""

# ============================================================================
# Section 5.4: Detailed Gurobi Comparison (Table 5.5)
# ============================================================================
if [ "$USE_GUROBI" = true ]; then
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BLUE}Section 5.4: Multi-Scale Gurobi Comparison (Table 5.5)${NC}"
    echo -e "${BLUE}============================================================================${NC}"
    echo ""

    echo -e "${YELLOW}Running multi-scale comparison with Gurobi...${NC}"
    echo "Sizes: 10, 50, 100, 200 nodes"
    echo "Instances per size: 10"
    echo "Gurobi time limit: 4 hours (14,400s)"
    echo ""

    python experiments/optimality_gap_analysis.py \
        --sizes 10 50 100 200 \
        --methods gurobi ppo_gnn \
        --num_instances_per_size 10 \
        --gurobi_time_limit 14400 \
        --gurobi_threads 8 \
        --device $DEVICE \
        --output "$OUTPUT_DIR/gaps/table_5_5.csv"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Table 5.5 (Gurobi Multi-Scale) completed${NC}"
    else
        echo -e "${RED}✗ Table 5.5 failed${NC}"
        exit 1
    fi
    echo ""

    # ========================================================================
    # Section 5.4: LP Lower Bounds (Table 5.6)
    # ========================================================================
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BLUE}Section 5.4: LP Lower Bounds (Table 5.6)${NC}"
    echo -e "${BLUE}============================================================================${NC}"
    echo ""

    echo -e "${YELLOW}Computing LP relaxation lower bounds...${NC}"
    echo "Sizes: 10, 50, 100, 200 nodes"
    echo ""

    python experiments/lp_relaxation_bounds.py \
        --sizes 10 50 100 200 \
        --num_instances_per_size 10 \
        --methods lp_relaxation gurobi ppo_gnn \
        --device $DEVICE \
        --output "$OUTPUT_DIR/lp_bounds/table_5_6.csv"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Table 5.6 (LP Bounds) completed${NC}"
    else
        echo -e "${RED}✗ Table 5.6 failed${NC}"
        exit 1
    fi
    echo ""
else
    echo -e "${YELLOW}Skipping Tables 5.5 and 5.6 (require Gurobi)${NC}"
    echo ""
fi

# ============================================================================
# Section 5.5: Visualizations
# ============================================================================
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}Section 5.5: Generating Figures${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

# Figure 5.1: Cost and Unmet Demand Comparison
echo -e "${YELLOW}Generating Figure 5.1 (Cost and Unmet Demand Comparison)...${NC}"
python src/utils/visualization.py \
    --plot_type cost_unmet_comparison \
    --results_files "$OUTPUT_DIR/benchmark/table_5_4.csv" \
    --methods gurobi ppo_gnn classical_ppo clarke_wright \
    --output "$OUTPUT_DIR/figures/figure_5_1_cost_unmet.png" \
    --format png \
    --dpi 300

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Figure 5.1 generated${NC}"
else
    echo -e "${RED}✗ Figure 5.1 failed${NC}"
fi
echo ""

# Figure 5.2: Route Visualization
echo -e "${YELLOW}Generating Figure 5.2 (Route Visualization)...${NC}"
python src/utils/visualization.py \
    --plot_type route_comparison \
    --instance data/synthetic_networks/large_100_nodes/instance_01.json \
    --methods ppo_gnn classical_ppo clarke_wright \
    --checkpoints checkpoints/ppo_gnn_best.pth checkpoints/classical_ppo_best.pth \
    --output "$OUTPUT_DIR/figures/figure_5_2_routes.png" \
    --format png \
    --dpi 300

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Figure 5.2 generated${NC}"
else
    echo -e "${RED}✗ Figure 5.2 failed${NC}"
fi
echo ""

# ============================================================================
# Verification
# ============================================================================
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}Verifying Results${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

echo -e "${YELLOW}Running verification script...${NC}"
python scripts/verify_reproduction.py \
    --results_dir "$OUTPUT_DIR" \
    --reference_dir results/reference \
    --tolerance 5

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All results verified successfully${NC}"
else
    echo -e "${YELLOW}⚠ Some results differ from reference (check logs)${NC}"
fi
echo ""

# ============================================================================
# Summary
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}Reproduction Complete!${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - Table 5.2 (Ablation): $OUTPUT_DIR/ablation/table_5_2.csv"
echo "  - Table 5.3 (Generalization): $OUTPUT_DIR/ablation/table_5_3.csv"
echo "  - Table 5.4 (Benchmark): $OUTPUT_DIR/benchmark/table_5_4.csv"
if [ "$USE_GUROBI" = true ]; then
    echo "  - Table 5.5 (Gurobi Multi-Scale): $OUTPUT_DIR/gaps/table_5_5.csv"
    echo "  - Table 5.6 (LP Bounds): $OUTPUT_DIR/lp_bounds/table_5_6.csv"
fi
echo "  - Figure 5.1: $OUTPUT_DIR/figures/figure_5_1_cost_unmet.png"
echo "  - Figure 5.2: $OUTPUT_DIR/figures/figure_5_2_routes.png"
echo ""
echo "Log file: $LOG_FILE"
echo ""
echo -e "${GREEN}✓ All experiments completed successfully!${NC}"

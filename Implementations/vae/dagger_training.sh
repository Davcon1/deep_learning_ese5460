#!/bin/bash
# dagger_pipeline.sh
# Automated DAgger training pipeline for VAE motion policy

set -e  # Exit on error

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${SCRIPT_DIR}/../policies/trained_policies"
DATASET_DIR="${SCRIPT_DIR}/../datasets"
MODEL_DIR="${SCRIPT_DIR}/models"
MOTION_DIR="${SCRIPT_DIR}/../policies/motion_files"

# Training parameters
NUM_DAGGER_ITERATIONS=5
VAE_EPOCHS=100
BATCH_SIZE=2048
LEARNING_RATE=5e-4
BETA=0.01

# Trajectory collection parameters
NUM_ENVS=300
NUM_ROLLOUTS=150
ROLLOUT_LENGTH=500
MIN_TRAJ_LENGTH=150

# Motion files to use (should match checkpoint names)
MOTIONS=(
    "dance1_subject2"
    "dance2_subject3"
    "walk1_subject2"
    "walk1_subject5"
    "walk2_subject3"
    "run1_subject2"
    "run2_subject1"
    "run2_subject4"
    "jumps1_subject1"
    "jumps1_subject2"
)

# Task name
TASK="Tracking-Flat-G1-v0"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

print_header() {
    echo ""
    echo "============================================================"
    echo " $1"
    echo "============================================================"
    echo ""
}

print_step() {
    echo ">>> $1"
}

check_file_exists() {
    if [ ! -f "$1" ]; then
        echo "ERROR: File not found: $1"
        exit 1
    fi
}

# =============================================================================
# STEP 0: INITIAL SETUP
# =============================================================================

print_header "DAGGER PIPELINE - INITIALIZATION"

# Create directories
mkdir -p "$DATASET_DIR"
mkdir -p "$MODEL_DIR"

print_step "Directories created"
print_step "Dataset dir: $DATASET_DIR"
print_step "Model dir: $MODEL_DIR"

# =============================================================================
# STEP 1: INITIAL TRAJECTORY COLLECTION (Iteration 0 - Expert Only)
# =============================================================================

collect_initial_trajectories() {
    print_header "ITERATION 0: COLLECTING EXPERT TRAJECTORIES"
    
    for MOTION in "${MOTIONS[@]}"; do
        CHECKPOINT="${CHECKPOINT_DIR}/${MOTION}.pt"
        OUTPUT="${DATASET_DIR}/traj_${MOTION}_iter0.pt"
        
        # Skip if output file already exists
        if [ -f "$OUTPUT" ]; then
            echo "Skipping $MOTION - trajectory file already exists: $OUTPUT"
            continue
        fi
        
        print_step "Collecting trajectories for: $MOTION"
        
        check_file_exists "$CHECKPOINT"
        
        python "${SCRIPT_DIR}/generate_trajectories.py" \
            --task "$TASK" \
            --checkpoint "$CHECKPOINT" \
            --motion_name "$MOTION" \
            --num_envs "$NUM_ENVS" \
            --num_rollouts "$NUM_ROLLOUTS" \
            --rollout_length "$ROLLOUT_LENGTH" \
            --min_traj_length "$MIN_TRAJ_LENGTH" \
            --output "$OUTPUT" \
            --headless
        
        if [ $? -eq 0 ]; then
            echo "✓ Successfully collected trajectories for $MOTION"
        else
            echo "✗ Failed to collect trajectories for $MOTION"
            exit 1
        fi
    done
}

# =============================================================================
# STEP 2: COMBINE DATASETS
# =============================================================================

combine_datasets() {
    local ITERATION=$1
    local OUTPUT_FILE="${DATASET_DIR}/combined_dataset_iter${ITERATION}.pt"
    
    print_header "COMBINING DATASETS FOR ITERATION $ITERATION"
    
    # Skip if combined file already exists
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Skipping - combined dataset already exists: $OUTPUT_FILE"
        return 0
    fi
    
    python "${SCRIPT_DIR}/combine_datasets.py" \
        --input_dir "$DATASET_DIR" \
        --output "$OUTPUT_FILE" \
        --pattern "traj_*_iter${ITERATION}.pt"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully combined datasets for iteration $ITERATION"
    else
        echo "✗ Failed to combine datasets"
        exit 1
    fi
}

# =============================================================================
# STEP 3: TRAIN VAE
# =============================================================================

train_vae() {
    local ITERATION=$1
    
    print_header "TRAINING VAE - ITERATION $ITERATION"
    
    local DATA_PATH="${DATASET_DIR}/combined_dataset_iter${ITERATION}.pt"
    local OUTPUT_DIR="${MODEL_DIR}/iter${ITERATION}"
    local MODEL_FILE="${OUTPUT_DIR}/vae_model_final.pth"
    
    # Check if model already exists in iter folder
    if [ -f "$MODEL_FILE" ]; then
        echo "Skipping - VAE model already exists: $MODEL_FILE"
        return 0
    fi
    
    # Check if model exists in base models folder (from manual training)
    local ALT_MODEL_FILE="${MODEL_DIR}/vae_model_final.pth"
    local ALT_MODEL_FILE_ITER="${MODEL_DIR}/vae_model_100_iter${ITERATION}.pth"
    
    if [ "$ITERATION" -eq 0 ]; then
        if [ -f "$ALT_MODEL_FILE" ]; then
            echo "Found existing model at $ALT_MODEL_FILE"
            mkdir -p "$OUTPUT_DIR"
            cp "$ALT_MODEL_FILE" "$MODEL_FILE"
            echo "✓ Copied to $MODEL_FILE"
            return 0
        elif [ -f "$ALT_MODEL_FILE_ITER" ]; then
            echo "Found existing model at $ALT_MODEL_FILE_ITER"
            mkdir -p "$OUTPUT_DIR"
            cp "$ALT_MODEL_FILE_ITER" "$MODEL_FILE"
            echo "✓ Copied to $MODEL_FILE"
            return 0
        fi
    fi
    
    check_file_exists "$DATA_PATH"
    
    python "${SCRIPT_DIR}/train.py" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --epochs "$VAE_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LEARNING_RATE" \
        --beta "$BETA" \
        --iter "$ITERATION"
        
    if [ $? -eq 0 ]; then
        echo "✓ Successfully trained VAE for iteration $ITERATION"
    else
        echo "✗ Failed to train VAE"
        exit 1
    fi
}

# =============================================================================
# STEP 4: DAGGER TRAJECTORY COLLECTION (Iterations 1+)
# =============================================================================

collect_dagger_trajectories() {
    local ITERATION=$1
    local PREV_ITERATION=$((ITERATION - 1))
    
    print_header "ITERATION $ITERATION: COLLECTING DAGGER TRAJECTORIES"
    
    local MODEL_PATH="${MODEL_DIR}/iter${PREV_ITERATION}/vae_model_final.pth"
    check_file_exists "$MODEL_PATH"
    
    for MOTION in "${MOTIONS[@]}"; do
        CHECKPOINT="${CHECKPOINT_DIR}/${MOTION}.pt"
        OUTPUT="${DATASET_DIR}/traj_${MOTION}_iter${ITERATION}.pt"
        
        # Skip if output file already exists
        if [ -f "$OUTPUT" ]; then
            echo "Skipping $MOTION - trajectory file already exists: $OUTPUT"
            continue
        fi
        
        print_step "Collecting DAgger trajectories for: $MOTION (iteration $ITERATION)"
        
        check_file_exists "$CHECKPOINT"
        
        python "${SCRIPT_DIR}/dagger.py" \
            --task "$TASK" \
            --checkpoint "$CHECKPOINT" \
            --motion_name "$MOTION" \
            --model_path "$MODEL_PATH" \
            --iteration "$ITERATION" \
            --num_envs "$NUM_ENVS" \
            --num_rollouts "$NUM_ROLLOUTS" \
            --rollout_length "$ROLLOUT_LENGTH" \
            --min_traj_length "$MIN_TRAJ_LENGTH" \
            --output "$OUTPUT" \
            --headless
        
        if [ $? -eq 0 ]; then
            echo "✓ Successfully collected DAgger trajectories for $MOTION"
        else
            echo "✗ Failed to collect DAgger trajectories for $MOTION"
            exit 1
        fi
    done
}

# =============================================================================
# STEP 5: AGGREGATE ALL DATASETS
# =============================================================================

aggregate_all_datasets() {
    local ITERATION=$1
    local OUTPUT_FILE="${DATASET_DIR}/combined_dataset_iter${ITERATION}.pt"
    
    print_header "AGGREGATING ALL DATASETS UP TO ITERATION $ITERATION"
    
    # Skip if combined file already exists
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Skipping - aggregated dataset already exists: $OUTPUT_FILE"
        return 0
    fi
    
    # Combine all iterations into one dataset
    python "${SCRIPT_DIR}/combine_datasets.py" \
        --input_dir "$DATASET_DIR" \
        --output "$OUTPUT_FILE" \
        --pattern "traj_*_iter*.pt" \
        --max_iteration "$ITERATION"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully aggregated all datasets"
    else
        echo "✗ Failed to aggregate datasets"
        exit 1
    fi
}

# =============================================================================
# MAIN PIPELINE
# =============================================================================

main() {
    print_header "STARTING DAGGER PIPELINE"
    echo "Number of iterations: $NUM_DAGGER_ITERATIONS"
    echo "Motions: ${MOTIONS[*]}"
    echo ""
    
    # Iteration 0: Expert-only data collection
    print_header "========== ITERATION 0 (EXPERT ONLY) =========="
    collect_initial_trajectories
    combine_datasets 0
    train_vae 0
    
    # Iterations 1+: DAgger loop
    for ((i=1; i<=NUM_DAGGER_ITERATIONS; i++)); do
        print_header "========== DAGGER ITERATION $i =========="
        
        # Collect new trajectories using current VAE + expert
        collect_dagger_trajectories $i
        
        # Aggregate all data (from all iterations)
        aggregate_all_datasets $i
        
        # Train VAE on aggregated dataset
        train_vae $i
        
        echo ""
        echo "✓ Completed DAgger iteration $i"
        echo ""
    done
    
    print_header "DAGGER PIPELINE COMPLETE"
    echo "Final model saved to: ${MODEL_DIR}/iter${NUM_DAGGER_ITERATIONS}/vae_model_final.pth"
}

# =============================================================================
# RUN
# =============================================================================

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --iterations)
            NUM_DAGGER_ITERATIONS="$2"
            shift 2
            ;;
        --epochs)
            VAE_EPOCHS="$2"
            shift 2
            ;;
        --num_envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --num_rollouts)
            NUM_ROLLOUTS="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --iterations N      Number of DAgger iterations (default: 5)"
            echo "  --epochs N          VAE training epochs (default: 100)"
            echo "  --num_envs N        Number of parallel environments (default: 512)"
            echo "  --num_rollouts N    Number of rollouts per motion (default: 250)"
            echo "  --force             Force regeneration of all files (ignore existing)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main pipeline
main
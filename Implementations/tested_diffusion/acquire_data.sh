VAE_MODEL_PATH="../vae/models/iter4/vae_model_100_iter4.pth"
OUTPUT_DIR="./data"
NUM_ENVS=5000
NUM_ROLLOUTS=100
ROLLOUT_LENGTH=150
OU_SIGMA=0.1

mkdir -p $OUTPUT_DIR

MOTIONS=(
    "walk1_subject2"
    "walk1_subject5"
    "walk2_subject3"
    "run1_subject2"
    "run2_subject1"
    "run2_subject4"
)

for motion in "${MOTIONS[@]}"; do
    echo ""
    echo "========================================"
    echo "[START] $motion"
    echo "========================================"
    
    python data_collection.py \
        --task Tracking-Flat-G1-v0 \
        --model_path $VAE_MODEL_PATH \
        --motion_name $motion \
        --num_rollouts $NUM_ROLLOUTS \
        --num_envs $NUM_ENVS \
        --rollout_length $ROLLOUT_LENGTH \
        --ou_sigma $OU_SIGMA \
        --headless \
        --output "${OUTPUT_DIR}/${motion}.pt" \
        2>&1 | tee "${OUTPUT_DIR}/${motion}.log"
    
    echo "[DONE] $motion"
done

echo ""
echo "========================================"
echo "ALL COMPLETE"
echo "========================================"
ls -lh $OUTPUT_DIR/*.pt
#!/bin/bash
# Ablation study: CSSM variants with varying depth and timesteps
# Records timing statistics and accuracy for each configuration

# Configuration
DATASET="${DATASET:-imagenette}"      # imagenette or pathfinder
PATHFINDER_DIFF="${PATHFINDER_DIFF:-9}"  # 9, 14, or 20 (only for pathfinder)
GPU="${GPU:-0}"                        # GPU ID
EPOCHS="${EPOCHS:-50}"                 # Training epochs
BATCH_SIZE="${BATCH_SIZE:-32}"         # Batch size
BASE_DIR="${BASE_DIR:-./ablation_results}"

# Experiment configurations
CSSM_TYPES=("opponent" "standard")
DEPTHS=(1 2 4 6 12)
SEQ_LENS=(8 16 32 48 96)

# Create results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${BASE_DIR}/${DATASET}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# Log file
LOG_FILE="${RESULTS_DIR}/ablation_log.txt"
SUMMARY_FILE="${RESULTS_DIR}/summary.csv"

# Initialize summary CSV
echo "cssm_type,depth,seq_len,final_train_acc,final_val_acc,avg_step_ms,throughput_samples_sec,total_params" > "$SUMMARY_FILE"

echo "============================================" | tee -a "$LOG_FILE"
echo "CSSM Ablation Study" | tee -a "$LOG_FILE"
echo "Dataset: $DATASET" | tee -a "$LOG_FILE"
echo "Epochs: $EPOCHS" | tee -a "$LOG_FILE"
echo "Batch Size: $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "GPU: $GPU" | tee -a "$LOG_FILE"
echo "Results: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

# Counter for progress
TOTAL_RUNS=$((${#CSSM_TYPES[@]} * ${#DEPTHS[@]} * ${#SEQ_LENS[@]}))
CURRENT_RUN=0

for CSSM_TYPE in "${CSSM_TYPES[@]}"; do
    for DEPTH in "${DEPTHS[@]}"; do
        for SEQ_LEN in "${SEQ_LENS[@]}"; do
            CURRENT_RUN=$((CURRENT_RUN + 1))
            RUN_NAME="${CSSM_TYPE}_d${DEPTH}_t${SEQ_LEN}"

            echo "" | tee -a "$LOG_FILE"
            echo "[$CURRENT_RUN/$TOTAL_RUNS] Running: $RUN_NAME" | tee -a "$LOG_FILE"
            echo "  CSSM: $CSSM_TYPE, Depth: $DEPTH, Seq Len: $SEQ_LEN" | tee -a "$LOG_FILE"

            # Build command
            CMD="CUDA_VISIBLE_DEVICES=$GPU python main.py \
                --arch vit \
                --cssm $CSSM_TYPE \
                --depth $DEPTH \
                --seq_len $SEQ_LEN \
                --dataset $DATASET \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --project cssm_ablation \
                --run_name $RUN_NAME"

            # Add pathfinder difficulty if applicable
            if [ "$DATASET" == "pathfinder" ]; then
                CMD="$CMD --pathfinder_difficulty $PATHFINDER_DIFF"
            fi

            # Run experiment and capture output
            RUN_LOG="${RESULTS_DIR}/${RUN_NAME}.log"
            echo "  Log: $RUN_LOG" | tee -a "$LOG_FILE"

            START_TIME=$(date +%s)
            eval "$CMD" 2>&1 | tee "$RUN_LOG"
            EXIT_CODE=${PIPESTATUS[0]}
            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))

            if [ $EXIT_CODE -eq 0 ]; then
                echo "  Status: SUCCESS (${DURATION}s)" | tee -a "$LOG_FILE"

                # Extract metrics from log
                FINAL_TRAIN_ACC=$(grep "Train Acc:" "$RUN_LOG" | tail -1 | grep -oP 'Train Acc: \K[0-9.]+')
                FINAL_VAL_ACC=$(grep "Val Acc:" "$RUN_LOG" | tail -1 | grep -oP 'Val Acc: \K[0-9.]+')
                AVG_STEP_MS=$(grep "Timing:" "$RUN_LOG" | tail -1 | grep -oP 'avg=\K[0-9.]+')
                THROUGHPUT=$(grep "Timing:" "$RUN_LOG" | tail -1 | grep -oP 'throughput=\K[0-9.]+')
                TOTAL_PARAMS=$(grep "Model parameters:" "$RUN_LOG" | grep -oP '[0-9,]+' | tr -d ',')

                # Append to summary
                echo "$CSSM_TYPE,$DEPTH,$SEQ_LEN,$FINAL_TRAIN_ACC,$FINAL_VAL_ACC,$AVG_STEP_MS,$THROUGHPUT,$TOTAL_PARAMS" >> "$SUMMARY_FILE"
            else
                echo "  Status: FAILED (exit code $EXIT_CODE)" | tee -a "$LOG_FILE"
                echo "$CSSM_TYPE,$DEPTH,$SEQ_LEN,FAILED,FAILED,FAILED,FAILED,FAILED" >> "$SUMMARY_FILE"
            fi
        done
    done
done

echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "Ablation Study Complete!" | tee -a "$LOG_FILE"
echo "Summary: $SUMMARY_FILE" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

# Print summary table
echo ""
echo "Results Summary:"
column -t -s',' "$SUMMARY_FILE"

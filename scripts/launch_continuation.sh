#!/bin/bash
# ============================================================================
# CONTINUATION MONITOR
# ============================================================================
# Monitors the tier runs and auto-launches post-tier experiments when each
# GPU becomes free. Designed to run as a nohup background process.
#
# Usage:
#   nohup scripts/launch_continuation.sh > logs/continuation_monitor.log 2>&1 &
# ============================================================================

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "Continuation monitor started: $(date)"
echo "Watching for tier completion..."

GPU0_LAUNCHED=false
GPU1_LAUNCHED=false

while true; do
    # Check if GPU 0 tier runs are done (tier1+tier2)
    if [ "$GPU0_LAUNCHED" = false ]; then
        # GPU 0 is done when its nohup process (running tier1→tier2) exits
        if ! ps aux | grep -v grep | grep "run_full_benchmark.*--tier [12].*--gpu 0" | grep -q python3; then
            # Double-check: tier 2 should have all files
            T2_COUNT=$(ls results/tier2_benchmark/*.json 2>/dev/null | wc -l)
            if [ "$T2_COUNT" -gt 5000 ]; then
                echo "GPU 0 free (tier2 has $T2_COUNT files): $(date)"
                echo "Launching post-tier runs on GPU 0..."
                nohup bash -c 'scripts/launch_post_tier_runs.sh gpu0' \
                    > logs/post_tier_gpu0.log 2>&1 &
                echo "GPU 0 post-tier PID: $!"
                GPU0_LAUNCHED=true
            fi
        fi
    fi

    # Check if GPU 1 tier runs are done (tier3+tier4)
    if [ "$GPU1_LAUNCHED" = false ]; then
        if ! ps aux | grep -v grep | grep "run_full_benchmark.*--tier [34].*--gpu 1" | grep -q python3; then
            T3_COUNT=$(ls results/tier3_benchmark/*.json 2>/dev/null | wc -l)
            T4_COUNT=$(ls results/tier4_benchmark/*.json 2>/dev/null | wc -l)
            if [ "$T3_COUNT" -gt 8000 ] && [ "$T4_COUNT" -gt 500 ]; then
                echo "GPU 1 free (tier3=$T3_COUNT, tier4=$T4_COUNT): $(date)"
                echo "Launching post-tier runs on GPU 1..."
                nohup bash -c 'scripts/launch_post_tier_runs.sh gpu1' \
                    > logs/post_tier_gpu1.log 2>&1 &
                echo "GPU 1 post-tier PID: $!"
                GPU1_LAUNCHED=true
            fi
        fi
    fi

    # Exit when both are launched
    if [ "$GPU0_LAUNCHED" = true ] && [ "$GPU1_LAUNCHED" = true ]; then
        echo "Both GPUs launched post-tier runs. Monitor exiting: $(date)"
        exit 0
    fi

    # Check every 5 minutes
    sleep 300
done

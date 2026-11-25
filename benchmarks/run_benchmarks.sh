#!/usr/bin/env bash
set -euo pipefail

ADB="${ADB:-/mnt/c/Users/Prajesh gogoi/AppData/Local/Android/Sdk/platform-tools/adb.exe}"
THREADS="${1:-4}"
ITERS="${2:-500}"
OUT_DIR="benchmarks/logs"
mkdir -p "$OUT_DIR"

echo "=== Run Metadata ===" | tee "$OUT_DIR/run_summary.txt"
echo "Date: $(date -Iseconds)" | tee -a "$OUT_DIR/run_summary.txt"
"$ADB" shell "getprop ro.product.model; dumpsys battery | grep -i temperature" | tee -a "$OUT_DIR/run_summary.txt"

echo -e "\n=== Total (Fused metal_plate_double_fp32.pte) ===" | tee -a "$OUT_DIR/run_summary.txt"
"$ADB" shell "cd /data/local/tmp/patchcore_test && ./app_csv /data/local/tmp/patchcore/metal_plate_double_fp32.pte $THREADS $ITERS total.csv" | tee -a "$OUT_DIR/run_summary.txt"

echo -e "\n=== Backbone (backbone_plate.pte) ===" | tee -a "$OUT_DIR/run_summary.txt"
"$ADB" shell "cd /data/local/tmp/patchcore_test && ./app_csv backbone_plate.pte $THREADS $ITERS backbone.csv" | tee -a "$OUT_DIR/run_summary.txt"

"$ADB" pull /data/local/tmp/patchcore_test/total.csv "$OUT_DIR/total_${ITERS}.csv"
"$ADB" pull /data/local/tmp/patchcore_test/backbone.csv "$OUT_DIR/backbone_${ITERS}.csv"
echo "Done. Logs saved to $OUT_DIR/"

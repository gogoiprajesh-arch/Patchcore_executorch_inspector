### Android App Demo (Live On-Device)

| OK Plate (`14.13`) | Scratches (`19.81`) | Partial Rust (`18.86`) | Total Rust + Llama (`19.97`) |
| :---: | :---: | :---: | :---: |
| ![OK](report_figures/app_demo/ok_plate.png) | ![Scratches](report_figures/app_demo/defect_scratches.png) | ![Partial Rust](report_figures/app_demo/defect_partial_rust.png) | ![Total Rust + Llama](report_figures/app_demo/defect_total_rust_llama.png) |



## On-Device Benchmarks

All latency and memory measurements were collected on-device (`arm64-v8a`, ExecuTorch 1.5.1, XNNPACK delegate, FP32, 4 threads) over **500 measured iterations** with the first **10 warm-up iterations discarded**.

| **Date** | `2026-11-24` |

### Raw Logs & Reproducibility

Raw per-iteration CSV logs (`total_500.csv`, `backbone_500.csv`) and the benchmark script are stored in [`benchmarks/`](benchmarks/).

**1. Prerequisites:** Download `metal_plate_double_fp32.pte`, `backbone_plate.pte`, and `app_csv` from the [Releases](https://github.com/gogoiprajesh-arch/Patchcore_executorch_inspector/releases) page, then push them to the device once via ADB:

```bash
adb shell mkdir -p /data/local/tmp/patchcore /data/local/tmp/patchcore_test
adb push metal_plate_double_fp32.pte /data/local/tmp/patchcore/
adb push backbone_plate.pte app_csv /data/local/tmp/patchcore_test/
adb shell chmod +x /data/local/tmp/patchcore_test/app_csv
```
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs("report_figures/failures", exist_ok=True)
os.makedirs("report_figures/good_detections", exist_ok=True)

data = torch.load("reference_plate.pt", weights_only=False)
scores = np.array([s.item() for s in data["scores"]])
labels = np.array(data["labels"])
names = data["file_names"]

thr = 15.631  # match your app threshold


def apply_postprocess_colormap(amap: np.ndarray, threshold: float):
    # Exact port of normalize_map() from postprocess.cpp
    norm = np.clip(amap / (2.0 * threshold), 0.0, 1.0)

    # Exact port of apply_colormap() from postprocess.cpp
    rgb = np.zeros((*norm.shape, 3), dtype=np.float32)

    m1 = norm < 0.333
    t1 = norm[m1] / 0.333
    rgb[m1, 1] = t1
    rgb[m1, 2] = 1.0 - t1

    m2 = (norm >= 0.333) & (norm < 0.666)
    t2 = (norm[m2] - 0.333) / 0.333
    rgb[m2, 0] = t2
    rgb[m2, 1] = 1.0

    m3 = norm >= 0.666
    t3 = np.clip((norm[m3] - 0.666) / 0.334, 0.0, 1.0)
    rgb[m3, 0] = 1.0
    rgb[m3, 1] = 1.0 - t3

    return norm, rgb


def save_fig(idx, out_path, tag):
    img = data["input_tensors"][idx][0].permute(1, 2, 0).numpy().clip(0, 1)
    amap = data["maps"][idx].squeeze().numpy()

    norm, cmap_rgb = apply_postprocess_colormap(amap, thr)

    # Keep clean background transparent; blend hot regions like the Android app
    alpha = np.clip((norm - 0.35) / 0.65, 0.0, 1.0) * 0.75
    blended = img * (1.0 - alpha[..., None]) + cmap_rgb * alpha[..., None]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))
    axes[0].imshow(img)
    axes[0].set_title(f"Input: {names[idx]}", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(blended)
    if scores[idx] >= thr:
        ys, xs = np.where(norm >= 0.5)
        if len(xs) > 0:
            rect = patches.Rectangle(
                (xs.min(), ys.min()), xs.max() - xs.min(), ys.max() - ys.min(),
                linewidth=1.5, edgecolor="red", facecolor="none"
            )
            axes[1].add_patch(rect)

    axes[1].set_title(f"{tag} | Score: {scores[idx]:.2f} (Thr: {thr:.2f})", fontsize=10)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# 1. Failures: real false alarms (good >= thr) + real misses (bad < thr)
fa = [i for i in range(len(scores)) if labels[i] == 0 and scores[i] >= thr]
fn = [i for i in range(len(scores)) if labels[i] == 1 and scores[i] < thr]
failures = sorted(fa, key=lambda i: -scores[i]) + sorted(fn, key=lambda i: scores[i])

# Pad with closest borderline cases if fewer than 5 real errors
if len(failures) < 5:
    borderline = sorted(
        [i for i in range(len(scores)) if i not in failures],
        key=lambda i: abs(scores[i] - thr)
    )
    failures += borderline[: 5 - len(failures)]

for rank, idx in enumerate(failures[:10], 1):
    kind = "False Alarm" if (labels[idx] == 0 and scores[idx] >= thr) else \
           "Miss" if (labels[idx] == 1 and scores[idx] < thr) else "Borderline"
    out = f"report_figures/failures/{rank:02d}_{kind.lower().replace(' ', '_')}.png"
    save_fig(idx, out, kind)

# 2. Good Detections: top-scoring defective sample from each defect type + overall top
seen_types = set()
good_dets = []
for idx in np.argsort(-scores):
    if labels[idx] == 1:
        dt = data["defect_type"][idx]
        if dt not in seen_types:
            seen_types.add(dt)
            good_dets.append(idx)
for idx in np.argsort(-scores):
    if labels[idx] == 1 and idx not in good_dets and len(good_dets) < 5:
        good_dets.append(idx)

for rank, idx in enumerate(good_dets, 1):
    dt = data["defect_type"][idx]
    out = f"report_figures/good_detections/{rank:02d}_{dt}.png"
    save_fig(idx, out, f"Detected ({dt})")

print(f"Saved {min(len(failures), 10)} failure/borderline figs and {len(good_dets)} good detection figs.")
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import functional as F
from sklearn.metrics import roc_auc_score

from Patchcore_wrapper import PatchCoreExport, pc, mean, std

TEST_DIR = Path("../Phase4_Pipeline/dataset/metal_plate/metal_plate/test")
OUT_PT = "reference_plate.pt"          
OUT_TXT = "reference_scores_plate.txt"
EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(d: Path):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS)


def best_threshold(good: np.ndarray, bad: np.ndarray):
    """Midpoint if the groups separate, otherwise best F1 (lowest threshold among ties,
    which favours catching defects over avoiding a false alarm)."""
    if good.max() < bad.min():
        return (good.max() + bad.min()) / 2, "midpoint of highest good and lowest defect"
    s = np.sort(np.unique(np.concatenate([good, bad])))
    cands = (s[:-1] + s[1:]) / 2

    def f1(t):
        tp = (bad >= t).sum()
        fp = (good >= t).sum()
        fn = (bad < t).sum()
        return 2 * tp / (2 * tp + fp + fn)

    best = max(f1(t) for t in cands)
    thr = min(t for t in cands if np.isclose(f1(t), best))
    return thr, "scores overlap: best F1, lowest threshold among ties"


def main():
    wrapper = PatchCoreExport(pc, mean, std).eval()

    type_dirs = sorted(d for d in TEST_DIR.iterdir() if d.is_dir())
    print("Test folders:", [d.name for d in type_dirs])

    data = {"input_tensors": [], "file_names": [], "labels": [],
            "scores": [], "maps": [], "defect_type": []}

    with open(OUT_TXT, "w") as log, torch.no_grad():
        for d in type_dirs:
            for img_path in list_images(d):
                img = Image.open(img_path).convert("RGB")
                t = F.to_tensor(img)
                t = F.resize(t, size=[256, 256],
                             interpolation=F.InterpolationMode.BILINEAR, antialias=True)
                batch = t.unsqueeze(0)
                score, amap = wrapper(batch)

                label = 0 if d.name == "good" else 1
                data["input_tensors"].append(batch)
                data["file_names"].append(f"{d.name}/{img_path.name}")
                data["labels"].append(label)
                data["scores"].append(score)
                data["maps"].append(amap)
                data["defect_type"].append(d.name)

                line = f"{d.name}/{img_path.name} | Label: {label} | Score: {score.item():.4f}"
                print(line)
                log.write(line + "\n")

    scores = np.array([s.item() for s in data["scores"]])
    labels = np.array(data["labels"])
    names = data["file_names"]
    good, bad = scores[labels == 0], scores[labels == 1]

    print("-" * 50)
    print(f"Images: {len(good)} good, {len(bad)} defective")
    print("ROC AUC:", roc_auc_score(labels, scores))
    print(f"Highest good: {good.max():.4f}   Lowest defect: {bad.min():.4f}")

    print("\nPer defect type (min / median / max):")
    for d in type_dirs:
        m = np.array([sc for sc, ty in zip(scores, data["defect_type"]) if ty == d.name])
        print(f"  {d.name:15s} n={len(m):3d}  {m.min():.2f} / {np.median(m):.2f} / {m.max():.2f}")

    thr, how = best_threshold(good, bad)
    print(f"\nThreshold: {thr:.4f}   ({how})")
    print(f"False alarms: {(good >= thr).sum()} of {len(good)}")
    print(f"Missed defects: {(bad < thr).sum()} of {len(bad)}")

    print("\nImages within 3% of the threshold (may flip on the phone):")
    for n, sc in zip(names, scores):
        if abs(sc - thr) / thr < 0.03:
            print(f"  {n}  {sc:.4f}")

    gi = np.where(labels == 0)[0]
    bi = np.where(labels == 1)[0]
    print("\nSuggested images to check on the phone vs Python:")
    print("  highest good  :", names[gi[np.argmax(scores[gi])]], f"{good.max():.4f}")
    print("  lowest defect :", names[bi[np.argmin(scores[bi])]], f"{bad.min():.4f}")
    print("  highest defect:", names[bi[np.argmax(scores[bi])]], f"{bad.max():.4f}")

    torch.save(data, OUT_PT)
    print(f"\nSaved {OUT_PT} and {OUT_TXT}")


if __name__ == "__main__":
    main()
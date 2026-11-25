import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from Patchcore_wrapper import PatchCoreExport, pc, mean, std

data = torch.load("reference_plate.pt", weights_only=False)
inputs = data["input_tensors"]
labels = np.array(data["labels"])

wrapper = PatchCoreExport(pc, mean, std).eval()
pc.eval()

orig_scores, fast_scores = [], []

with torch.no_grad():
    for batch in inputs:
        f_score, _ = wrapper(batch)
        fast_scores.append(f_score.item())

        ref = pc((batch - mean) / std)
        orig_scores.append(ref.pred_score.item())

orig_scores = np.array(orig_scores)
fast_scores = np.array(fast_scores)

print(f"Max score diff : {np.max(np.abs(orig_scores - fast_scores)):.2e}")
print(f"Original AUROC : {roc_auc_score(labels, orig_scores):.6f}")
print(f"nn.Linear AUROC: {roc_auc_score(labels, fast_scores):.6f}")
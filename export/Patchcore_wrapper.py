import torch
import torch.nn as nn
from anomalib.models import Patchcore


class PatchCoreExport(nn.Module):
    def __init__(self, pc_model, mean, std):
        super().__init__()
        self.pc = pc_model.eval()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        x = (x - self.mean) / self.std   # Normalize
        output_size = x.shape[-2:]

        features = self.pc.feature_extractor(x)
        features = {k: self.pc.feature_pooler(v) for k, v in features.items()}
        
        embedding = self.pc.generate_embedding(features)

        batch_size, C, height, width = embedding.shape    
     # Reshaping (B,C, H, W) to (B*H*W,C)
        embedding = self.pc.reshape_embedding(embedding)

       # apply nearest neighbor search
        patch_scores, locations = self.pc.nearest_neighbors(embedding=embedding, n_neighbors=1)

        # reshape to batch dimension
        patch_scores = patch_scores.reshape((batch_size, -1))
        locations = locations.reshape((batch_size, -1))

        # compute anomaly score
        pred_score = self.pc.compute_anomaly_score(patch_scores, locations, embedding)

        # reshape to w, h
        patch_scores = patch_scores.reshape((batch_size, 1, height, width))

        # get anomaly map
        anomaly_map = self.pc.anomaly_map_generator(patch_scores, output_size)

        return pred_score, anomaly_map

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


ckpt = "/home/prajesh_gogoi/my_projects/torch_projects/arm_challenge/results/Patchcore/MVTecAD/metal_plate/latest/weights/lightning/model.ckpt"

model = Patchcore.load_from_checkpoint(ckpt, backbone="resnet18", layers=("layer2", "layer3"), pre_trained=False)

pc = model.model.eval()


if __name__ == "__main__":
    print(pc.memory_bank.shape)
    print(pc.num_neighbors)

    wrapper = PatchCoreExport(pc, torch.zeros(1, 3, 1, 1), torch.ones(1,3,1,1)).eval()



    x = torch.rand(1, 3, 256, 256)
    ref = pc((x - mean) / std)
    score, amap = wrapper(x)   # wrapper normalizes internally

    # print(torch.allclose(score, ref.pred_score, atol=1e-5))
    # print((amap - ref.anomaly_map).abs().max())

    xn = (x - mean) / std

    # Check A: identity wrapper on already-normalized input -> should match exactly
    w_id = PatchCoreExport(pc, torch.zeros(1,3,1,1), torch.ones(1,3,1,1)).eval()
    s_a, m_a = w_id(xn)
    ref = pc(xn)
    print(torch.allclose(s_a, ref.pred_score), (m_a - ref.anomaly_map).abs().max())

    # Check B: real-normalization wrapper on raw input -> should also match
    w_real = PatchCoreExport(pc, mean, std).eval()
    s_b, m_b = w_real(x)
    print(torch.allclose(s_b, ref.pred_score), (m_b - ref.anomaly_map).abs().max())




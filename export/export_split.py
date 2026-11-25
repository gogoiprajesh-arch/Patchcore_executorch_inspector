import torch
from torch import nn
from Patchcore_wrapper import PatchCoreExport, pc, mean, std
from export_executorch import lower_and_save
import copy
import torch.nn.functional as F
from executorch.devtools.backend_debug import get_delegation_info

class PatchCoreBackbone(nn.Module):
    def __init__(self, pc, mean, std):
        super().__init__()
        self.pc = pc.eval()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        x = (x - self.mean) / self.std   # Normalize
  
        features = self.pc.feature_extractor(x)
        features = {k: self.pc.feature_pooler(v) for k, v in features.items()}
        
        embedding = self.pc.generate_embedding(features)
   
        # Reshaping (B,C, H, W) to (B*H*W,C)
        embedding = self.pc.reshape_embedding(embedding)

        return embedding.contiguous()
        

class PatchCoreHead(nn.Module):
    def __init__(self, pc, grid_h, grid_w, out_size):
        super().__init__()
        self.pc = pc.eval()
        self.gh, self.gw, self.out_size = grid_h, grid_w, out_size

    def forward(self, embedding):

        # apply nearest neighbor search
        patch_scores, locations = self.pc.nearest_neighbors(embedding=embedding, n_neighbors=1)

        # reshape to batch dimension
        patch_scores = patch_scores.reshape((1, -1))
        locations = locations.reshape((1, -1))

        # compute anomaly score
        pred_score = self.pc.compute_anomaly_score(patch_scores, locations, embedding)

        # reshape to w, h
        patch_scores = patch_scores.reshape((1, 1, self.gh, self.gw))

        # get anomaly map
        anomaly_map = self.pc.anomaly_map_generator(patch_scores, self.out_size)

        return pred_score, anomaly_map


class PatchCoreHeadFast(nn.Module):
    def __init__(self, pc, grid_h, grid_w, out_size):
        super().__init__()
        self.pc = pc.eval()
        self.gh, self.gw, self.out_size = grid_h, grid_w, out_size
        self.register_buffer("bank_sq", pc.memory_bank.pow(2).sum(-1).unsqueeze(0))   # (1, K)
        self.fc = nn.Linear(384, pc.memory_bank.shape[0], bias=False)       # K = memory_bank.shape[0]
        self.fc.weight = nn.Parameter(pc.memory_bank, requires_grad=False)

    def forward(self, embedding):
        dot = self.fc(embedding)            # (N, K) = embedding @ bank^T
        d2 = embedding.pow(2).sum(-1, keepdim=True) - 2 * dot + self.bank_sq
        min_d2, locations = d2.min(dim=1)                          # min over squared distances

        # compute anomaly score
        patch_scores = min_d2.clamp_min(0).sqrt().reshape(1, -1)
        locations = locations.reshape(1, -1)

        pred_score = self.pc.compute_anomaly_score(patch_scores, locations, embedding)

        # reshape to w, h
        patch_scores = patch_scores.reshape((1, 1, self.gh, self.gw))

        # get anomaly map
        anomaly_map = self.pc.anomaly_map_generator(patch_scores, self.out_size)

        return pred_score, anomaly_map

class PatchCoreHeadFastTest(nn.Module):
    def __init__(self, pc, grid_h, grid_w, out_size):
        super().__init__()
        bank = pc.memory_bank
        self.pc = copy.deepcopy(pc).eval()
        self.pc.memory_bank = torch.empty(0)
        self.gh, self.gw, self.out_size = grid_h, grid_w, out_size
        self.register_buffer("bank_sq", bank.pow(2).sum(-1).unsqueeze(0))
        self.fc = nn.Linear(384, bank.shape[0], bias=False)
        self.fc.weight = nn.Parameter(bank.clone(), requires_grad=False)

    def forward(self, embedding):
        dot = self.fc(embedding)            # (N, K) = embedding @ bank^T
        d2 = embedding.pow(2).sum(-1, keepdim=True) - 2 * dot + self.bank_sq
        min_d2, locations = d2.min(dim=1)                          # min over squared distances

        # compute anomaly score
        patch_scores = min_d2.clamp_min(0).sqrt().reshape(1, -1)
        locations = locations.reshape(1, -1)

        pred_score = self.pc.compute_anomaly_score(patch_scores, locations, embedding)

        # reshape to w, h
        patch_scores = patch_scores.reshape((1, 1, self.gh, self.gw))

        # get anomaly map
        anomaly_map = self.pc.anomaly_map_generator(patch_scores, self.out_size)

        return pred_score, anomaly_map

class PatchCoreHeadConv(nn.Module):
    def __init__(self, pc, grid_h, grid_w, out_size):
        super().__init__()
        self.pc = pc.eval()
        self.gh, self.gw, self.out_size = grid_h, grid_w, out_size
        K, C = pc.memory_bank.shape
        self.conv = nn.Conv2d(C, K, kernel_size=1, bias=False)
        self.conv.weight = nn.Parameter(pc.memory_bank.view(K, C, 1, 1), requires_grad=False)
        self.register_buffer("bank_sq", pc.memory_bank.pow(2).sum(-1).view(1, K, 1, 1))

    def forward(self, embedding):                       # (N, C) rows, N = gh*gw
        feat = embedding.reshape(1, self.gh, self.gw, -1).permute(0, 3, 1, 2).contiguous()  # (1, C, gh, gw)
        dot = self.conv(feat)                           # (1, K, gh, gw)
        e_sq = feat.pow(2).sum(1, keepdim=True)         # (1, 1, gh, gw)
        d2 = e_sq - 2 * dot + self.bank_sq
        min_d2, locations = d2.min(dim=1)               # (1, gh, gw) each
        patch_scores = min_d2.clamp_min(0).sqrt().reshape(1, -1)
        locations = locations.reshape(1, -1)

        # compute anomaly score
        pred_score = self.pc.compute_anomaly_score(patch_scores, locations, embedding)

        # reshape to w, h
        patch_scores = patch_scores.reshape((1, 1, self.gh, self.gw))

        # get anomaly map
        anomaly_map = self.pc.anomaly_map_generator(patch_scores, self.out_size)

        return pred_score, anomaly_map
        
class PatchCoreFull(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(1, 3, 256, 256)

    pc_w = copy.deepcopy(pc)
    pc_w.num_neighbors = 9

    pc_s = pc

    bb_pc = copy.deepcopy(pc)
    bb_pc.memory_bank = torch.empty(0)  

    backbone = PatchCoreBackbone(bb_pc, mean, std).eval()
    head = PatchCoreHead(pc, 32, 32, (256, 256)).eval()

    head_fast = PatchCoreHeadFast(pc_w, 32, 32, (256, 256)).eval()
    head_test = PatchCoreHeadFastTest(pc_s, 32, 32, (256, 256)).eval()

    Patchcore_singlebank = PatchCoreFull(backbone, head_test)
    Patchcore_doublebank = PatchCoreFull(backbone, head_fast)

    with torch.no_grad():
        score_1, anomaly_1 = PatchCoreExport(pc, mean, std).eval()(x)
        score_2, anomaly_2 = PatchCoreExport(pc_w, mean, std).eval()(x)

    print(score_1.shape, score_1.dtype)
    print(anomaly_1.shape,anomaly_1.dtype)
    with torch.no_grad():
        emb = backbone(x).detach()
    
    edge_double = lower_and_save(head_fast, (emb,), path = "head_metalfp32.pte")






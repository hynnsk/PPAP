from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

def tensor_correlation(a, b):
    return torch.einsum("nchw,ncij->nhwij", a, b)


def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def sample(t: torch.Tensor, coords: torch.Tensor):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)


@torch.jit.script
def super_perm(size: int, device: torch.device):
    perm = torch.randperm(size, device=device, dtype=torch.long)
    perm[perm == torch.arange(size, device=device)] += 1
    return perm % size

class EvalLoss(nn.Module):

    def __init__(self,
                 n_classes: int):
        super().__init__()

        self.n_classes = n_classes
        self.linear_loss = LinearLoss()

    def forward(self, model_input, linear_output: torch.Tensor() = None,
                cluster_output: torch.Tensor() = None) \
            -> Tuple[torch.Tensor, Dict[str, float]]:
        img, label = model_input

        linear_loss = self.linear_loss(linear_output, label, self.n_classes)
        cluster_loss = cluster_output[0]
        loss = linear_loss + cluster_loss
        loss_dict = {"loss": loss.item(), "linear": linear_loss.item(),
                     "cluster": cluster_loss.item()}

        return loss, loss_dict


class ContrastiveCorrelationLoss(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2

    def helper(self, f1, f2, c1, c2, shift):
        with torch.no_grad():
            # Comes straight from backbone which is currently frozen. this saves mem.
            fd = tensor_correlation(norm(f1), norm(f2))

            if self.cfg["pointwise"]:
                old_mean = fd.mean()
                fd -= fd.mean([3, 4], keepdim=True)
                fd = fd - fd.mean() + old_mean

        cd = tensor_correlation(norm(c1), norm(c2))

        if self.cfg["zero_clamp"]:
            min_val = 0.0
        else:
            min_val = -9999.0

        if self.cfg["stabilize"]:
            loss = - cd.clamp(min_val, .8) * (fd - shift)
        else:
            loss = - cd.clamp(min_val) * (fd - shift)

        return loss, cd

    def forward(self,
                orig_feats: torch.Tensor,
                orig_feats_pos: torch.Tensor,
                orig_code: torch.Tensor,
                orig_code_pos: torch.Tensor,
                ):

        coord_shape = [orig_feats.shape[0], self.cfg["feature_samples"], self.cfg["feature_samples"], 2]

        coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
        coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1

        feats = sample(orig_feats, coords1)
        code = sample(orig_code, coords1)

        feats_pos = sample(orig_feats_pos, coords2)
        code_pos = sample(orig_code_pos, coords2)

        pos_intra_loss, pos_intra_cd = self.helper(
            feats, feats, code, code, self.cfg["corr_loss"]["pos_intra_shift"])
        pos_inter_loss, pos_inter_cd = self.helper(
            feats, feats_pos, code, code_pos, self.cfg["corr_loss"]["pos_inter_shift"])

        neg_losses = []
        neg_cds = []
        for i in range(self.cfg["neg_samples"]):
            perm_neg = super_perm(orig_feats.shape[0], orig_feats.device)
            feats_neg = sample(orig_feats[perm_neg], coords2)
            code_neg = sample(orig_code[perm_neg], coords2)
            neg_inter_loss, neg_inter_cd = self.helper(
                feats, feats_neg, code, code_neg, self.cfg["corr_loss"]["neg_inter_shift"])
            neg_losses.append(neg_inter_loss)
            neg_cds.append(neg_inter_cd)

        neg_inter_loss = torch.cat(neg_losses, axis=0)
        neg_inter_cd = torch.cat(neg_cds, axis=0)

        return (self.cfg["corr_loss"]["pos_intra_weight"] * pos_intra_loss.mean() +
                self.cfg["corr_loss"]["pos_inter_weight"] * pos_inter_loss.mean() +
                self.cfg["corr_loss"]["neg_inter_weight"] * neg_inter_loss.mean(),
                {"self_loss": pos_intra_loss.mean().item(),
                 "knn_loss": pos_inter_loss.mean().item(),
                 "rand_loss": neg_inter_loss.mean().item()}
                )


class LinearLoss(nn.Module):

    def __init__(self):
        super(LinearLoss, self).__init__()
        self.linear_loss = nn.CrossEntropyLoss()

    def forward(self, linear_logits: torch.Tensor, label: torch.Tensor, n_classes: int):
        flat_label = label.reshape(-1)
        mask = (flat_label >= 0) & (flat_label < n_classes)

        linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
        linear_logits = linear_logits.permute(0, 2, 3, 1).reshape(-1, n_classes)
        linear_loss = self.linear_loss(linear_logits[mask], flat_label[mask]).mean()

        return linear_loss

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    From: https://github.com/HobbitLong/SupContrast"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, opt=None):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.cossim = nn.CosineSimilarity()

        self.initialcrit_pos = opt["initialcrit_pos"] # Phi
        self.initialcrit_amb = opt["initialcrit_amb"] # Psi
        self.prop_iter = opt["prop_iter"]
        self.sigma_pos = opt["sigma_pos"]
        self.sigma_amb = opt["sigma_amb"]
        self.trainpatchsplit = opt["trainpatchsplit"]

        self.split = 784 # for mini-iters


    def forward(self, z, feat):
        device = z.device

        # ------------ #
        dim = feat.size(1)
        feat = feat.permute(0,2,3,1).reshape(-1,dim)
        feat = F.normalize(feat, dim=-1)

        batch_size = z.shape[0]

        mini_iters = int(batch_size / self.split)
        loss = torch.tensor(0).to(device)

        logits_mask_split = torch.scatter(
            torch.ones((self.split, batch_size), device=torch.device('cuda:0'), dtype=torch.float16), 1,
            torch.arange(self.split, device=torch.device('cuda:0')).view(-1, 1), 0)

        pos_num_all=0
        neg_num_all=0

        for mi in range(mini_iters):

            sampling_idx = torch.randperm(self.split)[:int(self.split/self.trainpatchsplit)]

            logits_mask_split_sample = logits_mask_split[sampling_idx]

            feat_split = feat[mi * self.split: (mi + 1) * self.split]
            feat_split = feat_split[sampling_idx]

            with torch.cuda.amp.autocast(enabled=True):
                feat_cossim_split = torch.matmul(feat_split, feat.transpose(0, 1))

            mask_one = (self.initialcrit_pos < feat_cossim_split).type(torch.float16)

            mask_new = mask_one.clone().detach()
            proxy_before = feat_split.clone().detach()
            Psi = torch.ones((feat_split.size(0), 1), dtype=torch.float16, device=torch.device('cuda:0')) * self.initialcrit_amb
            Phi = torch.ones((feat_split.size(0), 1), dtype=torch.float16, device=torch.device('cuda:0')) * self.initialcrit_pos
            output_cossim_proxy = feat_cossim_split.clone().detach()

            for _prop in range(self.prop_iter):

                proxy = feat.unsqueeze(0) * mask_new.unsqueeze(-1)
                proxy = torch.mean(proxy, dim=1)
                proxy = F.normalize(proxy, dim=1)

                with torch.cuda.amp.autocast(enabled=True):
                    output_cossim_proxy = torch.matmul(proxy, feat.transpose(0, 1))

                moving_sim = self.cossim(proxy_before, proxy)
                Psi = Psi + ((1. - moving_sim.unsqueeze(-1)) / self.sigma_amb)
                Phi = Phi - ((1. - moving_sim.unsqueeze(-1)) / self.sigma_pos)
                mask_new = (Phi < output_cossim_proxy).type(torch.float16).clone().detach()
                proxy_before = proxy.clone().detach()

            neglect_base = torch.tensor((Psi > output_cossim_proxy), dtype=torch.float16)

            mask = mask_new * logits_mask_split_sample
            neglect_mask = torch.logical_or(mask, neglect_base).type(torch.float16)

            neglect_logits_mask = neglect_mask * logits_mask_split_sample

            modeloutput_z_split = z[mi * self.split: (mi + 1) * self.split]
            modeloutput_z_split = modeloutput_z_split[sampling_idx]

            with torch.cuda.amp.autocast(enabled=True):
                anchor_dot_contrast_split = torch.div(
                    torch.matmul(modeloutput_z_split, z.T),
                    self.temperature)

            logits_max_split, _ = torch.max(anchor_dot_contrast_split, dim=1, keepdim=True)
            logits_split = anchor_dot_contrast_split - logits_max_split.detach()

            exp_logits_split_neg = torch.exp(logits_split) * neglect_logits_mask
            log_prob_split_neg = logits_split - torch.log(exp_logits_split_neg.sum(1, keepdim=True))

            mask = mask.type(torch.float32)
            log_prob_split_neg = log_prob_split_neg.type(torch.float32)

            nonzero_idx = torch.where(mask.sum(1) != 0.)

            mean_log_prob = (mask[nonzero_idx] * log_prob_split_neg[nonzero_idx]).sum(1) / (mask[nonzero_idx].sum(1))
            loss = loss - torch.mean((self.temperature / self.base_temperature) * mean_log_prob)

            logits_mask_split = torch.roll(logits_mask_split, self.split, dims=1)

            pos_num_all += (torch.sum(mask.type(torch.float32)) / mask.size(0))
            neg_num_all += (torch.sum(neglect_base.type(torch.float32)) / neglect_base.size(0))

        loss = loss / mini_iters
        pos_num = pos_num_all / mini_iters
        neg_num = neg_num_all / mini_iters

        return loss, pos_num, neg_num

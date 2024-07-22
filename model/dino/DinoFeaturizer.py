import torch
import torch.nn as nn
import model.dino.vision_transformer as vits
import torch.nn.functional as F
import copy

class DinoFeaturizer(nn.Module):

    def __init__(self, dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        patch_size = self.cfg["pretrained"]["dino_patch_size"]
        self.patch_size = patch_size
        self.feat_type = self.cfg["pretrained"]["dino_feat_type"]
        arch = self.cfg["pretrained"]["model_type"]
        self.model = vits.__dict__[arch](
            patch_size=patch_size,
            num_classes=0)
        self.n_feats = self.cfg["dim"]

        self.dropout = torch.nn.Dropout2d(p=.1)

        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError("Unknown arch and patch size")

        if cfg["pretrained"]["pretrained_weights"] is not None:
            state_dict = torch.load(cfg["pretrained"]["pretrained_weights"], map_location="cpu")
            state_dict = state_dict["teacher"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

            msg = self.model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(
                cfg["pretrained"]["pretrained_weights"], msg))
        else:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            self.model.load_state_dict(state_dict, strict=True)

        # set last block as decoder
        for i, blk in enumerate(self.model.blocks):
            if len(self.model.blocks) - i == 1:
                self.decoder = copy.deepcopy(blk)

        for p in self.model.parameters():
            p.requires_grad = False

        for name, m in self.decoder.named_parameters():
            m.requires_grad = True

        self.project_head = nn.Linear(self.dim, self.dim)

    def forward(self, img):

        assert (img.shape[2] % self.patch_size == 0)
        assert (img.shape[3] % self.patch_size == 0)

        with torch.no_grad():
            crit_feat, midfeat = self.model.get_midfeat(img, mid_n=2, last_n=1)

        image_feat = self.decoder(midfeat)
        image_feat = self.model.norm(image_feat)

        feat_h = img.shape[2] // self.patch_size
        feat_w = img.shape[3] // self.patch_size

        image_feat = image_feat[:, 1:, :].reshape(image_feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)

        crit_feat = crit_feat[:, 1:, :].reshape(crit_feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)

        code = image_feat

        z = code.permute(0, 2, 3, 1).reshape(-1, self.dim)
        z = self.project_head(z)
        z = F.normalize(z, dim=1)

        return self.dropout(image_feat), code, z, self.dropout(crit_feat)

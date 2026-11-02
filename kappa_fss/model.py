import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from .modules import KTP, build_prototypes, cosine_prior_maps

class KAPPAFSS(nn.Module):
    def __init__(self, config):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        for parameter in self.stem.parameters():
            parameter.requires_grad = False
        for layer in (self.layer1, self.layer2, self.layer3):
            for parameter in layer.parameters():
                parameter.requires_grad = False
        self.ktp = KTP(
            in_channels=1024,
            reduced=config.feature_dim,
            n_angles=config.ktp_angles,
            lengths=config.ktp_lengths,
            width=config.ktp_width,
            tau_init=config.ktp_tau,
        )
        self.boundary_width = config.assouad_band_px
        self.decoder = nn.Sequential(
            nn.Conv2d(1028, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

    def train(self, mode=True):
        super().train(mode)
        self.stem.eval()
        self.layer1.eval()
        self.layer2.eval()
        self.layer3.eval()
        return self

    def encode(self, image):
        image = self.stem(image)
        image = self.layer1(image)
        image = self.layer2(image)
        return self.layer3(image)

    def forward(self, support_images, support_masks, query_image):
        support_features = []
        for image in support_images:
            features = self.encode(image.unsqueeze(0))
            features, _ = self.ktp(features)
            support_features.append(features[0])
        query_features = self.encode(query_image.unsqueeze(0))
        query_features, coherence = self.ktp(query_features)
        query_features = query_features[0]
        prototypes = build_prototypes(
            support_features,
            support_masks,
            boundary_width=self.boundary_width,
        )
        priors = cosine_prior_maps(query_features, prototypes).unsqueeze(0)
        decoder_input = torch.cat([query_features.unsqueeze(0), priors], dim=1)
        logits = self.decoder(decoder_input)
        logits = F.interpolate(logits, size=query_image.shape[-2:], mode="bilinear", align_corners=False)
        return {"logits": logits, "coherence": coherence}

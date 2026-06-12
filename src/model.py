import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models.feature_extraction import create_feature_extractor

class SSHHead(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.conv_context = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.cls_head = nn.Conv2d(out_channels, 2, kernel_size=1)
        self.box_head = nn.Conv2d(out_channels, 4, kernel_size=1)
        self.kps_head = nn.Conv2d(out_channels, 10, kernel_size=1)

    def forward(self, x):
        x = self.conv_context(x)
        return self.cls_head(x), self.box_head(x), self.kps_head(x)


class LightweightFaceDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        base_model = mobilenet_v2(weights=weights)
        
        self.return_nodes = {
            'features.6': 'feat_small',
            'features.13': 'feat_medium',
            'features.18': 'feat_large',
        }
        
        self.backbone = create_feature_extractor(base_model, return_nodes=self.return_nodes)
        
        self.ssh_small = SSHHead(in_channels=32)
        self.ssh_medium = SSHHead(in_channels=96)
        self.ssh_large = SSHHead(in_channels=320)

    def forward(self, x):
        features = self.backbone(x)
        
        cls_s, box_s, kps_s = self.ssh_small(features['feat_small'])
        cls_m, box_m, kps_m = self.ssh_medium(features['feat_medium'])
        cls_l, box_l, kps_l = self.ssh_large(features['feat_large'])
        
        return {
            "cls": [cls_s, cls_m, cls_l],
            "box": [box_s, box_m, box_l],
            "kps": [kps_s, kps_m, kps_l],
        }

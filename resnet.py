import torch
import torch.nn as nn
import torchvision.models as models


def get_resnet(depth=50, width_multiplier=1):
    """
    Returns ResNet backbone with projection head removed.
    Supported depths: 18, 34, 50, 101, 152
    """
    resnet_map = {
        18:  models.resnet18,
        34:  models.resnet34,
        50:  models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }

    if depth not in resnet_map:
        raise ValueError(f"Unsupported ResNet depth: {depth}. "
                         f"Choose from {list(resnet_map.keys())}")

    # Load architecture (no pretrained weights)
    backbone = resnet_map[depth](weights=None)

    # Get output feature dimension before FC layer
    in_dim = backbone.fc.in_features

    # Remove the final FC layer — we add our own projection head
    backbone.fc = nn.Identity()

    return backbone, in_dim


class ResNetSimCLR(nn.Module):
    """
    ResNet backbone + SimCLR projection head.
    Mirrors original SimCLR: base_model -> projection_head
    """
    def __init__(self, depth=50, proj_out_dim=128,
                 num_proj_layers=3, width_multiplier=1):
        super(ResNetSimCLR, self).__init__()

        from model_util import ProjectionHead

        self.backbone, in_dim = get_resnet(depth, width_multiplier)
        self.projection_head = ProjectionHead(
            in_dim=in_dim,
            mid_dim=in_dim,
            out_dim=proj_out_dim,
            num_layers=num_proj_layers
        )

    def forward(self, x):
        # x shape: (batch, C, H, W)
        h = self.backbone(x)          # (batch, in_dim)
        z = self.projection_head(h)   # (batch, proj_out_dim)
        return h, z


if __name__ == '__main__':
    # Quick test
    model = ResNetSimCLR(depth=18, proj_out_dim=128)
    x = torch.randn(4, 3, 224, 224)
    h, z = model(x)
    print(f"Backbone output (h): {h.shape}")    # (4, 512)
    print(f"Projection output (z): {z.shape}")  # (4, 128)
    print("resnet.py OK!")

import torch.nn as nn

from .models import aggregators, backbones


class DinoSalad(nn.Module):

    def __init__(
        self,
        dinov2_ckpt_path=None,
    ):
        super().__init__()

        backbone_config = {
            'num_trainable_blocks': 4,
            'return_token': True,
            'norm_layer': True,
            'dino_v2_weights': dinov2_ckpt_path
        }
        agg_config = {
            'num_channels': 768,
            'num_clusters': 64,
            'cluster_dim': 128,
            'token_dim': 256,
        }
        self.backbone = backbones.DINOv2(model_name='dinov2_vitb14',
                                         **backbone_config)
        self.aggregator = aggregators.SALAD(**agg_config)

        print('DinoSalad init finished')

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x

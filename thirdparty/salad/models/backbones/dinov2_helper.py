import os

import torch
import torch.nn as nn

DINOV2_ARCHS = {
    # 'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    # 'dinov2_vitl14': 1024,
    # 'dinov2_vitg14': 1536,
}


def _make_dinov2_model():
    from .dinov2.models import vision_transformer as vits

    arch_name = 'vit_base'
    vit_kwargs = dict(
        img_size=518,
        patch_size=14,
        init_values=1.0,
        ffn_layer='mlp',
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    )
    model = vits.__dict__[arch_name](**vit_kwargs)

    return model


class DINOv2(nn.Module):
    """
    DINOv2 model

    Args:
        model_name (str): The name of the model architecture
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_trainable_blocks (int): The number of last blocks in the model that are trainable.
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """

    def __init__(self,
                 model_name='dinov2_vitb14',
                 num_trainable_blocks=2,
                 norm_layer=False,
                 return_token=False,
                 dino_v2_weights=None):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(
        ), f'Unknown model name {model_name}'
        if (dino_v2_weights is not None and os.path.exists(dino_v2_weights)):
            self.model = _make_dinov2_model()
            self.model.load_state_dict(torch.load(dino_v2_weights))
        else:
            self.model = torch.hub.load('facebookresearch/dinov2',
                                        model_name,
                                        pretrained=True)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

    def forward(self, x):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """

        B, C, H, W = x.shape

        x = self.model.prepare_tokens_with_masks(x)

        # First blocks are frozen
        with torch.no_grad():
            for blk in self.model.blocks[:-self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()

        # Last blocks are trained
        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)

        t = x[:, 0]
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        f = f.reshape(
            (B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)

        if self.return_token:
            return f, t
        return f

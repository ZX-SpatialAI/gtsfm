import torch

model = torch.hub.load("serizba/salad", "dinov2_salad")
model.eval()
# model.cuda()

# dinov2_vitg14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

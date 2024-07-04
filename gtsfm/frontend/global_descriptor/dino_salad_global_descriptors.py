import os

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from gtsfm.common.image import Image
from gtsfm.frontend.global_descriptor.global_descriptor_base import GlobalDescriptorBase
from thirdparty.salad.dino_salad import DinoSalad

WEIGHTS_DIR = '/root/autodl-tmp/weights'


class MonoDataset(Dataset):

    def __init__(self, images, input_transform=None):
        self.input_transform = input_transform
        self.images = images

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)


def input_transform(image_size=None):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    if image_size:
        return T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        return T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])


class DinoSaladGlobalDescriptor(GlobalDescriptorBase):
    """DinoSalad global descriptor"""

    def __init__(self, image_size=(1288, 728)) -> None:
        """ """
        self._model = self._load_model()
        self.image_transform = input_transform(image_size)

    @torch.no_grad
    def _load_model(self):
        dinov2_ckpt_path = os.path.join(WEIGHTS_DIR,
                                        'dinov2_vitb14_pretrain.pth')
        dino_salad_ckpt_path = os.path.join(WEIGHTS_DIR, 'dino_salad.ckpt')
        model = DinoSalad(dinov2_ckpt_path)
        model.load_state_dict(torch.load(dino_salad_ckpt_path))
        model = model.eval()
        model = model.to('cuda')

        return model

    @torch.no_grad
    def describe(self, image: Image) -> np.ndarray:
        descriptor = self._model(
            torch.from_numpy(self.image_transfor(image.value_array)))

        return descriptor.detach().cpu().numpy()

    @torch.no_grad
    def describe(self, image_files) -> np.ndarray:
        dataset = MonoDataset(image_files, self.image_transform)
        dataloader = DataLoader(dataset,
                                num_workers=4,
                                batch_size=4,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False)

        descriptors = []
        device = 'cuda'
        with torch.autocast(device_type=device, dtype=torch.float16):
            for (imgs, _) in tqdm(dataloader):
                output = self.model(imgs.to(device)).cpu()
                descriptors.append(output)

        descriptors = torch.cat(descriptors)

        return descriptors.detach().cpu().numpy()

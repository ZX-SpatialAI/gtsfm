import argparse

import dino_salad
import numpy as np
import torch
import torchvision.transforms as T
from dataloaders.val.MonoDataset import MonoDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# from utils.validation import find_image_pairs
# from vpr_model import VPRModel

# from utils.validation import get_validation_recalls

# VAL_DATASETS = ['MSLS', 'MSLS_Test', 'pitts30k_test', 'pitts250k_test', 'Nordland', 'SPED']
VAL_DATASETS = ['MONO']


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


def get_val_dataset(dataset_name, image_size=None):
    dataset_name = dataset_name.lower()
    transform = input_transform(image_size=image_size)

    # if 'nordland' in dataset_name:
    #     ds = NordlandDataset(input_transform=transform)

    # elif 'msls_test' in dataset_name:
    #     ds = MSLSTest(input_transform=transform)

    # elif 'msls' in dataset_name:
    #     ds = MSLS(input_transform=transform)

    # elif 'pitts' in dataset_name:
    #     ds = PittsburghDataset(which_ds=dataset_name, input_transform=transform)

    # elif 'sped' in dataset_name:
    #     ds = SPEDDataset(input_transform=transform)
    if 'mono' in dataset_name:
        image_dir = '/root/autodl-tmp/teastone/images'
        ds = MonoDataset(image_dir, input_transform=transform)
    else:
        raise ValueError

    num_references = ds.num_references
    num_queries = ds.num_queries
    ground_truth = ds.ground_truth
    return ds, num_references, num_queries, ground_truth


def get_descriptors(model, dataloader, device):
    descriptors = []
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for (imgs, _) in tqdm(dataloader):
                output = model(imgs.to(device)).cpu()
                descriptors.append(output)

    return torch.cat(descriptors)


def load_model(ckpt_path):
    model = dino_salad.DinoSalad(
        dinov2_ckpt_path='/root/autodl-tmp/weights/dinov2_vitb14_pretrain.pth')

    print(f'Load weights form {ckpt_path}')
    model.load_state_dict(torch.load(ckpt_path))
    print('Weights loaded')
    model = model.eval()
    print('DinoSalad eval finished')
    model = model.to('cuda')
    print(f"Loaded model from {ckpt_path} Successfully!")
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval VPR model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Model parameters
    parser.add_argument("--ckpt_path",
                        type=str,
                        required=True,
                        default=None,
                        help="Path to the checkpoint")

    # Datasets parameters
    parser.add_argument(
        '--val_datasets',
        nargs='+',
        default=VAL_DATASETS,
        help='Validation datasets to use',
        choices=VAL_DATASETS,
    )
    parser.add_argument('--image_size',
                        nargs='*',
                        default=None,
                        help='Image size (int, tuple or None)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')

    args = parser.parse_args()

    # Parse image size
    if args.image_size:
        if len(args.image_size) == 1:
            args.image_size = (args.image_size[0], args.image_size[0])
        elif len(args.image_size) == 2:
            args.image_size = tuple(args.image_size)
        else:
            raise ValueError('Invalid image size, must be int, tuple or None')

        args.image_size = tuple(map(int, args.image_size))
    else:
        args.image_size = (1288, 728)

    return args


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True

    args = parse_args()

    model = load_model(args.ckpt_path)

    for val_name in args.val_datasets:
        val_dataset, num_references, num_queries, ground_truth = get_val_dataset(
            val_name, args.image_size)
        val_loader = DataLoader(val_dataset,
                                num_workers=16,
                                batch_size=args.batch_size,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False)

        print(f'Evaluating on {val_name}')
        descriptors = get_descriptors(model, val_loader, 'cuda')
        np.save("descriptors.npy", descriptors.detach().cpu().numpy())
        # find_image_pairs(descriptors)

        # print(f'Descriptor dimension {descriptors.shape[1]}')
        # r_list = descriptors[ : num_references]
        # q_list = descriptors[num_references : ]

        # print('total_size', descriptors.shape[0], num_queries + num_references)

        # testing = isinstance(val_dataset, MSLSTest)

        # preds = get_validation_recalls(
        #     r_list=r_list,
        #     q_list=q_list,
        #     k_values=[1, 5, 10, 15, 20, 25],
        #     gt=ground_truth,
        #     print_results=True,
        #     dataset_name=val_name,
        #     faiss_gpu=False,
        #     testing=testing,
        # )

        # if testing:
        #     val_dataset.save_predictions(preds, args.ckpt_path + '.' + model.agg_arch + '.preds.txt')

        del descriptors
        print('========> DONE!\n\n')

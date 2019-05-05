import imageio
import numpy as np
from argparse import ArgumentParser

import torch

from trainer import Trainer
from utils.tools import get_config

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--model-path', default='', type=str,
                    help='Path to save model')
args = parser.parse_args()


def main():
    config = get_config(args.config)
    if config['cuda']:
        device = torch.device("cuda:{}".format(config['gpu_ids'][0]))
    else:
        device = torch.device("cpu")
    trainer = Trainer(config)
    trainer.load_state_dict(load_weights(args.model_path, device), strict=False)
    trainer.eval()

    image = imageio.imread(args.image)
    image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0).cuda()
    mask = imageio.imread(args.mask)
    mask = (torch.FloatTensor(mask[:, :, 0]) / 255).unsqueeze(0).unsqueeze(0).cuda()

    x = (image / 127.5 - 1) * (1 - mask).cuda()
    with torch.no_grad():
        _, result, _ = trainer.netG(x, mask)

    imageio.imwrite(args.output, upcast(result[0].permute(1, 2, 0).detach().cpu().numpy()))


def load_weights(path, device):
    model_weights = torch.load(path)
    return {
        k: v.to(device)
        for k, v in model_weights.items()
    }


def upcast(x):
    return np.clip((x + 1) * 127.5 , 0, 255).astype(np.uint8)


if __name__ == '__main__':
    main()

import math

import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision.transforms.functional import rotate


def scale_features(features, scale_factor):
    return F.interpolate(features, scale_factor=scale_factor, mode='bilinear', align_corners=True)


def descale_features(features, orig_size):
    return F.interpolate(features, size=orig_size, mode='bilinear', align_corners=True)


def crop_features(features, start_hh, start_ww, size):
    cont = torch.zeros_like(features)
    cont[..., start_hh:start_hh + size, start_ww:start_ww + size] = features[..., start_hh:start_hh + size,
                                                                    start_ww:start_ww + size]
    return cont[..., start_hh:start_hh + size, start_ww:start_ww + size]


def gaussian_noise(feature, mean=0, variance=1, amplitude=1):
    device = feature.device
    nn, cc, hh, ww = feature.shape
    feature = feature.cpu().detach().numpy()
    noise = amplitude * np.random.normal(loc=mean, scale=variance, size=(1, hh, ww))
    noise = np.repeat(noise, nn*cc, axis=0).reshape((nn, cc, hh, ww))
    return torch.from_numpy(feature + noise).to(device)


def rotate_features(feature, degree=10):
    return rotate(feature, degree)


def flip_features(feature):
    return feature[..., ::-1]


def v_flip_features(feature):
    return feature[..., ::-1, :]


def tile_features(features, num_pieces):
    _, _, h, w = features.size()

    num_pieces_per_line = int(math.sqrt(num_pieces))

    h_per_patch = h // num_pieces_per_line
    w_per_patch = w // num_pieces_per_line

    """
    +-----+-----+
    |  1  |  2  |
    +-----+-----+
    |  3  |  4  |
    +-----+-----+

    +-----+-----+-----+-----+
    |  1  |  2  |  3  |  4  |
    +-----+-----+-----+-----+
    """
    patches = []
    for splitted_features in torch.split(features, h_per_patch, dim=2):
        for patch in torch.split(splitted_features, w_per_patch, dim=3):
            patches.append(patch)

    return torch.cat(patches, dim=0)


def merge_features(features, num_pieces, batch_size):
    """
    +-----+-----+-----+-----+
    |  1  |  2  |  3  |  4  |
    +-----+-----+-----+-----+
    
    +-----+-----+
    |  1  |  2  |
    +-----+-----+
    |  3  |  4  |
    +-----+-----+
    """
    features_list = list(torch.split(features, batch_size))
    num_pieces_per_line = int(math.sqrt(num_pieces))

    index = 0
    ext_h_list = []

    for _ in range(num_pieces_per_line):

        ext_w_list = []
        for _ in range(num_pieces_per_line):
            ext_w_list.append(features_list[index])
            index += 1

        ext_h_list.append(torch.cat(ext_w_list, dim=3))

    features = torch.cat(ext_h_list, dim=2)
    return features


def puzzle_module(x, func_list, num_pieces):
    tiled_x = tile_features(x, num_pieces)

    for func in func_list:
        tiled_x = func(tiled_x)

    merged_x = merge_features(tiled_x, num_pieces, x.size()[0])
    return merged_x

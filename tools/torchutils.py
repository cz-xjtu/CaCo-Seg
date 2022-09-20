import torch
from torch.utils.data import Dataset
from PIL import Image
import os.path
import random
import numpy as np
from tools import imutils
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_model(model, model_path, parallel=False):
    if parallel:
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)


def make_cam(x, epsilon=1e-5):
    # relu(x) = max(x, 0)
    x = F.relu(x)

    b, c, h, w = x.size()

    flat_x = x.view(b, c, (h * w))
    max_value = flat_x.max(dim=-1)[0].view((b, c, 1, 1))

    return F.relu(x - epsilon) / (max_value + epsilon)


def gap2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)

    return out


class Normalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        if imgarr.shape[-1] != 3:
            imgarr = imgarr[:, :, np.newaxis].repeat([3], axis=2)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img


class NormalizeForSegmentation():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        image, mask = data['image'], data['mask']

        image = np.asarray(image, dtype=np.float32)
        if image.shape[-1] != 3:
            image = image[:, :, np.newaxis].repeat([3], axis=2)
        mask = np.asarray(mask, dtype=np.int64)

        norm_image = np.empty_like(image, np.float32)

        norm_image[..., 0] = (image[..., 0] / 255. - self.mean[0]) / self.std[0]
        norm_image[..., 1] = (image[..., 1] / 255. - self.mean[1]) / self.std[1]
        norm_image[..., 2] = (image[..., 2] / 255. - self.mean[2]) / self.std[2]

        data['image'] = norm_image
        data['mask'] = mask

        return data


class TransposeForSegmentation:
    def __init__(self):
        pass

    def __call__(self, data):
        # h, w, c -> c, h, w
        data['image'] = data['image'].transpose((2, 0, 1))
        return data


class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1


class BatchNorm2dFixed(torch.nn.Module):

    def __init__(self, num_features, eps=1e-5):
        super(BatchNorm2dFixed, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.Tensor(num_features))
        self.bias = torch.nn.Parameter(torch.Tensor(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, input):
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            False, eps=self.eps)

    def __call__(self, x):
        return self.forward(x)


class SegmentationDataset(Dataset):
    def __init__(self, img_name_list_path, img_dir, label_dir, rescale=None, flip=False, cropsize=None,
                 img_transform=None, mask_transform=None):
        self.img_name_list_path = img_name_list_path
        self.img_dir = img_dir
        self.label_dir = label_dir

        self.img_transform = img_transform
        self.mask_transform = mask_transform

        self.img_name_list = open(self.img_name_list_path).read().splitlines()

        self.rescale = rescale
        self.flip = flip
        self.cropsize = cropsize

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):

        name = self.img_name_list[idx]

        img = Image.open(os.path.join(self.img_dir, name + '.jpg')).convert("RGB")
        mask = Image.open(os.path.join(self.label_dir, name + '.png'))

        if self.rescale is not None:
            s = self.rescale[0] + random.random() * (self.rescale[1] - self.rescale[0])
            adj_size = (round(img.size[0] * s / 8) * 8, round(img.size[1] * s / 8) * 8)
            img = img.resize(adj_size, resample=Image.CUBIC)
            mask = img.resize(adj_size, resample=Image.NEAREST)

        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        if self.cropsize is not None:
            img, mask = imutils.random_crop([img, mask], self.cropsize, (0, 255))

        mask = imutils.RescaleNearest(0.125)(mask)

        if self.flip is True and bool(random.getrandbits(1)):
            img = np.flip(img, 1).copy()
            mask = np.flip(mask, 1).copy()

        img = np.transpose(img, (2, 0, 1))

        return name, img, mask


class ExtractAffinityLabelInRadius:

    def __init__(self, cropsize, radius=5):
        self.radius = radius

        self.search_dist = []

        for x in range(1, radius):
            self.search_dist.append((0, x))

        for y in range(1, radius):
            for x in range(-radius + 1, radius):
                if x * x + y * y < radius * radius:
                    self.search_dist.append((y, x))

        self.radius_floor = radius - 1

        self.crop_height = cropsize - self.radius_floor
        self.crop_width = cropsize - 2 * self.radius_floor
        return

    def __call__(self, label):

        labels_from = label[:-self.radius_floor, self.radius_floor:-self.radius_floor]
        labels_from = np.reshape(labels_from, [-1])

        labels_to_list = []
        valid_pair_list = []

        for dy, dx in self.search_dist:
            labels_to = label[dy:dy + self.crop_height, self.radius_floor + dx:self.radius_floor + dx + self.crop_width]
            labels_to = np.reshape(labels_to, [-1])

            valid_pair = np.logical_and(np.less(labels_to, 255), np.less(labels_from, 255))

            labels_to_list.append(labels_to)
            valid_pair_list.append(valid_pair)

        bc_labels_from = np.expand_dims(labels_from, 0)
        concat_labels_to = np.stack(labels_to_list)
        concat_valid_pair = np.stack(valid_pair_list)

        pos_affinity_label = np.equal(bc_labels_from, concat_labels_to)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(bc_labels_from, 0)).astype(np.float32)

        fg_pos_affinity_label = np.logical_and(np.logical_and(pos_affinity_label, np.not_equal(bc_labels_from, 0)),
                                               concat_valid_pair).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(pos_affinity_label), concat_valid_pair).astype(np.float32)

        return bg_pos_affinity_label, fg_pos_affinity_label, neg_affinity_label


class AffinityFromMaskDataset(SegmentationDataset):
    def __init__(self, img_name_list_path, img_dir, label_dir, rescale=None, flip=False, cropsize=None,
                 img_transform=None, mask_transform=None, radius=5):
        super().__init__(img_name_list_path, img_dir, label_dir, rescale, flip, cropsize, img_transform, mask_transform)

        self.radius = radius

        self.extract_aff_lab_func = ExtractAffinityLabelInRadius(cropsize=cropsize // 8, radius=radius)

    def __getitem__(self, idx):
        name, img, mask = super().__getitem__(idx)

        aff_label = self.extract_aff_lab_func(mask)

        return name, img, aff_label

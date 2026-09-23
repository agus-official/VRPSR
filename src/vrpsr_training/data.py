from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import math
import numpy as np
import os
import random
import torch
from pathlib import Path
from torch.utils import data as data
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
import hashlib
from vrpsr_training.codec import get_prompt, proc

class SRTrainDataset(data.Dataset):
    """Modified dataset based on the dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt, root, qp=None, codec_type=None):
        super(SRTrainDataset, self).__init__()
        self.qp = qp
        self.codec_type = codec_type
        if self.qp is None:
            self.qps = [i for i in range(25, 49)]
        if self.codec_type is None:
            self.codec_types = ['libx264', 'libx265', 'libx266']
        self.opt = opt
        if 'crop_size' in opt:
            self.crop_size = opt['crop_size']
        else:
            self.crop_size = 512
        self.root = root
        self.images = list_images(root)
        self.transform = transforms.Compose([transforms.Resize(512), transforms.CenterCrop(512)])
        self.totensor = transforms.Compose([transforms.ToTensor()])
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']
        self.betap_range = opt['betap_range']
        self.sinc_prob = opt['sinc_prob']
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']
        self.final_sinc_prob = opt['final_sinc_prob']
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):
        qp = self.qp
        codec_type = self.codec_type
        if qp is None:
            qp = random.choice(self.qps)
        if codec_type is None:
            codec_type = random.choice(self.codec_types)
        try:
            img_gt = Image.open(self.images[index]).convert('RGB')
        except Exception as e:
            print('bad image {}'.format(self.images[index]))
            raise RuntimeError(f'Cannot read image: {self.images[index]}') from e
        if self.transform is not None:
            img_gt = self.transform(img_gt)
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(self.kernel_list, self.kernel_prob, kernel_size, self.blur_sigma, self.blur_sigma, [-math.pi, math.pi], self.betag_range, self.betap_range, noise_range=None)
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(self.kernel_list2, self.kernel_prob2, kernel_size, self.blur_sigma2, self.blur_sigma2, [-math.pi, math.pi], self.betag_range2, self.betap_range2, noise_range=None)
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor
        img_gt = self.totensor(img_gt)
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)
        return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'codec_type': codec_type, 'qp': qp}
        return return_d

    def __len__(self):
        return len(self.images)

class SRValidationDataset(data.Dataset):
    """Modified dataset based on the dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt, root, fixed_qp=None, fixed_codec_type=None):
        super(SRValidationDataset, self).__init__()
        self.fixed_qp = fixed_qp
        self.fixed_codcec_type = fixed_codec_type
        self.opt = opt
        if 'crop_size' in opt:
            self.crop_size = opt['crop_size']
        else:
            self.crop_size = 512
        self.root = root
        self.images = list_images(root)
        self.transform = transforms.Compose([transforms.Resize(512), transforms.CenterCrop(512)])
        self.totensor = transforms.Compose([transforms.ToTensor()])
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']
        self.betap_range = opt['betap_range']
        self.sinc_prob = opt['sinc_prob']
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']
        self.final_sinc_prob = opt['final_sinc_prob']
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1

    def set_seed_from_name(self, image_name):
        seed = int(hashlib.sha256(image_name.encode('utf-8')).hexdigest(), 16) % 2 ** 32
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        return seed

    def __getitem__(self, index):
        try:
            img_gt = Image.open(self.images[index]).convert('RGB')
        except Exception as e:
            print('bad image {}'.format(self.images[index]))
            raise RuntimeError(f'Cannot read image: {self.images[index]}') from e
        image_name = self.images[index].split('/')[-1]
        seed = self.set_seed_from_name(image_name)
        if self.transform is not None:
            img_gt = self.transform(img_gt)
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(self.kernel_list, self.kernel_prob, kernel_size, self.blur_sigma, self.blur_sigma, [-math.pi, math.pi], self.betag_range, self.betap_range, noise_range=None)
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(self.kernel_list2, self.kernel_prob2, kernel_size, self.blur_sigma2, self.blur_sigma2, [-math.pi, math.pi], self.betag_range2, self.betap_range2, noise_range=None)
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor
        img_gt = self.totensor(img_gt)
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)
        codec_type = self.fixed_codcec_type
        qp = self.fixed_qp
        image_name = os.path.basename(self.images[index])
        return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'codec_type': codec_type, 'qp': qp, 'seed': seed, 'image_name': image_name}
        return return_d

    def __len__(self):
        return len(self.images)

def degrade_train(configs, batch, device, val=False, use_usm=False, resize_lq=True, random_size=False):
    """Degradation pipeline, modified from Real-ESRGAN:
    https://github.com/xinntao/Real-ESRGAN
    """
    jpeger = DiffJPEG(differentiable=False).cuda()
    usm_sharpener = USMSharp().cuda()
    im_gt = batch['gt'].cuda()
    if use_usm:
        im_gt = usm_sharpener(im_gt)
    im_gt = im_gt.to(memory_format=torch.contiguous_format).float()
    kernel1 = batch['kernel1'].cuda()
    kernel2 = batch['kernel2'].cuda()
    sinc_kernel = batch['sinc_kernel'].cuda()
    ori_h, ori_w = im_gt.size()[2:4]
    out = filter2D(im_gt, kernel1)
    updown_type = random.choices(['up', 'down', 'keep'], configs.degradation['resize_prob'])[0]
    if updown_type == 'up':
        scale = random.uniform(1, configs.degradation['resize_range'][1])
    elif updown_type == 'down':
        scale = random.uniform(configs.degradation['resize_range'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scale, mode=mode)
    gray_noise_prob = configs.degradation['gray_noise_prob']
    if random.random() < configs.degradation['gaussian_noise_prob']:
        out = random_add_gaussian_noise_pt(out, sigma_range=configs.degradation['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
    else:
        out = random_add_poisson_noise_pt(out, scale_range=configs.degradation['poisson_scale_range'], gray_prob=gray_noise_prob, clip=True, rounds=False)
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*configs.degradation['jpeg_range'])
    out = torch.clamp(out, 0, 1)
    out = jpeger(out, quality=jpeg_p)
    if random.random() < configs.degradation['second_blur_prob']:
        out = filter2D(out, kernel2)
    updown_type = random.choices(['up', 'down', 'keep'], configs.degradation['resize_prob2'])[0]
    if updown_type == 'up':
        scale = random.uniform(1, configs.degradation['resize_range2'][1])
    elif updown_type == 'down':
        scale = random.uniform(configs.degradation['resize_range2'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, size=(int(ori_h / configs.sf * scale), int(ori_w / configs.sf * scale)), mode=mode)
    gray_noise_prob = configs.degradation['gray_noise_prob2']
    if random.random() < configs.degradation['gaussian_noise_prob2']:
        out = random_add_gaussian_noise_pt(out, sigma_range=configs.degradation['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
    else:
        out = random_add_poisson_noise_pt(out, scale_range=configs.degradation['poisson_scale_range2'], gray_prob=gray_noise_prob, clip=True, rounds=False)
    if random.random() < 0.5:
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // configs.sf, ori_w // configs.sf), mode=mode)
        out = filter2D(out, sinc_kernel)
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*configs.degradation['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
    else:
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*configs.degradation['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // configs.sf, ori_w // configs.sf), mode=mode)
        out = filter2D(out, sinc_kernel)
    im_lq = torch.clamp(out, 0, 1.0)
    gt_size = configs.degradation['gt_size']
    im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, configs.sf)
    lq, gt = (im_lq, im_gt)
    ori_lq = im_lq
    if resize_lq:
        lq = F.interpolate(lq, size=(gt.size(-2), gt.size(-1)), mode='bicubic')
    if random.random() < configs.degradation['no_degradation_prob'] or torch.isnan(lq).any():
        lq = gt
    lq = lq.contiguous()
    prompts, blur_264, bpp = get_prompt(lq, batch['codec_type'], batch['qp'])
    lq = lq * 2 - 1.0
    gt = gt * 2 - 1.0
    if random_size:
        lq, gt = randn_cropinput(lq, gt)
    lq = torch.clamp(lq, -1.0, 1.0)
    gt = torch.clamp(gt, -1.0, 1.0)
    return (lq.to(device), gt.to(device), ori_lq.to(device), prompts, blur_264, bpp)

def degrade_validation(configs, batch, device, seed=None, val=False, use_usm=False, resize_lq=True, random_size=False):
    """Degradation pipeline, modified from Real-ESRGAN:
    https://github.com/xinntao/Real-ESRGAN
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    jpeger = DiffJPEG(differentiable=False).cuda()
    usm_sharpener = USMSharp().cuda()
    im_gt = batch['gt'].cuda()
    if use_usm:
        im_gt = usm_sharpener(im_gt)
    im_gt = im_gt.to(memory_format=torch.contiguous_format).float()
    kernel1 = batch['kernel1'].cuda()
    kernel2 = batch['kernel2'].cuda()
    sinc_kernel = batch['sinc_kernel'].cuda()
    ori_h, ori_w = im_gt.size()[2:4]
    out = filter2D(im_gt, kernel1)
    updown_type = random.choices(['up', 'down', 'keep'], configs.degradation['resize_prob'])[0]
    if updown_type == 'up':
        scale = random.uniform(1, configs.degradation['resize_range'][1])
    elif updown_type == 'down':
        scale = random.uniform(configs.degradation['resize_range'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scale, mode=mode)
    gray_noise_prob = configs.degradation['gray_noise_prob']
    if random.random() < configs.degradation['gaussian_noise_prob']:
        out = random_add_gaussian_noise_pt(out, sigma_range=configs.degradation['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob, seed=seed)
    else:
        out = random_add_poisson_noise_pt(out, scale_range=configs.degradation['poisson_scale_range'], gray_prob=gray_noise_prob, clip=True, rounds=False, seed=seed)
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*configs.degradation['jpeg_range'])
    out = torch.clamp(out, 0, 1)
    out = jpeger(out, quality=jpeg_p)
    if random.random() < configs.degradation['second_blur_prob']:
        out = filter2D(out, kernel2)
    updown_type = random.choices(['up', 'down', 'keep'], configs.degradation['resize_prob2'])[0]
    if updown_type == 'up':
        scale = random.uniform(1, configs.degradation['resize_range2'][1])
    elif updown_type == 'down':
        scale = random.uniform(configs.degradation['resize_range2'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, size=(int(ori_h / configs.sf * scale), int(ori_w / configs.sf * scale)), mode=mode)
    gray_noise_prob = configs.degradation['gray_noise_prob2']
    if random.random() < configs.degradation['gaussian_noise_prob2']:
        out = random_add_gaussian_noise_pt(out, sigma_range=configs.degradation['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob, seed=seed)
    else:
        out = random_add_poisson_noise_pt(out, scale_range=configs.degradation['poisson_scale_range2'], gray_prob=gray_noise_prob, clip=True, rounds=False, seed=seed)
    if random.random() < 0.5:
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // configs.sf, ori_w // configs.sf), mode=mode)
        out = filter2D(out, sinc_kernel)
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*configs.degradation['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
    else:
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*configs.degradation['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // configs.sf, ori_w // configs.sf), mode=mode)
        out = filter2D(out, sinc_kernel)
    im_lq = torch.clamp(out, 0, 1.0)
    gt_size = configs.degradation['gt_size']
    im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, configs.sf)
    lq, gt = (im_lq, im_gt)
    ori_lq = im_lq
    if resize_lq:
        lq = F.interpolate(lq, size=(gt.size(-2), gt.size(-1)), mode='bicubic')
    if random.random() < configs.degradation['no_degradation_prob'] or torch.isnan(lq).any():
        lq = gt
    lq = lq.contiguous()
    prompts, blur_264, bpp = get_prompt(lq, batch['codec_type'], batch['qp'])
    lq = lq * 2 - 1.0
    gt = gt * 2 - 1.0
    lq = torch.clamp(lq, -1.0, 1.0)
    gt = torch.clamp(gt, -1.0, 1.0)
    return (lq.to(device), gt.to(device), ori_lq.to(device), prompts, blur_264.to(device), bpp)

def randn_cropinput(lq, gt, base_size=[64, 128, 256, 512]):
    cur_size_h = random.choice(base_size)
    cur_size_w = random.choice(base_size)
    init_h = lq.size(-2) // 2
    init_w = lq.size(-1) // 2
    lq = lq[:, :, init_h - cur_size_h // 2:init_h + cur_size_h // 2, init_w - cur_size_w // 2:init_w + cur_size_w // 2]
    gt = gt[:, :, init_h - cur_size_h // 2:init_h + cur_size_h // 2, init_w - cur_size_w // 2:init_w + cur_size_w // 2]
    assert lq.size(-1) >= 64
    assert lq.size(-2) >= 64
    return [lq, gt]

class CodecDataset(torch.utils.data.Dataset):

    def __init__(self, root, qp=None, codec_type=None):
        self.root = root
        self.images = list_images(root)
        self.transform = transforms.Compose([transforms.Resize(512), transforms.CenterCrop(512)])
        self.totensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.qp = qp
        self.codec_type = codec_type
        if self.qp is None:
            self.qps = [i for i in range(30, 51)]
        if self.codec_type is None:
            self.codec_types = ['libx264', 'libx265', 'libx266']

    def __getitem__(self, index):
        try:
            img_path = self.images[index]
            filename = os.path.splitext(os.path.basename(img_path))[0]
            img = Image.open(self.images[index]).convert('RGB')
        except Exception as e:
            print('bad image {}'.format(self.images[index]))
            raise RuntimeError(f'Cannot read image: {self.images[index]}') from e
        if self.transform is not None:
            img = self.transform(img)
        qp = self.qp
        codec_type = self.codec_type
        if qp is None:
            qp = random.choice(self.qps)
        if codec_type is None:
            codec_type = random.choice(self.codec_types)
        img, rec, packet_size = proc(img, codec_type, qp)
        prompt = 'A {0} {1:.4f} bpp compressed image.'.format(codec_type, packet_size)
        img = self.totensor(img)
        rec = self.totensor(rec)
        return (img, rec, prompt, filename, packet_size, codec_type, qp)

    def __len__(self):
        return len(self.images)

def list_images(root):
    path = Path(root)
    if path.is_file():
        images = [str(Path(line.strip())) for line in path.read_text().splitlines() if line.strip()]
    elif path.is_dir():
        images = sorted((str(p) for p in path.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}))
    else:
        raise FileNotFoundError(f'Dataset does not exist: {root}')
    if not images:
        raise ValueError(f'No images found: {root}')
    return images

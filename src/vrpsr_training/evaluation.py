import torch
from pathlib import Path
import pyiqa
import tqdm as tori
import util_image

def evaluate(in_path, ref_path, ntest, device):
    metric_paired_dict = {}
    in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
    assert in_path.is_dir()
    ref_path_list = None
    if ref_path is not None:
        ref_path = Path(ref_path) if not isinstance(ref_path, Path) else ref_path
        ref_path_list = sorted([x for x in ref_path.glob('*.[jpJP][pnPN]*[gG]')])
        if ntest is not None:
            ref_path_list = ref_path_list[:ntest]
        metric_paired_dict['psnr'] = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
        metric_paired_dict['lpips'] = pyiqa.create_metric('lpips').to(device)
        metric_paired_dict['dists'] = pyiqa.create_metric('dists').to(device)
        metric_paired_dict['ssim'] = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(device)
    lr_path_list = sorted([x for x in in_path.glob('*.[jpJP][pnPN]*[gG]')])
    if ntest is not None:
        lr_path_list = lr_path_list[:ntest]
    if not lr_path_list:
        raise ValueError(f'No evaluation images in {in_path}')
    if ref_path_list is not None and [p.name for p in lr_path_list] != [p.name for p in ref_path_list]:
        raise ValueError('Prediction/reference filenames differ; use a fresh output directory')
    print(f'Find {len(lr_path_list)} images in {in_path}')
    result = {}
    with torch.no_grad():
        for i in tori.tqdm(range(len(lr_path_list))):
            _in_path = lr_path_list[i]
            _ref_path = ref_path_list[i] if ref_path_list is not None else None
            im_in = util_image.imread(_in_path, chn='rgb', dtype='float32')
            im_in_tensor = util_image.img2tensor(im_in).to(device)
            if ref_path is not None:
                im_ref = util_image.imread(_ref_path, chn='rgb', dtype='float32')
                im_ref_tensor = util_image.img2tensor(im_ref).to(device)
                for key, metric in metric_paired_dict.items():
                    result[key] = result.get(key, 0) + metric(im_in_tensor, im_ref_tensor).item()
    if ref_path is not None:
        fid_metric = pyiqa.create_metric('fid')
        result['fid'] = fid_metric(in_path, ref_path)
    print_results = []
    for key, res in result.items():
        if key == 'fid':
            print(f'{key}: {res:.2f}')
            print_results.append(f'{key}: {res:.2f}')
        else:
            print(f'{key}: {res / len(lr_path_list):.5f}')
            print_results.append(f'{key}: {res / len(lr_path_list):.5f}')
    return print_results

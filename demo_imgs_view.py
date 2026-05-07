import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from igev_plusplus.core.igev_stereo import IGEVStereo
from igev_plusplus.core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import skimage.io
import cv2

from view_rerun import Viewer

DEVICE = 'cuda'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def disparity_to_depth(disp, fx, baseline_mm, 
                       min_disp=1.0, max_disp=None, 
                       invalid_value=np.nan):
    """
    视差图 → 深度图（安全掩码版，支持 numpy/torch）
    """
    # 1. 统一转为 numpy 计算，记录原始类型
    is_torch = type(disp).__module__ == 'torch'
    if is_torch:
        import torch
        device, dtype = disp.device, disp.dtype
        disp_np = disp.detach().cpu().numpy()
    else:
        disp_np = np.asarray(disp)
        
    disp_np = np.squeeze(disp_np).astype(np.float32)
    
    # 2. 构建有效视差掩码（替代 np.clip）
    valid_mask = (disp_np >= min_disp)
    if max_disp is not None:
        valid_mask &= (disp_np <= max_disp)
        
    # 3. 初始化深度图（无效区域填 invalid_value）
    depth = np.full_like(disp_np, invalid_value, dtype=np.float32)
    
    # 4. 仅对有效区域计算深度，避免除零/篡改
    depth[valid_mask] = (fx * baseline_mm) / disp_np[valid_mask]
    
    # 5. 恢复原始类型（如需 torch 输出）
    if is_torch:
        depth = torch.from_numpy(depth).to(device).to(dtype)
        
    return depth

def load_image_cv(imfile):
    img = cv2.imread(imfile, cv2.IMREAD_COLOR)  # (H, W, 3) BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为 RGB！关键一步
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)                 # (1, 3, H, W) RGB

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    viewer = Viewer()

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            # viewer.view_image("left_cam", image1.cpu().numpy())
            # viewer.view_image("right_cam", image2.cpu().numpy())

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = padder.unpad(disp)
            file_stem = imfile1.split('/')[-2]
            filename = os.path.join(output_directory, f'{file_stem}.png')
            disp = disp.cpu().numpy().squeeze()
            

            # ==================== 使用示例 ====================
            # 参数设置（人眼立体视觉模拟）
            fx = 673.0846              # 焦距 [pixels]，典型手机/小相机值
            baseline_mm = 60.12      # 人眼瞳距 ≈ 63mm [mm]
            min_disp = 15.0
            max_disp = 400.0 
            # 假设 disp 是你的模型输出/立体匹配结果 [H, W, 1] 或 [H, W]
            # disp = model_output  # torch.Tensor 或 np.ndarray

            viewer.view_depth("left_cam_disp", disp, fx)

            # 计算深度（单位：mm）
            depth_mm = disparity_to_depth(
                disp=disp, 
                fx=fx, 
                baseline_mm=baseline_mm,
                min_disp=min_disp,
                max_disp=max_disp,
            )
            viewer.view_depth("left_cam_depth", depth_mm, fx)

            plt.imsave(filename, disp.squeeze(), cmap='jet')
            
            if args.save_numpy:
                np.save(output_directory / f"{file_stem}.npy", disp.squeeze())

            break

            # disp = np.round(disp * 256).astype(np.uint16)
            # cv2.imwrite(filename, cv2.applyColorMap(cv2.convertScaleAbs(disp.squeeze(), alpha=0.01),cv2.COLORMAP_JET), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/igev_plusplus/sceneflow.pth')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="./demo-imgs/*/im0.jpg")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="./demo-imgs/*/im1.jpg")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', default=True, help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=16, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=768, help="max disp range")
    parser.add_argument('--s_disp_range', type=int, default=48, help="max disp of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_range', type=int, default=96, help="max disp of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_range', type=int, default=192, help="max disp of large disparity-range geometry encoding volume")
    parser.add_argument('--s_disp_interval', type=int, default=1, help="disp interval of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_interval', type=int, default=2, help="disp interval of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_interval', type=int, default=4, help="disp interval of large disparity-range geometry encoding volume")
    
    args = parser.parse_args()

    demo(args)

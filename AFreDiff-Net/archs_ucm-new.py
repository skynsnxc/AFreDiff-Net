import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import nibabel as nib
from torch.utils.data import DataLoader
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

# 从 archs_AFreDiff-Net.py 中提取的必要类（简化为展示）
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride  # 修改为 stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=stride)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class DiffusionMLP(nn.Module):
    def __init__(self, dim, num_steps=4, beta=0.001):
        super(DiffusionMLP, self).__init__()
        self.dim = dim
        self.num_steps = num_steps
        self.beta = beta
        self.denoise_conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.act = nn.GELU()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.register_buffer('alphas', torch.tensor(1.0 - beta))
        self.register_buffer('alphas_cumprod', torch.cumprod(torch.full((num_steps,), 1.0 - beta), dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        for t in range(self.num_steps - 1, -1, -1):
            noise_pred = self.denoise_conv(x)
            alpha_t = self.alphas.to(x.device)
            sigma_t = self.sqrt_one_minus_alphas_cumprod[t].to(x.device)
            x = (x - ((1 - alpha_t) / sigma_t) * noise_pred) / torch.sqrt(alpha_t)
            x = self.act(x)
        x = x.permute(0, 2, 3, 1).view(B, N, C)
        x = self.mlp(x)
        return x

class AFreDiff-NetBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, shift_size=5):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.dim = dim
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.dwconv = DWConv(mlp_hidden_dim)
        self.dwconv1 = DWConv(mlp_hidden_dim)
        self.act = act_layer()
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.drop = nn.Dropout(drop)

        self.freq_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.filter_predictor = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, 2, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.fusion_weight = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def _generate_adaptive_filter(self, x_2d, H, W):
        B, C, H, W = x_2d.shape
        W_half = W // 2 + 1
        filter_params = self.filter_predictor(x_2d)
        filter_params = filter_params[..., :W_half]
        center_freq = filter_params[:, 0:1, :, :]
        bandwidth = filter_params[:, 1:2, :, :]
        freq_y = torch.linspace(0, 1, H, device=x_2d.device).view(1, 1, H, 1)
        freq_x = torch.linspace(0, 1, W_half, device=x_2d.device).view(1, 1, 1, W_half)
        freq_grid = torch.sqrt(freq_y ** 2 + freq_x ** 2)
        high_pass = 1 - torch.exp(-((freq_grid - center_freq) ** 2) / (2 * bandwidth ** 2 + 1e-6))
        return high_pass

    def _freq_branch(self, x_2d, H, W):
        B, C, H, W = x_2d.shape
        x_2d = self.freq_conv(x_2d)
        x_freq = torch.fft.rfft2(x_2d, dim=(-2, -1), norm='ortho')
        high_pass = self._generate_adaptive_filter(x_2d, H, W)
        high_freq = x_freq * high_pass
        x_freq_enhanced = torch.fft.irfft2(high_freq, s=(H, W), norm='ortho')
        return x_freq_enhanced

    def _spatial_branch(self, x_2d):
        return self.spatial_conv(x_2d)

    def forward(self, x, H, W):
        x = self.norm2(x)
        B, N, C = x.shape
        x1 = x.clone().detach()
        x_2d = x.transpose(1, 2).view(B, C, H, W).contiguous()
        freq_out = self._freq_branch(x_2d, H, W)
        spatial_out = self._spatial_branch(x_2d)
        combined = torch.cat([freq_out, spatial_out], dim=1)
        fusion_weight = self.fusion_weight(combined)
        x_2d = freq_out * fusion_weight + spatial_out * (1 - fusion_weight)
        x = x_2d.flatten(2).transpose(1, 2)
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = self.act1(xn)
        x = self.drop(xn)
        x_s = x.reshape(B, C, H * W).contiguous()
        x = x_s.transpose(1, 2)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.dwconv1(x, H, W)
        x = self.drop(x)
        x += x1
        x = x + self.drop_path(x)
        return x

class AFreDiff-Net_Net(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=256, patch_size=16, in_chans=3,
                 embed_dims=[8, 16, 24, 32, 48, 64, 3], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        self.encoder1 = nn.Conv2d(embed_dims[-1], embed_dims[0], 3, stride=1, padding=1)
        self.ebn1 = nn.GroupNorm(4, embed_dims[0])
        self.norm1 = norm_layer(embed_dims[1])
        self.norm2 = norm_layer(embed_dims[2])
        self.norm3 = norm_layer(embed_dims[3])
        self.norm4 = norm_layer(embed_dims[4])
        self.norm5 = norm_layer(embed_dims[5])
        self.dnorm2 = norm_layer(embed_dims[4])
        self.dnorm3 = norm_layer(embed_dims[3])
        self.dnorm4 = norm_layer(embed_dims[2])
        self.dnorm5 = norm_layer(embed_dims[1])
        self.dnorm6 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.block_0_1 = nn.ModuleList([AFreDiff-NetBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.block0 = nn.ModuleList([AFreDiff-NetBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.block1 = nn.ModuleList([AFreDiff-NetBlock(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.block2 = nn.ModuleList([AFreDiff-NetBlock(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.block3 = nn.ModuleList([AFreDiff-NetBlock(
            dim=embed_dims[5], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.dblock0 = nn.ModuleList([AFreDiff-NetBlock(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.dblock1 = nn.ModuleList([AFreDiff-NetBlock(
            dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.dblock2 = nn.ModuleList([AFreDiff-NetBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.dblock3 = nn.ModuleList([AFreDiff-NetBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.dblock4 = nn.ModuleList([AFreDiff-NetBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])

        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 2, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[3], embed_dim=embed_dims[4])
        self.patch_embed5 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[4], embed_dim=embed_dims[5])

        self.decoder0 = nn.Conv2d(embed_dims[5], embed_dims[4], 1, stride=1, padding=0)
        self.decoder1 = nn.Conv2d(embed_dims[4], embed_dims[3], 1, stride=1, padding=0)
        self.decoder2 = nn.Conv2d(embed_dims[3], embed_dims[2], 1, stride=1, padding=0)
        self.decoder3 = nn.Conv2d(embed_dims[2], embed_dims[1], 1, stride=1, padding=0)
        self.decoder4 = nn.Conv2d(embed_dims[1], embed_dims[0], 1, stride=1, padding=0)
        self.decoder5 = nn.Conv2d(embed_dims[0], embed_dims[-1], 1, stride=1, padding=0)

        self.dbn0 = nn.GroupNorm(4, embed_dims[4])
        self.dbn1 = nn.GroupNorm(4, embed_dims[3])
        self.dbn2 = nn.GroupNorm(4, embed_dims[2])
        self.dbn3 = nn.GroupNorm(4, embed_dims[1])
        self.dbn4 = nn.GroupNorm(4, embed_dims[0])

        self.finalpre0 = nn.Conv2d(embed_dims[4], num_classes, kernel_size=1)
        self.finalpre1 = nn.Conv2d(embed_dims[3], num_classes, kernel_size=1)
        self.finalpre2 = nn.Conv2d(embed_dims[2], num_classes, kernel_size=1)
        self.finalpre3 = nn.Conv2d(embed_dims[1], num_classes, kernel_size=1)
        self.finalpre4 = nn.Conv2d(embed_dims[0], num_classes, kernel_size=1)
        self.final = nn.Conv2d(embed_dims[-1], num_classes, kernel_size=1)

        self.diffusion_mlp1 = DiffusionMLP(dim=embed_dims[5])
        self.diffusion_mlp2 = DiffusionMLP(dim=embed_dims[4])
        self.skip_conv = nn.Conv2d(embed_dims[5], embed_dims[4], 1, stride=1, padding=0)

    def forward(self, x, inference_mode=False):
        B = x.shape[0]
        out = self.encoder1(x)
        out = F.relu(F.max_pool2d(self.ebn1(out), 2, 2))
        t1 = out
        out, H, W = self.patch_embed1(out)
        for i, blk in enumerate(self.block_0_1):
            out = blk(out, H, W)
        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t2 = out
        out, H, W = self.patch_embed2(out)
        for i, blk in enumerate(self.block0):
            out = blk(out, H, W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t3 = out
        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out
        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t5 = out
        out, H, W = self.patch_embed5(out)
        for i, blk in enumerate(self.block3):
            out = blk(out, H, W)
        out = self.norm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = out.flatten(2).transpose(1, 2)
        out = self.diffusion_mlp1(out, H, W)
        skip = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn0(self.decoder0(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t5)

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        out = self.diffusion_mlp2(out, H, W)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        skip_adjusted = self.skip_conv(F.interpolate(skip, scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, skip_adjusted)

        if not inference_mode:
            outtpre0 = F.interpolate(out, scale_factor=32, mode='bilinear', align_corners=True)
            outtpre0 = self.finalpre0(outtpre0)

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock0):
            out = blk(out, H, W)
        out = self.dnorm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t4)
        if not inference_mode:
            outtpre1 = F.interpolate(out, scale_factor=16, mode='bilinear', align_corners=True)
            outtpre1 = self.finalpre1(outtpre1)

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t3)
        if not inference_mode:
            outtpre2 = F.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True)
            outtpre2 = self.finalpre2(outtpre2)

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        if not inference_mode:
            outtpre3 = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)
            outtpre3 = self.finalpre3(outtpre3)

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock3):
            out = blk(out, H, W)
        out = self.dnorm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        if not inference_mode:
            outtpre4 = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
            outtpre4 = self.finalpre4(outtpre4)

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock4):
            out = blk(out, H, W)
        out = self.dnorm6(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))
        out = self.final(out)

        if not inference_mode:
            return (outtpre0, outtpre1, outtpre2, outtpre3, outtpre4), out
        else:
            return out

# 数据集类（从之前的代码中提取）
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None, train=True, dataset_name='isic2018'):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        if self.dataset_name == 'brats':
            img_path = str(os.path.join(self.img_dir, img_id + self.img_ext))
            img_nii = nib.load(img_path).get_fdata()
            slice_idx = img_nii.shape[-1] // 2
            img = img_nii[:, :, slice_idx]
            img = np.stack([img, img, img], axis=-1)
        else:
            img = cv2.imread(str(os.path.join(self.img_dir, img_id + self.img_ext)))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = []
        if self.dataset_name == 'isic2017':
            mask.append(cv2.imread(str(os.path.join(self.mask_dir, img_id + "_segmentation" + self.mask_ext)), cv2.IMREAD_GRAYSCALE)[..., None])
        elif self.dataset_name == 'ph2':
            mask.append(cv2.imread(str(os.path.join(self.mask_dir, img_id + "_lesion" + self.mask_ext)), cv2.IMREAD_GRAYSCALE)[..., None])
        elif self.dataset_name == 'isic2018':
            mask.append(cv2.imread(str(os.path.join(self.mask_dir, img_id + self.mask_ext)), cv2.IMREAD_GRAYSCALE)[..., None])
        elif self.dataset_name == 'kvasir-seg':
            mask.append(cv2.imread(str(os.path.join(self.mask_dir, img_id + self.mask_ext)), cv2.IMREAD_GRAYSCALE)[..., None])
        elif self.dataset_name == 'brats':
            mask_path = str(os.path.join(self.mask_dir, img_id + "_seg" + self.mask_ext))
            mask_nii = nib.load(mask_path).get_fdata()
            slice_idx = mask_nii.shape[-1] // 2
            mask.append(mask_nii[:, :, slice_idx][..., None])
        elif self.dataset_name == 'busi':
            mask.append(cv2.imread(str(os.path.join(self.mask_dir, img_id + "_mask" + self.mask_ext)), cv2.IMREAD_GRAYSCALE)[..., None])
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img_normalized = img.astype('float32')
        img = ((img_normalized - np.min(img_normalized)) / (np.max(img_normalized) - np.min(img_normalized)))
        img = img.transpose(2, 0, 1)

        mask = mask.astype('float32') / 255.
        mask = mask.transpose(2, 0, 1)

        return img, mask, {'img_id': img_id}

# Dice 损失函数（用于医学图像分割）
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        return 1 - ((2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth))

# 主框架
def parse_args():
    parser = argparse.ArgumentParser(description="AFreDiff-Net_Net Training")
    parser.add_argument('--dataset', type=str, default='isic2018',
                        choices=['isic2017', 'ph2', 'isic2018', 'Kvasir-SEG', 'BUSI', 'BRATS'],
                        help='Dataset to use for training')
    parser.add_argument('--img-dir', type=str, required=True, help='Path to images')
    parser.add_argument('--mask-dir', type=str, required=True, help='Path to masks')
    parser.add_argument('--img-ext', type=str, default='.jpg', help='Image file extension')
    parser.add_argument('--mask-ext', type=str, default='.png', help='Mask file extension')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--img-size', type=int, default=256, help='Input image size')
    return parser.parse_args()

def main():
    args = parse_args()

    # 获取 img_ids（这里假设从文件系统获取）
    img_ids = [f.split('.')[0] for f in os.listdir(args.img_dir) if f.endswith(args.img_ext)]

    # 实例化 Dataset
    dataset = Dataset(
        img_ids=img_ids,
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        img_ext=args.img_ext,
        mask_ext=args.mask_ext,
        num_classes=1,
        transform=None,
        train=True,
        dataset_name=args.dataset
    )

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AFreDiff-Net_Net(
        num_classes=1,
        input_channels=3,
        deep_supervision=True,
        img_size=args.img_size
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = DiceLoss()

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for img, mask, _ in dataloader:
            img, mask = img.to(device).float(), mask.to(device).float()
            optimizer.zero_grad()
            outputs, final_output = model(img, inference_mode=False)

            # 计算深度监督损失
            loss = 0
            weights = [0.1, 0.1, 0.2, 0.3, 0.3]  # 深度监督各层的权重
            for i, output in enumerate(outputs):
                loss += weights[i] * criterion(output, mask)
            loss += criterion(final_output, mask)  # 最终输出损失

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(dataloader):.4f}")

    print("Training completed!")

if __name__ == "__main__":
    main()
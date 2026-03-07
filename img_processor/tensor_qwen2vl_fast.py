# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Differentiable Image processor class for Qwen2-VL, optimized for adversarial attacks."""

import math
from typing import Dict, List, Optional, Union, Tuple

import torch
import torch.nn.functional as F

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import (
    convert_to_rgb,
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    to_numpy_array,
    get_image_size,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    logging,
)
from ...video_utils import VideoInput, make_batched_videos

logger = logging.get_logger(__name__)


# ------------------------------------------------------------------------------------------------
# Helper Functions (Differentiable or Utility)
# ------------------------------------------------------------------------------------------------

# 调整 smart_resize 以避免使用非可微的 round/floor/ceil，尽管这些在预处理时是可接受的
# 因为它们只计算目标尺寸，而不是梯度。我们使用 math 库进行尺寸计算。
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
) -> Tuple[int, int]:
    """Rescales the image so that dimensions are divisible by 'factor', and total pixels are within min/max."""
    # 计算新的高度和宽度，使其为 factor 的倍数
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    
    # 避免 h_bar 或 w_bar 为 0 的情况
    if h_bar == 0: h_bar = factor
    if w_bar == 0: w_bar = factor

    # 检查最大像素限制
    if h_bar * w_bar > max_pixels:
        # 使用数学方法计算缩放因子 beta
        beta = math.sqrt((height * width) / max_pixels)
        # 向下取整到 factor 的倍数
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    
    # 检查最小像素限制
    elif h_bar * w_bar < min_pixels:
        # 使用数学方法计算缩放因子 beta
        beta = math.sqrt(min_pixels / (height * width))
        # 向上取整到 factor 的倍数
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
        
    # 再次确保 h_bar 和 w_bar 不小于 factor
    h_bar = max(factor, h_bar)
    w_bar = max(factor, w_bar)
    
    return int(h_bar), int(w_bar)

def _to_tensor(imgs: ImageInput) -> torch.Tensor:
    """Converts a PIL/Numpy image to a C, H, W float tensor."""
    if isinstance(imgs, torch.Tensor):
        if imgs.ndim == 4:
             # Already a batch tensor [B, C, H, W] or [B, H, W, C], assume [B, C, H, W]
             return imgs.float()
        elif imgs.ndim == 3:
             # Single image tensor [C, H, W] or [H, W, C]
             if infer_channel_dimension_format(imgs) == ChannelDimension.LAST:
                imgs = imgs.permute(2, 0, 1)
             return imgs.float()
        else:
            raise ValueError(f"Unsupported tensor input with {imgs.ndim} dimensions.")

    np_imgs = to_numpy_array(imgs)
    if np_imgs.ndim == 2:
        np_imgs = np_imgs[None, ...] # Add channel dim for grayscale [1, H, W]
    
    if infer_channel_dimension_format(np_imgs) == ChannelDimension.LAST:
        np_imgs = np_imgs.transpose(2, 0, 1) # Convert to [C, H, W]
        
    return torch.from_numpy(np_imgs.copy()).float()


def make_batched_images(images: ImageInput) -> List[ImageInput]:
    """Helper to ensure we have a list of images (PIL, Numpy, or Tensor)."""
    if isinstance(images, (list, tuple)):
        # Handle list of images (PIL, Numpy, or Tensor)
        if images and not isinstance(images[0], (list, tuple)):
            return list(images)
        # Handle list of list of images (flatterned)
        elif images and isinstance(images[0], (list, tuple)):
            return [img for img_list in images for img in img_list]

    return [images]


# ------------------------------------------------------------------------------------------------
# Differentiable Image Processor Class
# ------------------------------------------------------------------------------------------------

@auto_docstring
class Qwen2VLImageProcessorFast(BaseImageProcessor):
    """
    Constructs a Qwen2-VL image processor that is fully differentiable,
    optimized for adversarial attack research.
    """

    model_input_names = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]

    # Default parameters from Qwen2VLFastImageProcessor
    do_resize = True
    resample = PILImageResampling.BICUBIC
    do_rescale = True
    rescale_factor = 1 / 255  # Default
    do_normalize = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    min_pixels = 56 * 56
    max_pixels = 28 * 28 * 1280

    def __init__(self: "Qwen2VLDifferentiableImageProcessor", **kwargs: Unpack[Dict]):
        # Custom logic to handle min_pixels/max_pixels backward compatibility if provided via kwargs
        size = kwargs.pop("size", None)
        min_pixels = kwargs.pop("min_pixels", None)
        max_pixels = kwargs.pop("max_pixels", None)

        # Apply backward compatibility logic if min_pixels/max_pixels are provided
        if min_pixels is not None:
            self.min_pixels = min_pixels
        if max_pixels is not None:
            self.max_pixels = max_pixels
            
        if size is not None:
            if "shortest_edge" in size:
                 self.min_pixels = size["shortest_edge"]
            if "longest_edge" in size:
                 self.max_pixels = size["longest_edge"]

        # Ensure instance attributes are set from class attributes/kwargs
        super().__init__(
            do_resize=kwargs.pop("do_resize", self.do_resize),
            resample=kwargs.pop("resample", self.resample),
            do_rescale=kwargs.pop("do_rescale", self.do_rescale),
            rescale_factor=kwargs.pop("rescale_factor", self.rescale_factor),
            do_normalize=kwargs.pop("do_normalize", self.do_normalize),
            image_mean=kwargs.pop("image_mean", self.image_mean),
            image_std=kwargs.pop("image_std", self.image_std),
            do_convert_rgb=kwargs.pop("do_convert_rgb", self.do_convert_rgb),
            patch_size=kwargs.pop("patch_size", self.patch_size),
            temporal_patch_size=kwargs.pop("temporal_patch_size", self.temporal_patch_size),
            merge_size=kwargs.pop("merge_size", self.merge_size),
            **kwargs,
        )
        self.size = {"shortest_edge": self.min_pixels, "longest_edge": self.max_pixels}
        # Update with potentially new values from kwargs
        self.min_pixels = self.size["shortest_edge"]
        self.max_pixels = self.size["longest_edge"]
        
    def _preprocess_images_to_tensor(
        self,
        images: ImageInput,
        do_convert_rgb: bool,
    ) -> torch.Tensor:
        """Converts image inputs (PIL/Numpy/Tensor) into a single 4D PyTorch tensor [B, C, H, W]."""
        
        # 1. Handle Tensor Input
        if isinstance(images, torch.Tensor):
            if images.ndim == 3: # [C, H, W]
                return _to_tensor(images).unsqueeze(0)
            elif images.ndim == 4: # [B, C, H, W]
                return _to_tensor(images)
            else:
                raise ValueError(f"Unsupported tensor input with {images.ndim} dimensions.")

        # 2. Handle PIL/Numpy/List Input
        images_list = make_batched_images(images)
        if do_convert_rgb:
             images_list = [convert_to_rgb(image) for image in images_list]
             
        tensor_list = [_to_tensor(img) for img in images_list]
        
        # Standardize to [C, H, W] (already done in _to_tensor)
        
        if not tensor_list:
            return torch.empty(0, 3, 0, 0)
        
        # Stack to create a batch tensor [B, C, H, W]
        # NOTE: This assumes all images in the batch have the same dimensions *before* resizing, 
        # which is usually the case for standard batching.
        first_shape = tensor_list[0].shape
        if not all(t.shape == first_shape for t in tensor_list):
             logger.warning("Images in the batch have different dimensions. This may lead to incorrect behavior if resizing is disabled or if the difference is not handled by smart_resize logic.")
             # For a truly differentiable pipeline, we must ensure all inputs can be stacked.
             # In a typical DL setup, inputs are pre-resized to a common size or batched by size,
             # but since we removed the complex grouping logic, we stack them directly.
        
        return torch.stack(tensor_list, dim=0)


    def _preprocess(
        self,
        images: torch.Tensor, # Input is [B, C, H, W]
        do_resize: bool,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: List[float],
        image_std: List[float],
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable core preprocessing function for a batch of images/frames.
        This includes resizing, normalization, and the Qwen2-VL specific patching/flattening.
        """
        if images.ndim != 4:
             raise ValueError(f"Expected a 4D tensor batch [B, C, H, W], but got shape {images.shape}")
        
        batch_size, num_channels, height, width = images.shape
        resized_height, resized_width = height, width
        
        # 1. Differentiable Resize
        if do_resize:
             # smart_resize is not differentiable but only determines *target size*,
             # which is fine for the setup phase.
             resized_height, resized_width = smart_resize(
                 height,
                 width,
                 factor=patch_size * merge_size,
                 min_pixels=self.min_pixels,
                 max_pixels=self.max_pixels,
             )
             images = F.interpolate(
                 images,
                 size=(resized_height, resized_width),
                 mode='bicubic',
                 align_corners=False, # Standard for bicubic interpolation
             )
        
        # 2. Differentiable Rescale
        if do_rescale:
             images = images * rescale_factor
             
        # 3. Differentiable Normalize
        if do_normalize:
             # Ensure mean/std are on the correct device/dtype
             mean = torch.tensor(image_mean, device=images.device, dtype=images.dtype).view(1, num_channels, 1, 1)
             std = torch.tensor(image_std, device=images.device, dtype=images.dtype).view(1, num_channels, 1, 1)
             images = (images - mean) / std

        patches = images
        
        # --- 4. Qwen2-VL Specific Patching and Flattening (Core Logic) ---
        # Note: For *images*, the temporal dimension (T) is 1. T is the first dim here: [B, C, H, W]
        # The logic in the original Qwen2VLImageProcessor assumes the input to _preprocess is a **video tensor**
        # of shape [T, C, H, W] when processing a single video.
        
        # We adapt the _preprocess to handle a *batch* of images [B, C, H, W] or *multiple frames/videos* [T, C, H, W] 
        # that have been stacked/concatenated into [B/T, C, H, W].
        
        # For simplicity in this Differentiable Processor, we'll assume the input `images` is a batch of
        # **already-stacked video frames/images**, where the **first dimension** acts as the temporal/batch axis.
        
        is_video_input = batch_size > 1 and images.ndim == 4 # Heuristic check for multiple frames
        
        # For non-video (image) input, we temporarily add a time dimension T=1 to match the video processing path
        if not is_video_input:
            patches = patches.unsqueeze(1) # [B, 1, C, H, W]
        else:
            # If it is a video batch (e.g., [T, C, H, W] for T frames of *one* video),
            # we consider the first dimension as T, and assume B=1 for the whole video.
            # However, the subsequent reshape relies on a *true* batch dimension B at the start.
            # To simplify, we keep it as a batch of images/frames: [B, C, H, W]
            patches = patches.unsqueeze(1) # [B, 1, C, H, W], where B is the number of frames/images
        
        # For a batch of B images, patches is [B, 1, C, H, W]
        # For a video of T frames, patches is [T, 1, C, H, W] (assuming T=B)
        
        patches = patches.permute(0, 2, 1, 3, 4) # [B/T, C, 1, H, W]
        
        # Temporal padding (if it's not a single image/frame)
        num_frames = patches.shape[2] # T=1 for single image, T for video
        
        if num_frames % temporal_patch_size != 0:
            repeats_needed = temporal_patch_size - (num_frames % temporal_patch_size)
            # Repeat the *last frame* along the temporal axis
            repeats = patches[:, :, -1:].repeat(1, 1, repeats_needed, 1, 1)
            patches = torch.cat([patches, repeats], dim=2)
            num_frames += repeats_needed
        
        batch_size, channel, grid_t_full, resized_height, resized_width = patches.shape
        grid_t = grid_t_full // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

        # The core complex reshaping for Qwen2-VL's attention mechanism:
        # 1. Reshape into 9 dimensions, grouping by grids, merge sizes, and patch sizes
        # [B, C, T_full, H, W] -> [B, C, grid_t, t_patch, grid_h/m, m, p, grid_w/m, m, p]
        patches = patches.reshape(
            batch_size,
            channel,
            grid_t,
            temporal_patch_size,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        
        # 2. Reorder dimensions to group grid and patch information
        # Target permutation: (B, grid_t, grid_h/m, grid_w/m, m, m, C, t_patch, p_h, p_w)
        # Original: (B, C, grid_t, t_patch, grid_h/m, m, p_h, grid_w/m, m, p_w) -> dims 0..9
        patches = patches.permute(
            0, # B
            2, # grid_t
            4, # grid_h/m (index 4)
            7, # grid_w/m (index 7)
            5, # m_h (index 5)
            8, # m_w (index 8)
            1, # C (index 1)
            3, # t_patch (index 3)
            6, # p_h (index 6)
            9, # p_w (index 9)
        )
        
        # 3. Final flattening: (B, grid_t * grid_h * grid_w, C * t_patch * p_h * p_w)
        flatten_patches = patches.reshape(
            batch_size,
            grid_t * grid_h * grid_w,
            -1, # Automatically calculates C * t_patch * p_h * p_w
        )
        
        # grid_thw for a single image/video is (grid_t, grid_h, grid_w)
        grid_thw = torch.tensor([grid_t, grid_h, grid_w], device=images.device)

        return flatten_patches, grid_thw


    def preprocess(
        self,
        images: ImageInput,
        videos: Optional[VideoInput] = None, # Simplified video support
        do_resize: bool = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs: Unpack[Dict],
    ) -> BatchFeature:
        """
        Differentiable entry point for preprocessing an image or a batch of images/videos.
        """
        
        # --- Resolve arguments ---
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        return_tensors = return_tensors if return_tensors is not None else TensorType.PYTORCH
        
        # --- Image Processing ---
        data = {}
        if images is not None:
            # Convert all images to a single batch tensor [B, C, H, W]
            image_tensors = self._preprocess_images_to_tensor(images, do_convert_rgb)
            
            # The patching logic is performed assuming a single batch of images.
            pixel_values, image_grid_thw_tensor = self._preprocess(
                image_tensors,
                do_resize=do_resize,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                patch_size=self.patch_size,
                temporal_patch_size=self.temporal_patch_size,
                merge_size=self.merge_size,
            )
            # The grid is the same for all images in the batch, so we duplicate it.
            batch_size = pixel_values.shape[0]
            image_grid_thw = image_grid_thw_tensor.unsqueeze(0).repeat(batch_size, 1).tolist()
            
            data = {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}

        # --- Video Processing ---
        if videos is not None:
             # In the context of adversarial attacks, the video input is typically a single video 
             # (a list of frames [T, C, H, W]), or a batch of videos.
             videos_list = make_batched_videos(videos)
             pixel_values_list, video_grid_thw_list = [], []
             
             for video_frames in videos_list:
                  # Convert all frames to a single tensor [T, C, H, W]
                  frame_tensors = self._preprocess_images_to_tensor(video_frames, do_convert_rgb)
                  
                  # Process the video (frames stacked as batch)
                  patches, video_grid_thw_tensor = self._preprocess(
                       frame_tensors,
                       do_resize=do_resize,
                       do_rescale=do_rescale,
                       rescale_factor=rescale_factor,
                       do_normalize=do_normalize,
                       image_mean=image_mean,
                       image_std=image_std,
                       patch_size=self.patch_size,
                       temporal_patch_size=self.temporal_patch_size,
                       merge_size=self.merge_size,
                  )
                  # In the video case, the batch_size (B) is 1 because the time dimension (T) is handled inside the patch logic.
                  # The output patches should be [1, N_patches, Feature_dim].
                  if patches.shape[0] != 1:
                       patches = patches.mean(dim=0, keepdim=True) # A simple way to aggregate patches from multiple videos if they were mistakenly batched
                  
                  pixel_values_list.append(patches.squeeze(0)) # append the [N_patches, Feature_dim] tensor
                  video_grid_thw_list.append(video_grid_thw_tensor.tolist())
             
             # Concatenate all patches from the batch of videos
             pixel_values_videos = torch.stack(pixel_values_list, dim=0) # [B_videos, N_patches, Feature_dim]
             
             if images is None:
                  data = {"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw_list}
             else:
                  # If both images and videos are present, we add the video outputs as separate fields
                  data.update({"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw_list})

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["Qwen2VLImageProcessorFast"]
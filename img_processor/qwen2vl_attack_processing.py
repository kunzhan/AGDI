# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for Qwen2-VL."""

import math
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F


from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import (
    convert_to_rgb,
    # resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    VideoInput,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_valid_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from transformers.utils import TensorType, is_vision_available, logging


logger = logging.get_logger(__name__)


if is_vision_available():
    from PIL import Image


def make_batched_images(images) -> List[List[ImageInput]]:
    """
    Accepts images in list or nested list format, and makes a list of images for preprocessing.

    Args:
        images (`Union[List[List[ImageInput]], List[ImageInput], ImageInput]`):
            The input image.

    Returns:
        list: A list of images.
    """
    if isinstance(images, (list, tuple)) and isinstance(images[0], (list, tuple)) and is_valid_image(images[0][0]):
        return [img for img_list in images for img in img_list]

    elif isinstance(images, (list, tuple)) and is_valid_image(images[0]):
        return images

    elif is_valid_image(images):
        return [images]

    raise ValueError(f"Could not make batched images from {images}")


# Copied from transformers.models.llava_next_video.image_processing_llava_next_video.make_batched_videos
def make_batched_videos(videos) -> List[VideoInput]:
    if isinstance(videos, (list, tuple)) and isinstance(videos[0], (list, tuple)) and is_valid_image(videos[0][0]):
        return videos

    elif isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):
        if isinstance(videos[0], Image.Image):
            return [videos]
        elif len(videos[0].shape) == 4:
            return [list(video) for video in videos]

    elif is_valid_image(videos) and len(videos.shape) == 4:
        return [list(videos)]

    raise ValueError(f"Could not make batched video from {videos}")


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class Qwen2VLImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Qwen2-VL image processor that dynamically resizes images based on the original images.
    This version is modified to be fully differentiable for adversarial attack research.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats for each channel in the image.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats for each channel in the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        min_pixels (`int`, *optional*, defaults to `56 * 56`):
            The min pixels of the image to resize the image.
        max_pixels (`int`, *optional*, defaults to `28 * 28 * 1280`):
            The max pixels of the image to resize the image.
        patch_size (`int`, *optional*, defaults to 14):
            The spacial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The temporal patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
    """

    model_input_names = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]

    def __init__(
        self,
        do_resize: bool = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: int = 56 * 56,
        max_pixels: int = 28 * 28 * 1280,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        # resample is kept for compatibility but F.interpolate with 'bicubic' is used internally
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.size = {"min_pixels": min_pixels, "max_pixels": max_pixels}
        self.do_convert_rgb = do_convert_rgb

    def _preprocess(
        self,
        images: torch.Tensor,
        do_resize: bool = None,
        resample: PILImageResampling = None, # Not used, kept for signature consistency
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None, # Assumes input tensor is already in correct RGB format
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Differentiable preprocessing function for a batch of images represented as a PyTorch tensor.
        This version contains the corrected patching and flattening logic.
        """

        if images.ndim == 5 and images.shape[0] == 1:
            images = images.squeeze(0)
        
        if images.ndim != 4:
            raise ValueError(f"Expected a 4D tensor batch [B, C, H, W], but got shape {images.shape}")

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images)

        if input_data_format == ChannelDimension.LAST:
            images = images.permute(0, 3, 1, 2)
        

        if images.ndim == 3:
            images = images.unsqueeze(0)
        
        batch_size, num_channels, height, width = images.shape
        resized_height, resized_width = height, width
        
        if do_resize:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=self.patch_size * self.merge_size,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            images = F.interpolate(
                images,
                size=(resized_height, resized_width),
                mode='bicubic',
                align_corners=False,
            )

        if do_rescale:
            images = images * rescale_factor

        if do_normalize:
            mean = torch.tensor(image_mean, device=images.device, dtype=images.dtype).view(1, num_channels, 1, 1)
            std = torch.tensor(image_std, device=images.device, dtype=images.dtype).view(1, num_channels, 1, 1)
            images = (images - mean) / std


        
        patches = images
        

        if patches.shape[0] % self.temporal_patch_size != 0:
            repeats_needed = self.temporal_patch_size - (patches.shape[0] % self.temporal_patch_size)
            last_image_tensor = patches[-1].unsqueeze(0)
            repeats = last_image_tensor.repeat(repeats_needed, 1, 1, 1)
            patches = torch.cat([patches, repeats], dim=0)

        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size


        # shape: [grid_t, temp_patch_size, C, grid_h/merge_size, merge_size, patch_size, grid_w/merge_size, merge_size, patch_size]
        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        

        # (t, h/m, w/m, m, m, c, p_t, p_h, p_w)
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        

        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, -1
        )

        return flatten_patches, (grid_t, grid_h, grid_w)


    def preprocess(
        self,
        images: ImageInput,
        videos: VideoInput = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        """
        Preprocess an image or a batch of images.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size # size is used by smart_resize
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        
        # For adversarial attacks, we expect `return_tensors` to be 'pt'
        if return_tensors is None:
            return_tensors = TensorType.PYTORCH

        # The core logic requires PyTorch tensors. We handle conversion at the beginning.
        def _to_tensor(imgs):
            if not isinstance(imgs, torch.Tensor):
                np_imgs = to_numpy_array(imgs)
                if np_imgs.ndim == 2:
                    np_imgs = np_imgs[None, ...]
                if infer_channel_dimension_format(np_imgs) == ChannelDimension.LAST:
                    np_imgs = np_imgs.transpose(2, 0, 1)
                return torch.from_numpy(np_imgs.copy()).float()
            return imgs.float()

        data = {}
        # import ipdb
        # ipdb.set_trace()
        if images is not None:

            image_tensors = None
            if isinstance(images, torch.Tensor):
                if images.ndim == 3:
                    image_tensors = images.unsqueeze(0)  
                elif images.ndim == 4:
                    image_tensors = images  
                else:
                    raise ValueError(f"Unsupported tensor input with {images.ndim} dimensions.")
            else:
    
                images = make_batched_images(images)
                if do_convert_rgb:
                    images = [convert_to_rgb(image) for image in images]
                
                tensor_list = [_to_tensor(img) for img in images]


                if not tensor_list:
                    image_tensors = torch.empty(0)
                elif all(t.ndim == 3 for t in tensor_list):
                    image_tensors = torch.stack(tensor_list, dim=0) # [C,H,W] -> [B,C,H,W]
                elif all(t.ndim == 4 for t in tensor_list):
                    image_tensors = torch.cat(tensor_list, dim=0)   # [B,C,H,W] -> [Total_B,C,H,W]
                else:
                    raise ValueError("Inconsistent tensor dimensions in the input image list.")
            
            # Now call the differentiable _preprocess method
            pixel_values, image_grid_thw_list = self._preprocess(
                image_tensors,
                do_resize=do_resize,
                resample=resample,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                do_convert_rgb=do_convert_rgb,
                data_format=data_format,
                input_data_format=ChannelDimension.FIRST # We standardized it
            )
            # The grid is calculated based on the batch, let's assume it's the same for all
            # and format it as a list of tuples as the original code seems to do.
            vision_grid_thws = [image_grid_thw_list for _ in range(len(images))]
            data = {"pixel_values": pixel_values, "image_grid_thw": vision_grid_thws}

        if videos is not None:
            videos = make_batched_videos(videos)
            pixel_values_list, vision_grid_thws_list = [], []

            for video_frames in videos:
                if do_convert_rgb:
                    video_frames = [convert_to_rgb(frame) for frame in video_frames]
                
                video_tensor = torch.stack([_to_tensor(frame) for frame in video_frames])

                patches, video_grid_thw = self._preprocess(
                    video_tensor,
                    do_resize=do_resize,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=ChannelDimension.FIRST,
                )
                pixel_values_list.append(patches)
                vision_grid_thws_list.append(video_grid_thw)
            
            # Concatenate all patches from the batch of videos
            pixel_values = torch.cat(pixel_values_list, dim=0)
            data = {"pixel_values_videos": pixel_values, "video_grid_thw": vision_grid_thws_list}

        return BatchFeature(data=data, tensor_type=return_tensors)
__all__ = ["Qwen2VLImageProcessor"]

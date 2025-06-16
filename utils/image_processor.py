# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
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

from latentsync.utils.util import read_video, write_video
from torchvision import transforms
import cv2
from einops import rearrange
import torch
import numpy as np
from typing import Union
from .affine_transform import AlignRestore
from .face_detector import FaceDetector


def load_fixed_mask(resolution: int, mask_image_path="latentsync/utils/mask.png") -> torch.Tensor:
    mask_image = cv2.imread(mask_image_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.resize(mask_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4) / 255.0
    mask_image = rearrange(torch.from_numpy(mask_image), "h w c -> c h w")
    # 确保返回float类型的tensor
    mask_image = mask_image.float()
    print(f"Loaded mask_image: shape={mask_image.shape}, dtype={mask_image.dtype}, device={mask_image.device}")
    return mask_image


class ImageProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu", mask_image=None, 
                 face_detector_config=None):
        """
        Initialize ImageProcessor with configurable face detection parameters
        
        Args:
            resolution: target resolution
            device: compute device
            mask_image: mask image tensor
            face_detector_config: dict with face detection parameters like:
                {
                    'min_face_size': 40,
                    'min_face_height': 60, 
                    'aspect_ratio_range': (0.15, 2.0),
                    'detection_threshold': 0.4,
                    'debug': True
                }
        """
        self.resolution = resolution
        self.device = device  # 存储设备信息
        self.resize = transforms.Resize(
            (resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        )
        self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)

        self.restorer = AlignRestore(resolution=resolution, device=device)

        if mask_image is None:
            self.mask_image = load_fixed_mask(resolution)
        else:
            self.mask_image = mask_image
            
        # 确保mask_image在正确的设备上
        if device != "cpu":
            device_obj = torch.device(device)
            if self.mask_image.device != device_obj:
                self.mask_image = self.mask_image.to(device_obj)
                print(f"Moved mask_image to device: {device_obj}")

        if device == "cpu":
            self.face_detector = None
        else:
            # Configure face detector with user parameters
            if face_detector_config is None:
                face_detector_config = {
                    'min_face_size': 40,  # 降低最小人脸尺寸要求
                    'min_face_height': 60,  # 降低最小人脸高度要求
                    'aspect_ratio_range': (0.15, 2.0),  # 放宽宽高比要求
                    'debug': True
                }
            
            self.face_detector = FaceDetector(device=device, **face_detector_config)

    def affine_transform(self, image: torch.Tensor, max_retries=3, allow_no_face=True):
        """
        Apply affine transformation with multiple retry attempts and different thresholds
        
        Args:
            image: input image tensor
            max_retries: maximum number of detection attempts with different thresholds
            allow_no_face: if True, return None when no face detected instead of raising error
        """
        if self.face_detector is None:
            raise NotImplementedError("Using the CPU for face detection is not supported")
        
        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] == 3:  # CHW format
                image_np = image.permute(1, 2, 0).numpy()
            else:
                image_np = image.numpy()
        else:
            image_np = image
            
        # Ensure image is in correct format and range
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        
        print(f"Image shape: {image_np.shape}, dtype: {image_np.dtype}, range: [{image_np.min()}, {image_np.max()}]")
        
        # Try detection with progressively lower thresholds
        base_threshold = self.face_detector.detection_threshold
        thresholds = [base_threshold, 0.3, 0.2, 0.1]
        
        for attempt, threshold in enumerate(thresholds[:max_retries]):
            print(f"Face detection attempt {attempt + 1}/{max_retries} with threshold {threshold}")
            
            try:
                bbox, landmark_2d_106 = self.face_detector(image_np, threshold=threshold)
                
                if bbox is not None:
                    print(f"Face detected successfully on attempt {attempt + 1}")
                    break
                else:
                    print(f"No face detected on attempt {attempt + 1}")
                    
            except Exception as e:
                print(f"Error during face detection attempt {attempt + 1}: {str(e)}")
                
        if bbox is None:
            if allow_no_face:
                print("No face detected in this frame, will use original frame")
                return None, None, None
            else:
                # 提供更详细的错误信息和建议
                error_msg = (
                    f"Face not detected after {max_retries} attempts with thresholds {thresholds[:max_retries]}. "
                    f"Suggestions: 1) Ensure the image contains a clear, frontal face. "
                    f"2) Check image quality and lighting. "
                    f"3) Try preprocessing the image (resize, enhance contrast). "
                    f"4) Consider using a different frame from the video."
                )
                raise RuntimeError(error_msg)

        pt_left_eye = np.mean(landmark_2d_106[[43, 48, 49, 51, 50]], axis=0)  # left eyebrow center
        pt_right_eye = np.mean(landmark_2d_106[101:106], axis=0)  # right eyebrow center
        pt_nose = np.mean(landmark_2d_106[[74, 77, 83, 86]], axis=0)  # nose center

        landmarks3 = np.round([pt_left_eye, pt_right_eye, pt_nose])

        face, affine_matrix = self.restorer.align_warp_face(image_np.copy(), landmarks3=landmarks3, smooth=True)
        box = [0, 0, face.shape[1], face.shape[0]]  # x1, y1, x2, y2
        face = cv2.resize(face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
        face = rearrange(torch.from_numpy(face), "h w c -> c h w")
        # 确保face tensor是float类型
        face = face.float()
        return face, box, affine_matrix

    def preprocess_fixed_mask_image(self, image: torch.Tensor, affine_transform=False):
        if affine_transform:
            image, _, _ = self.affine_transform(image)
        else:
            image = self.resize(image)
        pixel_values = self.normalize(image / 255.0)
        
        # 确保mask_image与pixel_values在同一设备上
        mask_image = self.mask_image
        print(f"Debug: pixel_values device: {pixel_values.device}, mask_image device: {mask_image.device}")
        
        if pixel_values.device != mask_image.device:
            print(f"Device mismatch detected, moving mask_image from {mask_image.device} to {pixel_values.device}")
            mask_image = mask_image.to(pixel_values.device)
            
        masked_pixel_values = pixel_values * mask_image
        return pixel_values, masked_pixel_values, mask_image[0:1]

    def prepare_masks_and_masked_images(self, images: Union[torch.Tensor, np.ndarray], affine_transform=False):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")

        results = [self.preprocess_fixed_mask_image(image, affine_transform=affine_transform) for image in images]

        pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
        return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), torch.stack(masks_list)

    def process_images(self, images: Union[torch.Tensor, np.ndarray], device=None):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")
        images = self.resize(images)
        pixel_values = self.normalize(images / 255.0)
        
        # 如果指定了设备，确保tensor在正确的设备上
        if device is not None and pixel_values.device != device:
            pixel_values = pixel_values.to(device)
            
        return pixel_values


class VideoProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu"):
        self.image_processor = ImageProcessor(resolution, device)

    def affine_transform_video(self, video_path):
        video_frames = read_video(video_path, change_fps=False)
        results = []
        for frame in video_frames:
            frame, _, _ = self.image_processor.affine_transform(frame)
            results.append(frame)
        results = torch.stack(results)

        results = rearrange(results, "f c h w -> f h w c").numpy()
        return results


if __name__ == "__main__":
    video_processor = VideoProcessor(256, "cuda")
    video_frames = video_processor.affine_transform_video("assets/demo2_video.mp4")
    write_video("output.mp4", video_frames, fps=25)

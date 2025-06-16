"""
Standalone LatentSync ComfyUI Node Implementation
Complete independent package that doesn't require the full LatentSync project
"""

import os
import sys
import torch
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
from datetime import datetime

# Import local modules
from .config_manager import ConfigManager
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from accelerate.utils import set_seed
from DeepCache import DeepCacheSDHelper

# Import from local copies
from .models.unet import UNet3DConditionModel
from .pipelines.lipsync_pipeline import LipsyncPipeline
from .whisper.audio2feature import Audio2Feature


class LatentSyncStandaloneNode:
    """
    Standalone ComfyUI Node for LatentSync lip synchronization
    
    Features:
    - Completely independent deployment (no need for full LatentSync project)
    - Configurable model paths via config.json
    - Intelligent mixed face processing capability
    - Enhanced error handling and user guidance
    - Ready for open-source distribution
    """
    
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.dtype = torch.float16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7 else torch.float32
        
        # Initialize config manager
        self.config_manager = ConfigManager()
        
        # Ensure output directories exist
        self.config_manager.ensure_output_dirs()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "Path to input video file"
                }),
                "audio_path": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "Path to input audio file"
                }),
                "inference_steps": ("INT", {
                    "default": 20, 
                    "min": 10, 
                    "max": 100, 
                    "step": 1,
                    "tooltip": "Number of inference steps (higher = better quality, slower)"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 2.0, 
                    "min": 1.0, 
                    "max": 5.0, 
                    "step": 0.1,
                    "tooltip": "Guidance scale (higher = better lip-sync, may cause artifacts)"
                }),
                "seed": ("INT", {
                    "default": 1247, 
                    "min": -1, 
                    "max": 2**31-1,
                    "tooltip": "Random seed (-1 for random)"
                }),
            },
            "optional": {
                "face_detection_mode": (["lenient", "default", "very_lenient", "custom"], {
                    "default": "lenient",
                    "tooltip": "Face detection sensitivity mode"
                }),
                "min_face_size": ("INT", {
                    "default": 30, 
                    "min": 10, 
                    "max": 200,
                    "tooltip": "Minimum face width in pixels (only for custom mode)"
                }),
                "min_face_height": ("INT", {
                    "default": 50, 
                    "min": 20, 
                    "max": 300,
                    "tooltip": "Minimum face height in pixels (only for custom mode)"
                }),
                "detection_threshold": ("FLOAT", {
                    "default": 0.3, 
                    "min": 0.1, 
                    "max": 0.9, 
                    "step": 0.1,
                    "tooltip": "Face detection confidence threshold (only for custom mode)"
                }),
                "output_directory": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom output directory (optional, uses default if empty)"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("output_video_path", "statistics", "status", "model_info")
    FUNCTION = "process_video"
    CATEGORY = "LatentSync"
    
    def validate_model_paths(self) -> Tuple[bool, str]:
        """Validate that all required model paths are configured and exist"""
        model_status = self.config_manager.check_model_files()
        missing_models = [k for k, v in model_status.items() if not v]
        
        if missing_models:
            missing_str = ", ".join(missing_models)
            error_msg = f"""❌ Missing model files: {missing_str}

📝 Please configure model paths in config.json:
{self.config_manager.config_file}

Required models:
• unet_checkpoint: Path to latentsync_unet.pt
• whisper_model: Path to whisper tiny.pt or small.pt  
• auxiliary_models: Path to InsightFace auxiliary models directory
• vae_model: HuggingFace model ID (default: stabilityai/sd-vae-ft-mse)

💡 You can download models from:
- HuggingFace: https://huggingface.co/ByteDance/LatentSync-1.5
- Or use the original setup_env.sh script"""
            return False, error_msg
        
        return True, "✅ All model files validated successfully"
    
    def load_pipeline(self):
        """Load the LatentSync pipeline if not already loaded"""
        if self.pipeline is not None:
            return
            
        print("🔄 Loading LatentSync pipeline (standalone version)...")
        
        # Validate model paths first
        valid, msg = self.validate_model_paths()
        if not valid:
            raise RuntimeError(msg)
        
        # Load configuration
        self.config = self.config_manager.get_unet_config()
        
        # Get model paths
        try:
            checkpoint_path = self.config_manager.get_model_path("unet_checkpoint")
            whisper_model_path = self.config_manager.get_whisper_model_path()
            vae_model_id = self.config_manager.get_model_path("vae_model")
        except ValueError as e:
            raise RuntimeError(f"Configuration error: {e}")
        
        # Initialize scheduler
        scheduler_config_path = self.config_manager.get_scheduler_config_path()
        scheduler = DDIMScheduler.from_pretrained(scheduler_config_path)
        
        # Initialize audio encoder
        audio_encoder = Audio2Feature(
            model_path=whisper_model_path,
            device="cuda",
            num_frames=self.config["data"]["num_frames"],
            audio_feat_length=self.config["data"]["audio_feat_length"],
        )
        
        # Initialize VAE
        vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=self.dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        
        # Initialize UNet
        unet, _ = UNet3DConditionModel.from_pretrained(
            self.config["model"],
            checkpoint_path,
            device="cpu",
        )
        unet = unet.to(dtype=self.dtype)
        
        # Create pipeline
        self.pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        ).to("cuda")
        
        # Enable DeepCache for acceleration
        if self.config_manager.settings["performance"]["enable_deepcache"]:
            helper = DeepCacheSDHelper(pipe=self.pipeline)
            helper.set_params(
                cache_interval=self.config_manager.settings["performance"]["cache_interval"],
                cache_branch_id=self.config_manager.settings["performance"]["cache_branch_id"]
            )
            helper.enable()
        
        print("✅ LatentSync pipeline loaded successfully (standalone)!")
    
    def get_face_detection_config(self, mode: str, min_face_size: int = 30, 
                                 min_face_height: int = 50, detection_threshold: float = 0.3):
        """Get face detection configuration based on mode"""
        if mode == "custom":
            return {
                'min_face_size': min_face_size,
                'min_face_height': min_face_height,
                'aspect_ratio_range': (0.1, 3.0),
                'detection_threshold': detection_threshold,
                'debug': True
            }
        else:
            return self.config_manager.get_face_detection_config(mode)
    
    def get_model_info(self) -> str:
        """Get information about loaded models"""
        try:
            model_status = self.config_manager.check_model_files()
            info_lines = ["📊 Model Information:"]
            
            for model_type, exists in model_status.items():
                status = "✅" if exists else "❌"
                path = self.config_manager.settings["model_paths"].get(model_type, "Not configured")
                info_lines.append(f"{status} {model_type}: {path}")
            
            info_lines.append(f"🔧 Cross attention dim: {self.config_manager.settings['model_settings']['cross_attention_dim']}")
            info_lines.append(f"📐 Resolution: {self.config_manager.settings['model_settings']['resolution']}")
            info_lines.append(f"🎞️ Frames: {self.config_manager.settings['model_settings']['num_frames']}")
            
            return "\n".join(info_lines)
        except Exception as e:
            return f"❓ Model info unavailable: {e}"
    
    def process_video(self, video_path: str, audio_path: str, inference_steps: int = 20,
                     guidance_scale: float = 2.0, seed: int = 1247, 
                     face_detection_mode: str = "lenient", min_face_size: int = 30,
                     min_face_height: int = 50, detection_threshold: float = 0.3,
                     output_directory: str = ""):
        """
        Process video for lip synchronization
        
        Args:
            video_path: Path to input video
            audio_path: Path to input audio
            inference_steps: Number of inference steps
            guidance_scale: Guidance scale for generation
            seed: Random seed
            face_detection_mode: Face detection sensitivity mode
            min_face_size: Minimum face size (custom mode only)
            min_face_height: Minimum face height (custom mode only)
            detection_threshold: Detection threshold (custom mode only)
            output_directory: Custom output directory
            
        Returns:
            Tuple of (output_video_path, statistics, status, model_info)
        """
        
        try:
            # Validate inputs
            if not os.path.exists(video_path):
                return "", "", f"❌ Error: Video file not found: {video_path}", ""
            
            if not os.path.exists(audio_path):
                return "", "", f"❌ Error: Audio file not found: {audio_path}", ""
            
            # Load pipeline
            self.load_pipeline()
            
            # Set seed
            if seed != -1:
                set_seed(seed)
            else:
                torch.seed()
            
            # Get face detection config
            face_config = self.get_face_detection_config(
                face_detection_mode, min_face_size, min_face_height, detection_threshold
            )
            
            # Determine output directory
            if output_directory and os.path.isdir(output_directory):
                output_dir = Path(output_directory)
            else:
                # Use ComfyUI's output directory by default
                comfyui_output_dir = Path(self.config_manager.node_dir).parent.parent.parent / "output"
                if comfyui_output_dir.exists():
                    output_dir = comfyui_output_dir
                else:
                    # Fallback to node's output directory
                    output_dir = Path(self.config_manager.node_dir) / self.config_manager.settings["paths"]["output_dir"]
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = Path(video_path).stem
            final_output = output_dir / f"{video_name}_latentsync_{timestamp}.mp4"
            
            # Create temporary output file
            temp_dir = tempfile.mkdtemp()
            temp_output = os.path.join(temp_dir, "output.mp4")
            
            print(f"🎬 Starting LatentSync processing (standalone)...")
            print(f"📹 Video: {video_path}")
            print(f"🎵 Audio: {audio_path}")
            print(f"🎯 Face detection mode: {face_detection_mode}")
            print(f"⚙️ Settings: steps={inference_steps}, guidance={guidance_scale}, seed={torch.initial_seed()}")
            print(f"📁 Output: {final_output}")
            
            # Process video
            self.pipeline(
                video_path=video_path,
                audio_path=audio_path,
                video_out_path=temp_output,
                video_mask_path=temp_output.replace(".mp4", "_mask.mp4"),
                num_frames=self.config["data"]["num_frames"],
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                weight_dtype=self.dtype,
                width=self.config["data"]["resolution"],
                height=self.config["data"]["resolution"],
            )
            
            # Move output to final location
            shutil.move(temp_output, final_output)
            
            # Clean up temp directory
            shutil.rmtree(temp_dir)
            
            # Generate statistics
            statistics = f"""🎬 LatentSync Processing Complete (Standalone)!
📹 Input Video: {Path(video_path).name}
🎵 Input Audio: {Path(audio_path).name}
📁 Output: {final_output.name}
⚙️ Settings: {inference_steps} steps, guidance {guidance_scale}
🎯 Face Detection: {face_detection_mode} mode
🌱 Seed: {torch.initial_seed()}
🔧 Enhanced with mixed face processing
📦 Standalone deployment version"""
            
            status = "✅ Success: Video processed successfully with LatentSync standalone node!"
            model_info = self.get_model_info()
            
            print(status)
            return str(final_output), statistics, status, model_info
            
        except Exception as e:
            error_msg = f"❌ Error during LatentSync processing: {str(e)}"
            print(error_msg)
            
            # Provide specific error guidance
            if "Configuration error" in str(e) or "not configured" in str(e):
                error_msg += f"\n\n💡 Please check configuration file: {self.config_manager.config_file}"
            elif "Face not detected" in str(e):
                error_msg += "\n\n💡 Suggestions:\n• Try 'very_lenient' face detection mode\n• Ensure video contains clear, visible faces\n• Check video quality and lighting"
            elif "Model" in str(e) and "not found" in str(e):
                error_msg += "\n\n💡 Download models from: https://huggingface.co/ByteDance/LatentSync-1.5"
            
            model_info = self.get_model_info()
            return "", "", error_msg, model_info
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-process (could be optimized to check file timestamps)
        return float("NaN")


# For compatibility with older ComfyUI versions
if __name__ == "__main__":
    # Test the node
    node = LatentSyncStandaloneNode()
    print("LatentSync Standalone ComfyUI Node created successfully!")
    print(f"Config file: {node.config_manager.config_file}")
    print("Please configure model paths in config.json before use.") 
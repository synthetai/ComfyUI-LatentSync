"""
Configuration Manager for Standalone LatentSync ComfyUI Node
Handles model paths and settings without relying on external project structure
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Any
from omegaconf import OmegaConf


class ConfigManager:
    """
    Manages configuration for standalone LatentSync ComfyUI node
    """
    
    def __init__(self):
        self.node_dir = Path(__file__).parent
        self.config_file = self.node_dir / "config.json"
        self.settings = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json"""
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_file}\n"
                f"Please ensure config.json exists in the node directory."
            )
    

    
    def get_model_path(self, model_type: str) -> str:
        """Get model path for specified type"""
        paths = self.settings.get("model_paths", {})
        path = paths.get(model_type, "")
        
        if not path:
            raise ValueError(f"Model path for '{model_type}' not configured. Please check config.json")
        
        # Don't modify URLs or HuggingFace model IDs
        if path.startswith(("http://", "https://")):
            return path
        
        # Check if it's a HuggingFace model ID (contains "/" but doesn't start with "./" or "/")
        if "/" in path and not path.startswith(("./", "/")):
            # This is likely a HuggingFace model ID like "stabilityai/sd-vae-ft-mse"
            return path
        
        # Convert relative paths to absolute
        if not os.path.isabs(path):
            path = str(self.node_dir / path)
        
        return path
    
    def get_unet_config(self) -> Dict[str, Any]:
        """Get UNet configuration"""
        # Load from embedded config files
        config_path = self.node_dir / "configs" / "unet" / "stage2.yaml"
        if config_path.exists():
            config = OmegaConf.load(config_path)
            return OmegaConf.to_container(config)
        else:
            # Fallback to basic config
            return {
                "model": {
                    "cross_attention_dim": self.settings["model_settings"]["cross_attention_dim"],
                    "attention_head_dim": 64,
                    "in_channels": 4,
                    "out_channels": 4,
                    "down_block_types": [
                        "CrossAttnDownBlock3D",
                        "CrossAttnDownBlock3D", 
                        "CrossAttnDownBlock3D",
                        "DownBlock3D"
                    ],
                    "up_block_types": [
                        "UpBlock3D",
                        "CrossAttnUpBlock3D",
                        "CrossAttnUpBlock3D",
                        "CrossAttnUpBlock3D"
                    ],
                    "block_out_channels": [320, 640, 1280, 1280],
                    "layers_per_block": 2,
                    "norm_num_groups": 32
                },
                "data": {
                    "num_frames": self.settings["model_settings"]["num_frames"],
                    "audio_feat_length": self.settings["model_settings"]["audio_feat_length"],
                    "resolution": self.settings["model_settings"]["resolution"]
                }
            }
    
    def get_face_detection_config(self, mode: str = None) -> Dict[str, Any]:
        """Get face detection configuration"""
        if mode is None:
            mode = self.settings["face_detection"]["default_mode"]
        
        presets = self.settings["face_detection"]["presets"]
        return presets.get(mode, presets["lenient"])
    
    def get_scheduler_config_path(self) -> str:
        """Get scheduler configuration path"""
        return str(self.node_dir / "configs")
    
    def ensure_output_dirs(self):
        """Ensure output directories exist"""
        paths = self.settings.get("paths", {})
        
        for dir_key in ["output_dir", "temp_dir"]:
            dir_path = paths.get(dir_key, f"./{dir_key}")
            if not os.path.isabs(dir_path):
                dir_path = self.node_dir / dir_path
            
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def check_model_files(self) -> Dict[str, bool]:
        """Check if all required model files exist"""
        results = {}
        model_paths = self.settings.get("model_paths", {})
        
        for model_type, path in model_paths.items():
            if model_type == "vae_model" and path.startswith(("http://", "https://", "stabilityai/")):
                # HuggingFace model, assume available
                results[model_type] = True
            elif path:
                # Use get_model_path to resolve relative paths correctly
                try:
                    resolved_path = self.get_model_path(model_type)
                    results[model_type] = os.path.exists(resolved_path)
                except ValueError:
                    results[model_type] = False
            else:
                results[model_type] = False
        
        return results
    
    def get_whisper_model_path(self) -> str:
        """Get appropriate whisper model path based on cross_attention_dim"""
        cross_attention_dim = self.settings["model_settings"]["cross_attention_dim"]
        
        if cross_attention_dim == 768:
            return self.get_model_path("whisper_small")
        elif cross_attention_dim == 384:
            return self.get_model_path("whisper_model")
        else:
            raise ValueError(f"Unsupported cross_attention_dim: {cross_attention_dim}")


# Global config manager instance
config_manager = ConfigManager() 
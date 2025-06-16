"""
LatentSync ComfyUI Node - Standalone Version
Enhanced lip synchronization with intelligent mixed face processing
"""

from .latentsync_node import LatentSyncStandaloneNode

# ComfyUI node mappings
NODE_CLASS_MAPPINGS = {
    "LatentSyncStandaloneNode": LatentSyncStandaloneNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentSyncStandaloneNode": "LatentSync Lip Sync (Standalone)",
}

# Version info
__version__ = "1.0.0"
__author__ = "Enhanced by synthetai"
__license__ = "Apache 2.0"

print(f"🎬 LatentSync ComfyUI Node (Standalone) v{__version__} loaded successfully!")
print("📦 This is an independent package that can be deployed without the full LatentSync project.")
print("📘 Enhanced with mixed face processing - handles videos with and without faces intelligently.")

# Web info for potential open-source release
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"] 
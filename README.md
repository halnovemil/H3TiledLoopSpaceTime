H3TiledLoopSpaceTime

A ComfyUI custom node for tiled loop space-time processing.

Installation

Open Command Prompt (CMD)
Navigate to your ComfyUI custom nodes folder:
cd c:\comfy\custom_nodes
Clone the repository:
git clone https://github.com/halnovemil/H3TiledLoopSpaceTime.git
Restart ComfyUI.

Usage Recommendations

Use only as the Second Sampler Stage.
Use any workflow of your preference.

Recommended Settings

| Parameter       | Recommended Value              | Notes                                      |
|----------------|--------------------------------|--------------------------------------------|
| Temporal Tile  | 97 ~ 113 frames                | Best quality range                         |
| Overlap        | ~40%                           |                                            |
| Spatial        | 2x2                            | MAX VRAM Saving mode                       |
| Denoise        | Max 0.40                       | Keep low for good results                  |

Feel free to experiment with different values.

Enjoy!

Example VIDS is here >> https://huggingface.co/hal9000ace/H3TilingExampleVids/tree/main

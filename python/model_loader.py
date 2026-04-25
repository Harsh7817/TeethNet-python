import os
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Default to the small model; overridden by DEPTH_CHECKPOINT env if set
_CHECKPOINT = os.environ.get("DEPTH_CHECKPOINT", "/models/depth-anything-small-hf")

_MODEL = None
_PROCESSOR = None
_DEVICE = None

def get_model_and_processor():
    global _MODEL, _PROCESSOR, _DEVICE
    if _MODEL is None or _PROCESSOR is None:
        print("Loading model:", _CHECKPOINT, flush=True)
        # Strongly limit CPU threads to avoid WSL2/Docker oversubscription
        try:
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        except Exception:
            pass

        _PROCESSOR = AutoImageProcessor.from_pretrained(_CHECKPOINT)
        _MODEL = AutoModelForDepthEstimation.from_pretrained(_CHECKPOINT)

        if torch.cuda.is_available():
            _DEVICE = torch.device("cuda")
        else:
            _DEVICE = torch.device("cpu")

        _MODEL = _MODEL.to(_DEVICE)
        _MODEL.eval()
    return _MODEL, _PROCESSOR, _DEVICE
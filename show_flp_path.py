import pathlib
import sys

_VENDOR_ROOT = pathlib.Path(__file__).parent / "vendor" / "FasterLivePortrait"
sys.path.insert(0, str(_VENDOR_ROOT))
try:
    from src.pipelines import faster_live_portrait_pipeline as mod

    print(mod.__file__)
except ImportError as e:
    print(f"Import failed: {e}")
    print(f"sys.path: {sys.path}")

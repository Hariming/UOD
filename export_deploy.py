#!/usr/bin/env python3
import argparse
from uod.engine import export_checkpoint

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Save BFE/IFE/IFD only; removes NFE AND optimizer states")
    p.add_argument("--checkpoint", required=True); p.add_argument("--output", required=True)
    a = p.parse_args()
    count = export_checkpoint(a.checkpoint, a.output)
    print(f"Saved {a.output}: {count:,} state elements, NO NFE keys. This is a PyTorch state_dict, not ONNX.")

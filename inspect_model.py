#!/usr/bin/env python3
import argparse
import torch
from uod.engine import read_config, build_spec
from uod.model import UOD

if __name__ == "__main__":
    p=argparse.ArgumentParser(description="Inspect shared architecture and feature shapes without downloading weights")
    p.add_argument("--config", default="configs/uod_resnet50.yaml")
    p.add_argument("--size", type=int, default=128)
    a=p.parse_args(); cfg=read_config(a.config)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model=UOD(build_spec(cfg), pretrained=False)
    for name in ("bfe","ife","ifd","nfe_rgb","nfe_thermal"):
        module=getattr(model,name)
        print(f"{name:12s} {sum(p.numel() for p in module.parameters()):,} parameters")
    with torch.no_grad():
        out=model.forward_pair(torch.rand(1,3,a.size,a.size),torch.rand(1,1,a.size,a.size))
    for key in ("intrinsic_rgb","intrinsic_thermal","non_rgb","non_thermal"):
        print(key,[list(x.shape) for x in out[key]])
    print("Classification logits:",[list(x.shape) for x in out['pred_rgb'].cls_logits])
    print("Regression l/t/r/b:",[list(x.shape) for x in out['pred_rgb'].distances])
    print("\nIFE/NFE architecture:\n",model.ife)
    print("\nDecoupled IFD:\n",model.ifd)

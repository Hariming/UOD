#!/usr/bin/env python3
"""Checks file pairing, label validity and split leakage. Alignment needs visual inspection."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from uod.data import assert_disjoint
from uod.engine import read_config, build_dataset

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True); p.add_argument("--previews",type=int,default=8)
    p.add_argument("--output",default="data_audit")
    a=p.parse_args(); cfg=read_config(a.config)
    train,val=build_dataset(cfg,"train"),build_dataset(cfg,"val")
    assert_disjoint(train,val)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    report={}
    for split,ds in (("train",train),("val",val)):
        counts=[0]*len(cfg['data']['class_names']); empty=0
        for i in range(len(ds)):
            sample=ds[i]
            for label in sample['target']['labels']: counts[int(label)]+=1
            empty+=int(len(sample['target']['labels'])==0)
            if i < a.previews:
                left=(sample['rgb'].permute(1,2,0).numpy()*255).astype(np.uint8)
                right=(sample['thermal'][0].numpy()*255).astype(np.uint8)
                right=np.repeat(right[...,None],3,axis=2)
                image=Image.fromarray(np.concatenate([left,right],axis=1)); draw=ImageDraw.Draw(image)
                width=left.shape[1]
                for box,label in zip(sample['target']['boxes'],sample['target']['labels'],strict=True):
                    b=box.tolist()
                    draw.rectangle(b,outline=(255,220,0),width=2)
                    draw.rectangle([b[0]+width,b[1],b[2]+width,b[3]],outline=(255,220,0),width=2)
                image.save(out/f'{split}_{i:04d}.png')
        report[split]={'pairs':len(ds),'negative_pairs':empty,'class_counts':counts}
    report['warning']='Checks cannot prove geometric/time registration. Inspect previews; calibrate/register upstream.'
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

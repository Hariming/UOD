#!/usr/bin/env python3
"""Create toy ALIGNED pairs to test the pipeline; NOT a detection benchmark."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def generate(output: str | Path, train_pairs: int=8, val_pairs: int=4):
    root=Path(output); root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(2026)
    for split,count in (('train',train_pairs),('val',val_pairs)):
        directory=root/split
        for part in ('rgb','thermal','labels'): (directory/part).mkdir(parents=True,exist_ok=True)
        lines=[]
        for i in range(count):
            h,w=96,128
            rgb=Image.new('RGB',(w,h),(25,30,40)); thermal=Image.new('L',(w,h),25)
            draw_r,draw_t=ImageDraw.Draw(rgb),ImageDraw.Draw(thermal)
            labels=[]
            if i%7!=6:  # include negative pairs
                cls=i%2
                x1=int(rng.integers(8,36)); y1=int(rng.integers(8,24))
                x2=x1+int(rng.integers(38,65)); y2=y1+int(rng.integers(35,61))
                color=(170,85,50) if cls==0 else (45,120,195)
                draw_r.rectangle([x1,y1,x2,y2],fill=color,outline=(220,220,220))
                draw_t.rectangle([x1,y1,x2,y2],fill=180+20*cls,outline=240)
                labels=[f"{cls} {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}"]
            rgb.save(directory/'rgb'/f'{i:04d}.png'); thermal.save(directory/'thermal'/f'{i:04d}.png')
            (directory/'labels'/f'{i:04d}.txt').write_text('\n'.join(labels),encoding='utf-8')
            lines.append(json.dumps({'id':f'{split}_{i:04d}','scene_id':f'{split}_scene_{i}',
                 'rgb':f'{split}/rgb/{i:04d}.png','thermal':f'{split}/thermal/{i:04d}.png',
                 'label':f'{split}/labels/{i:04d}.txt'}))
        (root/f'{split}.jsonl').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return root

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',default='demo_data')
    a=p.parse_args(); print('Created',generate(a.output))

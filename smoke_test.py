#!/usr/bin/env python3
"""Offline CPU test: toy training, exact resume, NFE-free export, validation, 3-mode inference.
This is an execution test, not evidence of real-data accuracy.
"""
from __future__ import annotations
import argparse, copy, json, os, subprocess, sys, tempfile, time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--report',default='smoke_report.json')
    a=p.parse_args(); root=Path(__file__).resolve().parent
    env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    os.environ.update({'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
    import torch, yaml
    from make_demo_data import generate
    from uod.engine import load_checkpoint, export_checkpoint, load_detector, build_dataset, make_loader, evaluate
    from uod.geometry import candidates, suppress, fuse_candidates
    torch.set_num_threads(2); begin=time.monotonic()
    def run(*args):
        command=[sys.executable,*map(str,args)]
        result=subprocess.run(command,cwd=root,env=env,text=True,capture_output=True,timeout=600)
        print(result.stdout,end='',flush=True)
        if result.returncode:
            raise RuntimeError(result.stderr+'\nFailed: '+' '.join(command))
    with tempfile.TemporaryDirectory(prefix='uod_smoke_') as td:
        temp=Path(td); data=generate(temp/'data')
        cfg=yaml.safe_load((root/'configs/smoke.yaml').read_text())
        cfg['data'].update(train_manifest=str(data/'train.jsonl'),val_manifest=str(data/'val.jsonl'))
        cfg['training'].update(out_dir=str(temp/'full'),save_every=1)
        config_path=temp/'full.yaml'; config_path.write_text(yaml.safe_dump(cfg))
        run('train.py','--config',config_path,'--device','cpu')
        for path in ('epoch_002.pt','epoch_003.pt','best.pt'): (temp/'full'/path).unlink(missing_ok=True)
        resume=copy.deepcopy(cfg); resume['training'].update(out_dir=str(temp/'resume'),save_every=10)
        resume_path=temp/'resume.yaml'; resume_path.write_text(yaml.safe_dump(resume))
        run('train.py','--config',resume_path,'--device','cpu','--resume',temp/'full/epoch_001.pt')
        full=load_checkpoint(temp/'full/last.pt')['state_dict']
        resumed=load_checkpoint(temp/'resume/last.pt')['state_dict']
        unequal=[key for key in full if not torch.equal(full[key],resumed[key])]
        if unequal: raise AssertionError(f'CPU epoch-boundary resume mismatch: {unequal[:3]}')
        del full,resumed
        export_checkpoint(temp/'full/last.pt',temp/'deploy.pt')
        deployed=load_checkpoint(temp/'deploy.pt')
        assert not any(k.startswith('nfe_') for k in deployed['state_dict']) and 'optimizer' not in deployed
        model,_=load_detector(temp/'deploy.pt',torch.device('cpu'))
        ds=build_dataset(cfg,'val',False)
        loader=make_loader(ds,2,0,False,torch.device('cpu'))
        after=evaluate(model,loader,torch.device('cpu'),cfg.get('postprocess',{}))
        before=json.loads((temp/'full/val_epoch_003.json').read_text())
        if before!=after: raise AssertionError('Deployment validation differs')
        item=ds[0]
        with torch.no_grad():
            rgb=model(item['rgb'][None]); thermal=model(item['thermal'][None])
            rr,tt=model.paired_predictions(item['rgb'][None],item['thermal'][None])
        for x,y in zip(rgb.cls_logits,rr.cls_logits,strict=True):
            torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-5)  # batch convolution rounding may differ
        for x,y in zip(thermal.cls_logits,tt.cls_logits,strict=True):
            torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-5)
        r=candidates(rr,(128,128),item['valid_mask'][None],.001)[0]
        t=candidates(tt,(128,128),item['valid_mask'][None],.001)[0]
        for result in (suppress(r),suppress(t),fuse_candidates(r,t)):
            assert torch.isfinite(result['boxes']).all()
        report={'status':'PASS','device':'cpu','torch':str(torch.__version__),
            'checks':['3-epoch toy training','exact epoch-boundary resume','NFE-free export',
                      'deployment validation parity','RGB inference','thermal inference','RGBT concat+NMS inference'],
            'seconds':time.monotonic()-begin,
            'limitations':'Synthetic data only. CUDA/AMP, real-data accuracy and pretrained downloads were not tested.'}
        path=Path(a.report); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()

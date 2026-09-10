"""Numerical and end-to-end unit tests; no pretrained weights/network required."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch
from uod.model import ModelSpec, UOD, UODDetector, model_spec_dict
from uod.losses import FeatureSeparationLoss, UODLoss, DetectionLoss
from uod.data import (PairedDetectionDataset, assert_disjoint, read_thermal,
                      letterbox_tensor, transform_boxes)
from uod.geometry import undo_letterbox, fuse_candidates
from uod.engine import atomic_save, cpu_state_dict, export_checkpoint, load_detector, load_checkpoint
from uod.metrics import DetectionAP
from make_demo_data import generate


def small_model():
    torch.manual_seed(41)
    return UOD(ModelSpec(2, backbone='resnet18', channels=32, head_depth=1, normalization='none'))


def vec(a, b):
    return [torch.tensor([a, b], dtype=torch.float32).reshape(1, 2, 1, 1)]


class FeatureLossTests(unittest.TestCase):
    def test_separated_features_zero_loss(self):
        loss, _ = FeatureSeparationLoss()(vec(1,0),vec(1,0),vec(0,1),vec(0,-1))
        self.assertAlmostEqual(loss.item(), 0., places=6)

    def test_positive_term_counted_twice(self):
        # Positive D^2=2; all repulsive terms already satisfy margin=1.
        loss, stats = FeatureSeparationLoss()(vec(1,0),vec(0,1),vec(-1,0),vec(0,-1))
        self.assertAlmostEqual(loss.item(), 4., places=6)
        self.assertAlmostEqual(stats['positive_d2'].item(), 2., places=6)

    def test_normalization_is_channelwise_not_global(self):
        loss, _ = FeatureSeparationLoss()(vec(20,0),vec(0,3),vec(-8,0),vec(0,-9))
        self.assertAlmostEqual(loss.item(), 4., places=6)

    def test_collapse_is_not_formal_disentanglement(self):
        # This also documents that feature distance alone doesn't guarantee MI=0.
        loss, _ = FeatureSeparationLoss()(vec(1,0),vec(1,0),vec(1,0),vec(1,0))
        self.assertAlmostEqual(loss.item(), 3., places=6)

    def test_margin_is_threshold_on_squared_distance(self):
        loss, _ = FeatureSeparationLoss(margin=3)(vec(1,0),vec(1,0),vec(0,1),vec(0,-1))
        self.assertAlmostEqual(loss.item(), 2., places=6)

    def test_shapes_and_margin_fail_fast(self):
        with self.assertRaises(ValueError): FeatureSeparationLoss(margin=5)
        with self.assertRaises(ValueError):
            FeatureSeparationLoss()(vec(1,0),vec(1,0),[torch.ones(1,3,1,1)],vec(0,1))

    def test_padding_mask_option(self):
        # Same first pixel; orthogonal second pixel. N features safely separated.
        ir=torch.tensor([[[[1.,1.]],[[0.,0.]]]])
        it=torch.tensor([[[[1.,0.]],[[0.,1.]]]])
        nr=-ir; nt=-it
        mask=torch.tensor([[[[1,0]]]],dtype=torch.bool)
        masked,_=FeatureSeparationLoss(mask_padding=True)([ir],[it],[nr],[nt],mask)
        full,_=FeatureSeparationLoss()([ir],[it],[nr],[nt])
        self.assertLess(masked.item(),full.item())


class ModelTests(unittest.TestCase):
    def test_shared_modules_called_once_for_pair(self):
        model=small_model(); counts={'bfe':0,'ife':0,'ifd':0}
        handles=[]
        for name in counts:
            def hook(m, inp, out, key=name): counts[key]+=1
            handles.append(getattr(model,name).register_forward_hook(hook))
        with torch.no_grad(): model.forward_pair(torch.rand(2,3,128,128),torch.rand(2,1,128,128))
        for handle in handles: handle.remove()
        self.assertEqual(counts,{'bfe':1,'ife':1,'ifd':1})

    def test_nfe_architecture_equal_but_parameters_independent(self):
        m=small_model()
        shape=lambda module: {n:tuple(p.shape) for n,p in module.named_parameters()}
        self.assertEqual(shape(m.ife),shape(m.nfe_rgb)); self.assertEqual(shape(m.ife),shape(m.nfe_thermal))
        ids=lambda module: {p.data_ptr() for p in module.parameters()}
        self.assertFalse(ids(m.ife)&ids(m.nfe_rgb))
        self.assertFalse(ids(m.nfe_rgb)&ids(m.nfe_thermal))
        self.assertTrue(any(not torch.equal(p,q) for p,q in zip(m.ife.parameters(),m.nfe_rgb.parameters(),strict=True)))
        self.assertFalse(ids(m.ifd.cls_tower)&ids(m.ifd.reg_tower))

    def test_thermal_replication_and_deployment_parity(self):
        m=small_model().eval(); deploy=m.to_deploy(); thermal=torch.rand(1,1,128,128)
        with torch.no_grad():
            a=m(thermal); b=m(thermal.repeat(1,3,1,1)); c=deploy(thermal)
        for x,y,z in zip(a.cls_logits,b.cls_logits,c.cls_logits,strict=True):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
            torch.testing.assert_close(x,z,rtol=0,atol=0)
        self.assertFalse(any(n.startswith('nfe_') for n,_ in deploy.named_modules()))
        self.assertFalse(any(n.startswith('nfe_') for n in deploy.state_dict()))

    def test_yolov5_style_backbone_and_neck_shapes(self):
        spec=ModelSpec(2,backbone='yolov5n',neck='yolov5_pafpn',channels=32,head_depth=1,normalization='none')
        m=UOD(spec)
        with torch.no_grad():
            out=m.forward_pair(torch.rand(1,3,128,128),torch.rand(1,1,128,128))
        self.assertEqual([tuple(x.shape) for x in out['intrinsic_rgb']],
                         [(1,32,16,16),(1,32,8,8),(1,32,4,4)])
        self.assertEqual([tuple(x.shape) for x in out['pred_rgb'].cls_logits],
                         [(1,3,16,16),(1,3,8,8),(1,3,4,4)])
        self.assertFalse(any(n.startswith('nfe_') for n,_ in m.to_deploy().named_modules()))

    def test_backward_reaches_all_five_modules(self):
        m=small_model(); out=m.forward_pair(torch.rand(2,3,128,128),torch.rand(2,1,128,128))
        targets=[{'boxes':torch.tensor([[20.,20.,85.,110.]]),'labels':torch.tensor([i%2])} for i in range(2)]
        loss,_=UODLoss(2)(out,targets,(128,128),torch.ones(2,1,128,128,dtype=torch.bool))
        self.assertTrue(torch.isfinite(loss)); loss.backward()
        for name in ('bfe','ife','ifd','nfe_rgb','nfe_thermal'):
            total=sum(p.grad.abs().sum().item() for p in getattr(m,name).parameters() if p.grad is not None)
            self.assertGreater(total,0,name)
        torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True)
        opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
        before=m.ifd.cls_pred.weight.detach().clone(); opt.step()
        self.assertFalse(torch.equal(before,m.ifd.cls_pred.weight))

    def test_empty_gt_safe(self):
        m=small_model(); out=m.forward_pair(torch.rand(1,3,128,128),torch.rand(1,1,128,128))
        targets=[{'boxes':torch.empty(0,4),'labels':torch.empty(0,dtype=torch.long)}]
        loss,stats=UODLoss(2)(out,targets,(128,128),torch.ones(1,1,128,128,dtype=torch.bool))
        self.assertTrue(torch.isfinite(loss)); self.assertEqual(stats['rgb/l1'].item(),0.)
        loss.backward()

    def test_export_reload_exact(self):
        m=small_model().eval()
        with tempfile.TemporaryDirectory() as td:
            full,deploy=Path(td)/'full.pt',Path(td)/'deploy.pt'
            atomic_save({'format_version':1,'kind':'uod_training','model_spec':model_spec_dict(m),
                         'class_names':['a','b'],'preprocessing':{'image_size':128,'thermal_divisor':255.,'letterbox_fill':114/255},
                         'state_dict':cpu_state_dict(m),'epoch':0},full)
            export_checkpoint(full,deploy)
            ckpt=load_checkpoint(deploy)
            self.assertEqual(ckpt['kind'],'uod_deploy')
            self.assertFalse(any(k.startswith('nfe_') for k in ckpt['state_dict']))
            self.assertNotIn('optimizer',ckpt)
            loaded,_=load_detector(deploy,torch.device('cpu'))
            x=torch.rand(1,3,128,128)
            with torch.no_grad(): a,b=m(x),loaded(x)
            for p,q in zip(a.distances,b.distances,strict=True): torch.testing.assert_close(p,q,rtol=0,atol=0)


class DataAndMetricTests(unittest.TestCase):
    def test_paired_flip_is_synchronized(self):
        with tempfile.TemporaryDirectory() as td:
            root=generate(td,2,1)
            base=PairedDetectionDataset(root/'train.jsonl',2,128,augment=False)[0]
            flip=PairedDetectionDataset(root/'train.jsonl',2,128,augment=True,flip_probability=1.)[0]
            torch.testing.assert_close(base['rgb'].flip(-1),flip['rgb'])
            torch.testing.assert_close(base['thermal'].flip(-1),flip['thermal'])
            box=base['target']['boxes'].clone(); x1=box[:,0].clone()
            box[:,0]=128-box[:,2]; box[:,2]=128-x1
            torch.testing.assert_close(box,flip['target']['boxes'])
            with self.assertRaises(ValueError):
                ds=PairedDetectionDataset(root/'train.jsonl',2,128)
                assert_disjoint(ds,ds)

    def test_letterbox_coordinate_roundtrip(self):
        _,_,meta=letterbox_tensor(torch.rand(3,83,117),128)
        boxes=torch.tensor([[10.,12.,80.,70.]])
        torch.testing.assert_close(undo_letterbox(transform_boxes(boxes,meta),meta),boxes)

    def test_uint16_not_silently_clipped(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'thermal.png'
            Image.fromarray(np.full((20,30),40000,dtype=np.uint16)).save(path)
            with self.assertRaises(ValueError): read_thermal(path,255)
            result=read_thermal(path,65535)
            self.assertAlmostEqual(result.mean().item(),40000/65535,places=5)

    def test_rgbt_fuses_before_class_aware_nms(self):
        r={'boxes':torch.tensor([[0.,0.,20.,20.]]),'scores':torch.tensor([.8]),'labels':torch.tensor([0])}
        t={'boxes':torch.tensor([[1.,1.,21.,21.],[0.,0.,20.,20.]]),
           'scores':torch.tensor([.9,.7]),'labels':torch.tensor([0,1])}
        fused=fuse_candidates(r,t,.5,100)
        self.assertEqual(len(fused['boxes']),2)
        self.assertEqual(set(fused['labels'].tolist()),{0,1})
        self.assertAlmostEqual(fused['scores'][0].item(),.9,places=6)

    def test_ap_perfect_and_undefined_classes(self):
        metric=DetectionAP(2)
        gt={'boxes':torch.tensor([[10.,10.,30.,30.]]),'labels':torch.tensor([0])}
        pred={**gt,'scores':torch.tensor([.9])}
        metric.update([pred],[gt]); result=metric.compute()
        self.assertAlmostEqual(result['map50'],1.)
        self.assertAlmostEqual(result['map50_95'],1.)
        self.assertIsNone(result['per_class']['1']['ap50'])


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)

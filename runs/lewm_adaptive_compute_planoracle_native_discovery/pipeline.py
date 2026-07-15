#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.metadata, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent; REPO=ROOT.parents[1]; PRIOR=REPO/'runs/lewm_adaptive_compute_distribution_contract'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def dump(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n')
def load_prior():
 sys.path.insert(0,str(PRIOR))
 import common, generator, generator_seedfix
 return common,generator,generator_seedfix
def verify_pre():
 seal=json.loads((ROOT/'audit/pre_generation_seal.json').read_text())
 for p,e in seal['files'].items():
  if sha(REPO/p)!=e: raise RuntimeError(f'pre-generation seal drift: {p}')
 return seal

def generate(role):
 verify_pre()
 if role=='prospective':
  subprocess_check=[sys.executable,str(ROOT/'freeze/verify_gate_freeze.py')]
  import subprocess; subprocess.run(subprocess_check,check=True)
 ledger=json.loads((ROOT/'cohort_seed_ledger.json').read_text()); specs=ledger['roles'][role]
 outdir=ROOT/'data'/f'{role}_raw'; outdir.mkdir(exist_ok=True)
 manifest_path=ROOT/'data'/f'{role}_raw_manifest.json'
 if manifest_path.exists():
  m=json.loads(manifest_path.read_text());
  for x in m['episodes']:
   if sha(REPO/x['path'])!=x['sha256']: raise RuntimeError('raw drift')
  return m
 common,generator,seedfix=load_prior(); world=policy=None; records=[]; started=time.time()
 try:
  world,policy=generator.make_world('plan_oracle')
  for i,s in enumerate(specs):
   p=outdir/f"{s['episode_id']}.npz"
   if p.exists(): raise RuntimeError(f'unmanifested preexisting episode {p}')
   arrays,audit=seedfix.generate_episode_seedfixed(world,policy,phase=role,policy_type='plan_oracle',slot=s['slot'],trajectory_id=s['episode_id'],env_seed=s['env_seed'],policy_seed=s['policy_seed'],oracle_np_seed=s['oracle_np_seed'])
   # Preserve complete raw evidence; no contact field is ever exposed to fitting/evaluation.
   np.savez_compressed(p,**arrays)
   records.append({**s,'path':str(p.relative_to(REPO)),'sha256':sha(p),'bytes':p.stat().st_size,'initial_state_sha256':audit['initial_state_sha256'],'reset_amendment':audit['environment_reset_seed_forwarding_bug_corrected']})
   print(f'generated {role} {i+1}/{len(specs)}',flush=True)
 finally:
  if world is not None: world.close()
 vers={}
 for n in ('stable-worldmodel','ogbench','mujoco','gymnasium','numpy'):
  try: vers[n]=importlib.metadata.version(n)
  except: vers[n]=None
 m={'schema_version':1,'complete':True,'role':role,'episode_count':len(records),'episodes':records,'versions':vers,'DGP':'sealed PlanOracle DGP with deterministic reset amendment','elapsed_seconds':time.time()-started,'contact_used':False,'v3_or_hdf5_opened':False,'v5_episodes':0}
 dump(manifest_path,m); return m

def encode(role,device_name):
 verify_pre(); os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
 import torch
 common,_,_=load_prior(); runtime=common.load_runtime(); model_io=common.load_model_io(); device=runtime.choose_device(device_name)
 manifest=json.loads((ROOT/'data'/f'{role}_raw_manifest.json').read_text()); base,contract,prov=runtime.load_base_model(device); solver,v1,models=runtime.load_solver(device)
 before=runtime.module_audit(base,solver,v1); H=[];A=[];T=[];E=[];F=[]; ids=[]; steps=[]
 for pos,rec in enumerate(manifest['episodes']):
  p=REPO/rec['path'];
  if sha(p)!=rec['sha256']: raise RuntimeError('raw input drift')
  with np.load(p,allow_pickle=False) as z: pixels=z['pixels'][::5]; actions=z['action'][:200].astype(np.float32)
  latent=[]
  for j in range(0,41,64):
   x=model_io.pixel_transform(pixels[j:j+64],contract.image_size,device)
   latent.append(base.encode({'pixels':x.unsqueeze(0)})['emb'].squeeze(0).cpu())
  latent=torch.cat(latent).numpy().astype(np.float32); norm=(actions-common.FROZEN_ACTION_MEAN)/common.FROZEN_ACTION_STD; blocks=norm.reshape(40,25)
  h=np.stack([latent[j:j+3] for j in range(38)]).astype(np.float32); a=np.stack([blocks[j:j+3] for j in range(38)]).astype(np.float32); t=latent[3:].astype(np.float32)
  ht=torch.as_tensor(h,device=device); at=torch.as_tensor(a,device=device); b=runtime.base_predict(base,ht,at); outputs,updates=solver(ht,at,b,max_depth=4,return_updates=True)
  exits=torch.stack([outputs[d] for d in (1,2,3,4)],1).cpu().numpy().astype(np.float32)
  feats=[]
  for stage,d in enumerate((1,2,3)):
   feats.append(models.build_causal_features(ht,at,outputs[d],updates[d]).cpu().numpy().astype(np.float32))
  H.append(h); A.append(a); T.append(t); E.append(exits); F.append(np.stack(feats,1)); ids.append(np.full(38,pos,np.int32)); steps.append(np.arange(3,41,dtype=np.int16))
  print(f'encoded/evaluated {role} {pos+1}/{len(manifest["episodes"])}',flush=True)
 runtime.synchronize(device); after=runtime.module_audit(base,solver,v1)
 if before!=after or not before['passed']: raise RuntimeError('frozen module isolation failed')
 out=ROOT/'data'/f'{role}_evaluated.npz'; np.savez_compressed(out,target=np.concatenate(T),exits=np.concatenate(E),features=np.concatenate(F),episode_id=np.concatenate(ids),model_step=np.concatenate(steps))
 m={'schema_version':1,'role':role,'path':str(out.relative_to(REPO)),'sha256':sha(out),'rows':len(manifest['episodes'])*38,'base_provenance':prov,'model_contract':vars(contract),'module_before':before,'module_after':after,'contact_fields_loaded':False,'history_and_actions_not_persisted_after_feature_construction':True}
 dump(ROOT/'data'/f'{role}_evaluated_manifest.json',m); return m

def main():
 p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True); g=sub.add_parser('generate');g.add_argument('role',choices=('smoke','fit','selection','prospective')); e=sub.add_parser('encode');e.add_argument('role',choices=('smoke','fit','selection','prospective'));e.add_argument('--device',default='auto'); a=p.parse_args()
 print(json.dumps(generate(a.role) if a.cmd=='generate' else encode(a.role,a.device),sort_keys=True,default=str))
if __name__=='__main__': main()

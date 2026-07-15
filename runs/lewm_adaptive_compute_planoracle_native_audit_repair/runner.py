#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;REPO=R.parents[1];DIST=REPO/'runs/lewm_adaptive_compute_distribution_contract'
ALLOW={'pixels','action'}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n')
def verify():subprocess.run([sys.executable,str(R/'verify_pre_data.py')],check=True,stdout=subprocess.DEVNULL)
def modules():
 sys.path.insert(0,str(DIST));import common,generator,generator_seedfix
 return common,generator,generator_seedfix
def generate(role):
 verify();ledger=json.loads((R/'cohort_seed_ledger.json').read_text());specs=ledger['roles'][role];out=R/'data'/f'{role}_raw';out.mkdir(exist_ok=True);mp=R/'data'/f'{role}_raw_manifest.json'
 if mp.exists():
  m=json.loads(mp.read_text())
  if len(m['episodes'])!=len(specs) or any(sha(REPO/x['path'])!=x['sha256'] for x in m['episodes']):raise RuntimeError('existing raw manifest invalid')
  return m
 common,gen,fix=modules();world=policy=None;recs=[]
 try:
  world,policy=gen.make_world('plan_oracle')
  for i,s in enumerate(specs):
   p=out/f"{s['episode_id']}.npz"
   if p.exists():raise RuntimeError('unmanifested raw file')
   a,audit=fix.generate_episode_seedfixed(world,policy,phase=role,policy_type='plan_oracle',slot=s['slot'],trajectory_id=s['episode_id'],env_seed=s['env_seed'],policy_seed=s['policy_seed'],oracle_np_seed=s['oracle_np_seed'])
   np.savez_compressed(p,**a);recs.append({**s,'path':str(p.relative_to(REPO)),'sha256':sha(p),'bytes':p.stat().st_size,'initial_state_sha256':audit['initial_state_sha256'],'reset_amendment':audit['environment_reset_seed_forwarding_bug_corrected']});print(f'generated {role} {i+1}/{len(specs)}',flush=True)
 finally:
  if world is not None:world.close()
 m={'schema_version':1,'complete':True,'role':role,'episode_count':len(recs),'episodes':recs,'input_allowlist':sorted(ALLOW),'contact_or_privileged_loaded':False,'v3_test_targets_opened':False,'hdf5_opened':False,'v5_episodes':0};dump(mp,m);return m
def execute(role,device_name):
 verify();os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1');import torch
 common,_,_=modules();runtime=common.load_runtime();model_io=common.load_model_io();device=runtime.choose_device(device_name);base,contract,prov=runtime.load_base_model(device);solver,v1,models=runtime.load_solver(device)
 before=runtime.module_audit(base,solver,v1);rawm=json.loads((R/'data'/f'{role}_raw_manifest.json').read_text());G=np.load(R/'freeze/compiled_gate.npz');ar=torch.as_tensor(G['a_raw'],device=device);aw=torch.as_tensor(G['a_white'],device=device);br=torch.as_tensor(G['b_raw'],device=device);bw=torch.as_tensor(G['b_white'],device=device);thr=float(G['threshold'])
 targets=[];dense=[];sparse=[];calls=[];scores=[];features=[];ids=[];steps=[];lat=[]
 for ei,rec in enumerate(rawm['episodes']):
  p=REPO/rec['path'];
  if sha(p)!=rec['sha256']:raise RuntimeError('raw drift')
  with np.load(p,allow_pickle=False) as z:
   if not ALLOW.issubset(z.files):raise RuntimeError('allowlisted input absent')
   pixels=z['pixels'][::5];actions=z['action'][:200].astype(np.float32)
  latent=[]
  for j in range(0,41,64):latent.append(base.encode({'pixels':model_io.pixel_transform(pixels[j:j+64],contract.image_size,device).unsqueeze(0)})['emb'].squeeze(0).cpu())
  latent=torch.cat(latent).numpy().astype(np.float32);norm=(actions-common.FROZEN_ACTION_MEAN)/common.FROZEN_ACTION_STD;blocks=norm.reshape(40,25);h=np.stack([latent[j:j+3] for j in range(38)]);a=np.stack([blocks[j:j+3] for j in range(38)]);t=latent[3:]
  ht=torch.as_tensor(h,device=device);at=torch.as_tensor(a,device=device);b=runtime.base_predict(base,ht,at);outs,ups=solver(ht,at,b,max_depth=4,return_updates=True);den=torch.stack([outs[d] for d in (1,2,3,4)],1)
  anchored=solver.anchor(ht,at,b);cur=anchored[1];upd=cur-b;active_idx=torch.arange(len(ht),device=device);sp=cur.clone();cc=torch.ones(len(ht),dtype=torch.long,device=device);sc=torch.full((len(ht),3),float('nan'),device=device);ff=torch.full((len(ht),3,1046),float('nan'),device=device)
  for stage,adapter in enumerate(solver.adapters):
   if not len(active_idx):break
   ah=ht.index_select(0,active_idx);aa=at.index_select(0,active_idx);ac=cur.index_select(0,active_idx);au=upd.index_select(0,active_idx);feat=models.build_causal_features(ah,aa,ac,au);depth=torch.zeros((len(feat),3),dtype=feat.dtype,device=device);depth[:,stage]=1;x=torch.cat((feat,depth),1);score=torch.minimum(x@ar+br,x@aw+bw);sc[active_idx,stage]=score;ff[active_idx,stage]=feat;keep=score>thr;stop=active_idx[~keep]
   if len(stop):sp[stop]=cur[stop]
   active_idx=active_idx[keep]
   if not len(active_idx):break
   pre=cur.index_select(0,active_idx);u=adapter(ht.index_select(0,active_idx),at.index_select(0,active_idx),pre);cur=cur.index_copy(0,active_idx,pre+u);upd=upd.index_copy(0,active_idx,u);cc[active_idx]+=1
  if len(active_idx):sp[active_idx]=cur[active_idx]
  sel=den[torch.arange(len(ht),device=device),cc-1]
  if not torch.equal(sp,sel):raise RuntimeError('sparse/dense selected output mismatch')
  targets.append(t);dense.append(den.cpu().numpy());sparse.append(sp.cpu().numpy());calls.append(cc.cpu().numpy());scores.append(sc.cpu().numpy());features.append(ff.cpu().numpy());ids.append(np.full(38,ei,np.int32));steps.append(np.arange(3,41,dtype=np.int16));print(f'executed {role} {ei+1}/{len(rawm["episodes"])}',flush=True)
 runtime.synchronize(device);after=runtime.module_audit(base,solver,v1)
 if before!=after or not before['passed']:raise RuntimeError('module drift')
 out=R/'data'/f'{role}_execution.npz';np.savez_compressed(out,target=np.concatenate(targets),dense_exits=np.concatenate(dense),sparse_selected=np.concatenate(sparse),calls=np.concatenate(calls),scores=np.concatenate(scores),features=np.concatenate(features),episode_id=np.concatenate(ids),model_step=np.concatenate(steps))
 m={'schema_version':1,'role':role,'path':str(out.relative_to(REPO)),'sha256':sha(out),'rows':len(ids)*38,'sparse_dense_exact':True,'module_before':before,'module_after':after,'base_provenance':prov,'loaded_input_keys':sorted(ALLOW),'contact_or_privileged_loaded':False,'no_gradients':True};dump(R/'data'/f'{role}_execution_manifest.json',m);return m
def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True);g=s.add_parser('generate');g.add_argument('role',choices=('smoke','prospective'));e=s.add_parser('execute');e.add_argument('role',choices=('smoke','prospective'));e.add_argument('--device',default='auto');a=p.parse_args();print(json.dumps(generate(a.role) if a.cmd=='generate' else execute(a.role,a.device),default=str))
if __name__=='__main__':main()

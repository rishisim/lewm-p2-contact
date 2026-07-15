from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import critics
import models
import policy


class TinyV1(nn.Module):
    def __init__(self):
        super().__init__(); self.latent_dim=4; self.action_dim=2; self.history_len=3; self.max_depth=4
        self.iteration_embedding=nn.Embedding(4,2)
        self.block=nn.Sequential(nn.Linear(24,8),nn.GELU(),nn.Linear(8,8),nn.GELU(),nn.Linear(8,4))
        nn.init.zeros_(self.block[-1].weight); nn.init.zeros_(self.block[-1].bias)
    def forward(self,h,a,z,depths=(0,1,2,4)):
        requested=tuple(sorted(set(depths))); out={0:z} if 0 in requested else {}; current=z
        for k in range(max(requested)):
            emb=self.iteration_embedding(torch.full((len(z),),k,dtype=torch.long))
            current=current+self.block(torch.cat((h.flatten(1),a.flatten(1),current,emb),1))
            if k+1 in requested: out[k+1]=current
        return out


class SmokeProtocolTests(unittest.TestCase):
    def test_solver_critic_policy_exact_call_pipeline(self):
        torch.manual_seed(3); rng=np.random.default_rng(3); n=36
        h=torch.randn(n,3,4); a=torch.randn(n,3,2); z=torch.randn(n,4)
        solver=models.StagewiseResidualCascade(TinyV1(),later_stages=1,hidden_dim=7)
        with torch.no_grad(): solver.adapters[0].network[-1].bias.fill_(0.05)
        solver.eval().requires_grad_(False)
        outputs,updates=solver(h,a,z,return_updates=True)
        target=outputs[2].detach()+torch.from_numpy(rng.normal(scale=.01,size=(n,4)).astype(np.float32))
        feature=models.build_causal_features(h,a,outputs[1],updates[1]).numpy()
        losses=np.stack([((outputs[d]-target).square().mean(1)).numpy() for d in (1,2)],1)
        gain=losses[:,0]-losses[:,1]; episodes=np.repeat(np.arange(6),6)
        fitted,oof=critics.fit_cross_fitted_critics(feature,gain,episodes,folds=2,seeds=[7,8],hidden_dims=[8],epochs=2,batch_size=12,lr=1e-3,weight_decay=0,device=torch.device('cpu'))
        self.assertTrue(np.isfinite(oof).all())
        mean,_=critics.ensemble_scores(fitted,feature,torch.device('cpu'))
        calibrated=policy.calibrate_compute_price(mean[:,None],1.5,exit_calls=(1,2))
        selected=policy.causal_sequential_stopping(mean[:,None],compute_price=calibrated['compute_price'],exit_calls=(1,2))
        baseline=policy.strongest_transition_independent_baseline(losses,(1,2),n=n,target_total_calls=int(selected.sum()),seed=9)
        self.assertEqual(int(baseline['selected_calls'].sum()),int(selected.sum()))
        _,stats=solver.forward_selected(h,a,z,torch.from_numpy(selected),return_stats=True)
        self.assertEqual(stats.processed_rows,int(selected.sum()))


if __name__ == '__main__': unittest.main()

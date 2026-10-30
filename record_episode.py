"""Record a public LIBERO episode directly, without the competition pipeline.

This port is syntax checked; GPU/simulator end-to-end verification is pending.
"""
import argparse
import hashlib
import json
from pathlib import Path

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bddl',type=Path,required=True)
    p.add_argument('--initial-states',type=Path,required=True)
    p.add_argument('--index',type=int,default=1)
    p.add_argument('--instruction',required=True)
    p.add_argument('--interest',nargs='+',required=True,help='Exact BDDL object-of-interest names; do not omit to improve the metric')
    p.add_argument('--mode',choices=['resvla','stationary_control'],default='resvla')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--base-vlm',type=Path)
    p.add_argument('--seed',type=int,default=43)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Use a new output directory; existing outputs are never overwritten')
    if args.mode=='resvla' and not (args.checkpoint and args.base_vlm):p.error('Model paths required')
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from libero.libero.envs import OffScreenRenderEnv
    # Only load trusted initial-state files: historical torch files may use pickle.
    states=torch.load(args.initial_states,weights_only=False)
    state=states[args.index]
    policy=None
    if args.mode=='resvla':
        from resvla_policy import Policy
        policy=Policy(args.checkpoint,args.base_vlm,args.seed)
    env=OffScreenRenderEnv(bddl_file_name=str(args.bddl),camera_heights=128,camera_widths=128)
    args.output.mkdir(parents=True)
    try:
        env.reset();env.sim.set_state_from_flattened(state);env.sim.forward()
        zero=np.zeros(env.robots[0].action_dim)
        for _ in range(10):obs,_,_,_=env.step(zero)
        reference={k[:-4]:np.asarray(v).copy() for k,v in obs.items() if k.endswith('_pos')
                   and not k.startswith('robot0') and not k.endswith('_to_robot0_eef_pos')
                   and k[:-4] not in args.interest}
        maximum={k:0. for k in reference};crossings={};trace=[];done=False
        with imageio.get_writer(str(args.output/'rollout.mp4'),fps=20,macro_block_size=1) as writer:
            def capture():writer.append_data(np.ascontiguousarray(np.concatenate(
                [obs['agentview_image'][::-1],obs['robot0_eye_in_hand_image'][::-1]],axis=1)))
            capture()
            for step in range(1,301):
                action=policy.act(obs,args.instruction) if policy else zero.copy()
                obs,reward,done,_=env.step(action)
                disp={k:float(np.abs(obs[k+'_pos']-v).sum()) for k,v in reference.items()}
                for k,v in disp.items():
                    maximum[k]=max(maximum[k],v)
                    if v>.001 and k not in crossings:crossings[k]=step
                contacts=[[env.sim.model.geom_id2name(c.geom1),env.sim.model.geom_id2name(c.geom2)]
                          for c in env.sim.data.contact[:env.sim.data.ncon]]
                trace.append(dict(step=step,action=action.tolist(),eef_pos=obs['robot0_eef_pos'].tolist(),
                    displacement_l1=disp,contacts=contacts,reward=float(reward),done=bool(done)))
                capture()
                if done and policy:break
        result=dict(task=args.bddl.stem,mode=args.mode,initial_state_index=args.index,
            request_seed=args.seed if policy else None,steps=len(trace),done=bool(done),
            success=bool(done) and not crossings,first_crossing=crossings,max_displacement_l1_m=maximum,
            bddl_sha256=sha(args.bddl),initial_states_sha256=sha(args.initial_states),
            checkpoint_sha256=sha(args.checkpoint) if policy else None,
            interest=args.interest,port_status='New standalone port; not the historical recording executable')
        for name,data in [('result.json',result),('trace.json',trace)]:
            (args.output/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
    finally:env.close()

if __name__=='__main__':main()

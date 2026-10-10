import collections,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from flycade.snapshot import load_snapshot
from flycade.game import GameEnv,GameConfig,ACTIONS
from flycade.nes import NesEmulator

torch.set_num_threads(1)
root=Path.cwd();run=root/'reports/five-minute-001/run';out=root/'reports/five-minute-001/comparison'
summary=json.loads((run.parent/'summary.json').read_text());group=summary['catalog']['protocols'][0]
protocol=json.loads((run/'initial-evaluation-protocol.json').read_text())
original=[11,22,33];seeds=original+list(range(101,121));results=[]
protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [run/'initial.pt',run/'final.pt',run/'latest.json']}
for result in group['results']:
 update=result['snapshot_updates'];policy,meta,manifest=load_snapshot(run,result['snapshot_id'],root/'.flycade','cpu')
 em=NesEmulator(root/'.flycade')
 with GameEnv(em,GameConfig(**protocol['game'])) as env:
  episodes=[]
  for seed in seeds:
   random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);sampler=torch.Generator(device='cpu').manual_seed(seed)
   obs,_=env.reset(seed=seed);tail=collections.deque(maxlen=50);shots=collections.deque(maxlen=8);actions=collections.Counter();step=0;releases=0;prior_a=False
   while True:
    with torch.no_grad():
     dist,_=policy(torch.from_numpy(obs).unsqueeze(0));action=int(torch.multinomial(dist.probs.cpu(),1,generator=sampler).item())
    probabilities=dist.probs[0].tolist();ram=em.env.get_ram();state=int(ram[0x1d]);y=int(ram[0xb5])*256+int(ram[0xce]);jump='A' in ACTIONS[action]
    releases+=int(prior_a and not jump);prior_a=jump;actions[action]+=1;step+=1
    if seed in original and update:
     shots.append((step,env.last_frame.copy()))
    obs,_,term,trunc,info=env.step(action)
    tail.append({'step':step,'frame':env.frames,'position':info['position'],'y':y,'state':state,'action':action,'probabilities':probabilities,'reason':info['reason']})
    if term or trunc:break
   row={'seed':seed,'distance':info['max_progress'],'reason':info['reason'],'frames':env.frames,'actions':dict(actions),'jump_releases':releases}
   episodes.append(row)
   if seed in original:
    index=original.index(seed);assert row['distance']==result['distances'][index],(update,seed,row,result['distances'])
    (out/f'update-{update}-seed-{seed}-tail.json').write_text(json.dumps(list(tail),indent=2))
    if shots:
     for n,frame in [shots[0],shots[-1]]:Image.fromarray(frame).save(out/f'update-{update}-seed-{seed}-step-{n}.png')
  results.append({'updates':update,'snapshot':meta['snapshot_id'],'episodes':episodes})
for path,before in protected.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==before
aggregate=[]
for result in results:
 subsets={}
 for name,selected in [('original',original),('additional',seeds[3:]),('all',seeds)]:
  rows=[r for r in result['episodes'] if r['seed'] in selected];d=[r['distance'] for r in rows]
  subsets[name]={'count':len(rows),'mean':float(np.mean(d)),'median':float(np.median(d)),'min':min(d),'max':max(d),'endings':dict(collections.Counter(r['reason'] for r in rows)),'distances':d}
 aggregate.append({'updates':result['updates'],'subsets':subsets})
a=[r['distance'] for r in results[1]['episodes'][3:]];b=[r['distance'] for r in results[2]['episodes'][3:]]
comparison={'later_better':sum(y>x for x,y in zip(a,b)),'later_worse':sum(y<x for x,y in zip(a,b)),'ties':sum(y==x for x,y in zip(a,b)),'mean_paired_delta':float(np.mean(np.array(b)-np.array(a)))}
evidence={'results':results,'aggregate':aggregate,'paired_250_vs_307_additional':comparison,'protected_weights_unchanged':True,'limitations':['Same deterministic level; seeds vary action sampling, not maps.','20 additional fixed seeds; exploratory comparison, not a generalization claim.']}
(out/'evidence.json').write_text(json.dumps(evidence,indent=2))
print(json.dumps({'aggregate':aggregate,'paired':comparison},indent=2),flush=True)

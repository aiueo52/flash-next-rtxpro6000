"""Contribution policies for prune_policy_sim; CPU-only, immutable original routes.

Global and joint thresholds are trained on first-half complete verify steps.
Layer thresholds match a common expected removed score per layer/token row on
that same training split. Test rows include zero-removal rows in all quantiles.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from prune_policy_sim import Replay, EXPERTS


@dataclass
class Capture:
    path: Path
    ids: np.ndarray
    w: np.ndarray
    norm: np.ndarray
    h: np.ndarray
    layer: np.ndarray
    call: np.ndarray
    split: np.ndarray
    reconstruction: np.ndarray

    @property
    def width(self): return self.ids.shape[1]
    @property
    def score(self): return self.w.astype(np.float64) * self.norm / self.h[..., None]
    def subset(self, mask):
        return Capture(self.path, *[getattr(self,k)[mask] for k in
            ('ids','w','norm','h','layer','call','split','reconstruction')])


def load_capture(path):
    with np.load(path,allow_pickle=False) as z:
        if int(z['schema_version'])!=1 or str(z['role'])!='target_verify':
            raise ValueError('requires P3 schema v1 target_verify capture')
        if int(z['total_calls'])!=len(z['call']): raise ValueError('ring overflow: incomplete workload')
        required={'call','T','k','layer','ids','w','expert_norm','h_norm','input_norm',
                  'singleton','p2_drop','reconstruction_rel','unchanged','total_calls','method'}
        missing=required-set(z.files)
        if missing: raise ValueError(f'missing schema keys: {sorted(missing)}')
        n=len(z['call']); widths=np.unique(z['T']); ks=np.unique(z['k'])
        if n==0 or len(widths)!=1 or len(ks)!=1 or widths[0] not in (4,16):
            raise ValueError('requires nonempty fixed W4/W16 and top-k')
        width,k=int(widths[0]),int(ks[0])
        if int(z['total_calls'])!=n: raise ValueError('ring overflow: incomplete workload')
        order=np.argsort(z['call'],kind='stable')
        call=z['call'][order]; layer=z['layer'][order]
        if not np.array_equal(call,np.arange(n)): raise ValueError('missing/repeated call')
        if n%48 or not np.array_equal(layer,np.tile(np.arange(48),n//48)):
            raise ValueError('expected complete ordered 48-layer target-verify steps')
        def routes(key): return z[key][order,:width*k].reshape(n,width,k).copy()
        ids,w,norm=routes('ids'),routes('w'),routes('expert_norm')
        h=z['h_norm'][order].copy()
        if h.shape!=(n,width) or np.any(~np.isfinite(h)) or np.any(h<=0): raise ValueError('invalid residual norm')
        if np.any(~np.isfinite(norm)) or np.any(norm<0): raise ValueError('invalid expert norm')
        input_norm=z['input_norm'][order]
        if input_norm.shape!=(n,width) or np.any(~np.isfinite(input_norm)) or np.any(input_norm<=0):
            raise ValueError('invalid MoE input norm')
        if not z['unchanged'].all(): raise ValueError('recorder modified serving tensor/routing')
        r=Replay(ids,w)
        if not np.array_equal(routes('singleton'),r.mult==1): raise ValueError('singleton flags disagree')
        if not np.array_equal(routes('p2_drop'),baseline_mask(r)): raise ValueError('P2 flag mismatch')
        rec=z['reconstruction_rel'][order].copy()
        if np.any(~np.isfinite(rec)) or np.max(rec)>.025: raise ValueError('reconstruction check failed')
    # Whole verify steps: no shared layer/token rows across temporal split.
    steps=n//48
    if steps<4: raise ValueError('need >=4 steps for temporal holdout')
    split=call//48 >= steps//2
    return Capture(Path(path),ids,w,norm,h,layer,call,split,rec)


def baseline_mask(replay):
    drop=(replay.mult==1)&(replay.w<float(np.float32(.08)))
    np.put_along_axis(drop,replay.top,False,axis=2)
    return drop


class ContributionReplay:
    def __init__(self,capture,score=None):
        self.data=capture
        self.r=Replay(capture.ids,capture.w)
        self.true_score=capture.score
        self.score=np.asarray(self.true_score if score is None else score,dtype=np.float64)
        if self.score.shape!=capture.ids.shape or np.any(~np.isfinite(self.score)) or np.any(self.score<0):
            raise ValueError('invalid contribution score')
        self.top=np.zeros_like(self.score,dtype=bool)
        np.put_along_axis(self.top,self.r.top,True,axis=2)
        self.single=(self.r.mult==1)&~self.top
        # All route statistics refer to the ORIGINAL call, including protected
        # top-1 routes. An expert with either protected route cannot be removed.
        size=len(capture.ids)*EXPERTS
        group_cost=np.bincount(self.r.keys.ravel(),weights=self.score.ravel(),minlength=size)
        protected=np.bincount(self.r.keys[self.top],minlength=size)>0
        self.joint=((self.r.mult==1)|(self.r.mult==2))&~protected[self.r.keys]
        self.group_score=group_cost[self.r.keys]

    def drop(self,kind,threshold):
        if kind=='baseline': return baseline_mask(self.r)
        if kind=='singleton': return self.single&(self.score<threshold)
        if kind=='layer':
            thresholds=np.asarray(threshold)
            if thresholds.shape!=(48,): raise ValueError('layer thresholds must have 48 entries')
            return self.single&(self.score<thresholds[self.data.layer,None,None])
        if kind=='joint': return self.joint&(self.group_score<threshold)
        raise ValueError(kind)

    def stats(self,drop):
        # Verify complete groups rather than silently assigning fractional D.
        n=len(self.r.ids)
        selected_keys=self.r.keys[drop]
        cnt=np.bincount(selected_keys,minlength=n*EXPERTS)
        if np.any(cnt[selected_keys]!=self.r.mult[drop]): raise ValueError('partial expert removal')
        removed=(drop/self.r.mult).sum(axis=(1,2))
        row=(self.true_score*drop).sum(2)
        return dict(D=float((self.r.distinct-removed).mean()),mean=float(row.mean()),
                    p95=float(np.percentile(row,95)),p99=float(np.percentile(row,99)))


def layer_cost_curves(replays):
    curves=[]
    for layer in range(48):
        costs=[];rows=0
        for r in replays:
            mask=r.data.layer==layer
            rows+=int(mask.sum())*r.data.width
            costs.append(r.score[mask][r.single[mask]])
        values=np.concatenate(costs)
        unique,counts=np.unique(values,return_counts=True)
        cumulative=np.cumsum(unique*counts)/max(rows,1)
        curves.append((unique,cumulative))
    return curves


def calibrated_layers(replays,budget,curves=None):
    """Common training mean sum(s_removed) per token row; strict whole-tie cut."""
    if curves is None:
        curves=layer_cost_curves(replays)
    thresholds=np.zeros(48)
    for layer,(unique,cumulative) in enumerate(curves):
        n=int(np.searchsorted(cumulative,budget,side='right'))
        thresholds[layer]=np.nextafter(unique[n-1],np.inf) if n else 0.
    return thresholds


def fit_proxy(captures):
    """Training-only mean expert output norm, shrunk with 8 layer-mean samples."""
    sums=np.zeros((48,EXPERTS)); counts=np.zeros_like(sums)
    for c in captures:
        keys=c.layer[:,None,None]*EXPERTS+c.ids
        sums+=np.bincount(keys.ravel(),weights=c.norm.ravel(),minlength=48*EXPERTS).reshape(48,EXPERTS)
        counts+=np.bincount(keys.ravel(),minlength=48*EXPERTS).reshape(48,EXPERTS)
    mean=sums.sum(1)/np.maximum(counts.sum(1),1)
    table=(sums+8*mean[:,None])/(counts+8)
    return table,counts


def proxy_score(c,table):
    return c.w*table[c.layer[:,None,None],c.ids]/c.h[...,None]


def selection_recall(truth,pred, mult):
    # Whole-expert recall, with a pair counted once, and route recall.
    route_denom=truth.sum()
    expert_denom=(truth/mult).sum()
    hit=truth&pred
    return dict(route_recall=float(hit.sum()/route_denom) if route_denom else None,
        expert_recall=float((hit/mult).sum()/expert_denom) if expert_denom else None,
        precision=float((hit/mult).sum()/(pred/mult).sum()) if pred.any() else None)


def matched_threshold(r,kind,target_removed):
    """Match number of removed experts on held-out inputs for ranking diagnosis.

    This is an oracle diagnostic, not a deployable held-out calibration.
    """
    costs=r.score[r.single] if kind=='singleton' else r.group_score[r.joint]
    values=np.unique(costs)
    lo,hi=0,len(values)
    while lo<hi:
        mid=(lo+hi)//2
        drop=r.drop(kind,np.nextafter(values[mid],np.inf))
        removed=(drop/r.r.mult).sum()/len(r.r.ids)
        if removed>=target_removed: hi=mid
        else: lo=mid+1
    return float(np.nextafter(values[min(lo,len(values)-1)],np.inf)) if len(values) else 0.


def analyze(paths,output,report):
    captures=[load_capture(p) for p in paths]
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    rows=[]; proxy_rows=[]; selected=[]; calibration={}; completeness=[]; layer_rows=[]
    for width in (4,16):
        cs=[c for c in captures if c.width==width]
        expected={'code-edit','prose-en','prose-ja','agent-loop'}
        names=[c.path.stem.split(f'-w{width}-')[-1] for c in cs]
        if len(cs)!=4 or set(names)!=expected:
            raise ValueError(f'W{width}: expected each of four workloads exactly once, got {names}')
        train=[c.subset(~c.split) for c in cs]
        tr=[ContributionReplay(c) for c in train]
        table,counts=fit_proxy(train)
        np.savez_compressed(output/f'proxy-w{width}.npz',norm_table=table,counts=counts)
        loo_tables=[fit_proxy(train[:i]+train[i+1:])[0] for i in range(len(train))]
        np.savez_compressed(output/f'proxy-loo-w{width}.npz',norm_tables=np.stack(loo_tables),
                            excluded_workloads=np.asarray(names))
        single=np.concatenate([r.score[r.single] for r in tr])
        joint=np.concatenate([r.group_score[r.joint] for r in tr])
        # Dense training-derived sweep; selection uses held-out diagnostic
        # frontier, so best-per-workload is explicitly an oracle candidate.
        grid=np.linspace(0,1,101)
        th_single=np.unique(np.r_[0,np.nextafter(np.quantile(single,grid),np.inf)])
        th_joint=np.unique(np.r_[0,np.nextafter(np.quantile(joint,grid),np.inf)])
        layer_max=max(sum((r.score[r.data.layer==l]*r.single[r.data.layer==l]).sum() for r in tr)/
            sum((r.data.layer==l).sum()*width for r in tr) for l in range(48))
        budgets=np.linspace(0,layer_max,101)
        curves=layer_cost_curves(tr)
        layers=[calibrated_layers(tr,b,curves) for b in budgets]
        policies=[('baseline',0.08)]
        policies += [('singleton',float(t)) for t in th_single]
        policies += [('joint',float(t)) for t in th_joint]
        policies += [('layer',t) for t in layers]
        calibration[str(width)]=dict(singleton=th_single.tolist(),joint=th_joint.tolist(),
            budgets=budgets.tolist(),layer_thresholds=[x.tolist() for x in layers])
        for workload_index,c in enumerate(cs):
            test=c.subset(c.split); replay=ContributionReplay(test)
            baseline=replay.stats(replay.drop('baseline',.08))
            name=c.path.stem.split(f'-w{width}-')[-1]
            completeness.append(dict(width=width,workload=name,calls=len(c.call),steps=len(c.call)//48,
                train_steps=int((~c.split).sum())//48,test_steps=int(c.split.sum())//48,
                recon_p99=float(np.percentile(c.reconstruction,99)),recon_max=float(c.reconstruction.max())))
            target=3.7 if width==4 else 6.7
            target_removed=float(replay.r.distinct.mean())-baseline['D']+target
            exact_single=matched_threshold(replay,'singleton',target_removed)
            exact_joint=matched_threshold(replay,'joint',target_removed)
            low,high=0.,layer_max
            for _ in range(36):
                mid=(low+high)/2
                cut=calibrated_layers(tr,mid,curves)
                removed=(replay.drop('layer',cut)/replay.r.mult).sum()/len(test.call)
                if removed>=target_removed: high=mid
                else: low=mid
            exact_layer=calibrated_layers(tr,high,curves)
            capture_policies=policies+[('singleton',exact_single),('joint',exact_joint),('layer',exact_layer)]
            calibration[str(width)].setdefault('oracle_refinements',{})[name]=dict(
                singleton=exact_single,joint=exact_joint,layer_budget=high,layer_thresholds=exact_layer.tolist())
            local=[]
            for index,(kind,threshold) in enumerate(capture_policies):
                stat=replay.stats(replay.drop(kind,threshold))
                row=dict(width=width,workload=name,policy=kind,index=index,threshold=(float(threshold) if kind!='layer' else None),
                    delta_D=baseline['D']-stat['D'],step_saving_us=(baseline['D']-stat['D'])*69.9,**stat)
                rows.append(row); local.append(row)
            target=3.7 if width==4 else 6.7
            qualifying=[row for row in local if row['delta_D']>=target]
            best=min(qualifying,key=lambda x:(x['p95'],x['mean'])) if qualifying else None
            if best is not None:
                best=dict(best,baseline_p95=baseline['p95'],p95_ratio=best['p95']/baseline['p95'])
                selected.append(best)
            # Evaluate one threshold per family that reaches requested target
            # with least p95 on test. Tables are train-only; recall at same cut
            # AND matched D distinguish calibration drift from ranking quality.
            for kind in ('singleton','joint','layer'):
                feasible=[row for row in local if row['policy']==kind and row['delta_D']>=target]
                if not feasible: continue
                chosen=min(feasible,key=lambda x:(x['p95'],x['mean']))
                _,threshold=capture_policies[chosen['index']]
                true_drop=replay.drop(kind,threshold)
                for layer_id in range(48):
                    mask=test.layer==layer_id
                    mass=(replay.true_score[mask]*true_drop[mask]).sum(2)
                    if kind=='layer':
                        training_mass=[]
                        for training in tr:
                            lm=training.data.layer==layer_id
                            ld=training.single[lm]&(training.score[lm]<threshold[layer_id])
                            training_mass.append((training.true_score[lm]*ld).sum(2).ravel())
                        train_mean=float(np.concatenate(training_mass).mean())
                        layer_cut=float(threshold[layer_id])
                    else:
                        train_mean=None;layer_cut=float(threshold)
                    layer_rows.append(dict(width=width,workload=name,policy=kind,layer=layer_id,
                        threshold=layer_cut,calibration_mean=train_mean,mean=float(mass.mean()),
                        p95=float(np.percentile(mass,95)),p99=float(np.percentile(mass,99))))
                proxy=ContributionReplay(test,proxy_score(test,table))
                pred=proxy.drop(kind,threshold)
                ps=proxy.stats(pred)
                recall=selection_recall(true_drop,pred,replay.r.mult)
                # For layer equalization also preserve the layer policy as-is;
                # pooled matched-D diagnostic only applies global/group cuts.
                matched=None
                if kind!='layer':
                    removed=(true_drop/replay.r.mult).sum()/len(test.call)
                    cut=matched_threshold(proxy,kind,removed)
                    pm=proxy.drop(kind,cut)
                    matched=dict(threshold=cut,**selection_recall(true_drop,pm,replay.r.mult),**proxy.stats(pm))
                loo=ContributionReplay(test,proxy_score(test,loo_tables[workload_index]))
                loo_drop=loo.drop(kind,threshold)
                loo_result=dict(**selection_recall(true_drop,loo_drop,replay.r.mult),**loo.stats(loo_drop))
                if kind!='layer':
                    loo_cut=matched_threshold(loo,kind,(true_drop/replay.r.mult).sum()/len(test.call))
                    loo_matched=loo.drop(kind,loo_cut)
                    loo_result['matched_D']=dict(**selection_recall(true_drop,loo_matched,replay.r.mult),**loo.stats(loo_matched))
                seen=counts[test.layer[:,None,None],test.ids]>0
                proxy_rows.append(dict(width=width,workload=name,policy=kind,true_p95=chosen['p95'],
                    true_D=chosen['D'],proxy_delta_D=baseline['D']-ps['D'],unseen_fraction=float((~seen).mean()),
                    same_cut=dict(**recall,**ps),matched_D=matched,leave_workload_out=loo_result))
    common=[]
    for width in (4,16):
        grid_count=1+len(calibration[str(width)]['singleton'])+len(calibration[str(width)]['joint'])+len(calibration[str(width)]['budgets'])
        target=3.7 if width==4 else 6.7
        baselines={r['workload']:r['p95'] for r in rows if r['width']==width and r['policy']=='baseline'}
        candidates=[]
        for index in range(1,grid_count):
            group=[r for r in rows if r['width']==width and r['index']==index]
            if len(group)==4 and all(r['delta_D']>=target for r in group):
                ratio=max(r['p95']/baselines[r['workload']] for r in group)
                candidates.append((ratio,group))
        if candidates:
            worst,group=min(candidates,key=lambda x:x[0])
            common += [dict(r,baseline_p95=baselines[r['workload']],p95_ratio=r['p95']/baselines[r['workload']],
                            common_worst_ratio=worst) for r in group]
    import csv
    for name,data in [('frontier.csv',rows),('selected.csv',selected),('completeness.csv',completeness),('layer-error.csv',layer_rows),('common-policy.csv',common)]:
        with (output/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]) if data else ['empty'])
            writer.writeheader(); writer.writerows(data)
    (output/'proxy-evaluation.json').write_text(json.dumps(proxy_rows,indent=2)+'\n')
    (output/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    # Non-dominated D / p95 frontier, all policies; include the full grid CSV.
    pareto=[]
    for width in (4,16):
        for workload in sorted({r['workload'] for r in rows}):
            pool=sorted([r for r in rows if r['width']==width and r['workload']==workload],key=lambda x:(x['D'],x['p95']))
            best_error=float('inf')
            for r in pool:
                if r['p95']<best_error:
                    pareto.append(r); best_error=r['p95']
    with (output/'pareto.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(pareto)
    lines=['\n## Captured workload coverage\n',
        '| W | workload | steps | train / test steps | reconstruction p99 / max relative L2 |',
        '|---|---|---:|---:|---:|']
    for c in completeness:
        lines.append(f"| {c['width']} | {c['workload']} | {c['steps']} | {c['train_steps']} / {c['test_steps']} | {c['recon_p99']:.6f} / {c['recon_max']:.6f} |")
    lines+=['\n## Test frontier: baseline and best qualifying point per policy family\n',
        'The base threshold grid, layer calibration and norm tables use the temporal training half. Additional target-crossing thresholds are refined on the test data; the displayed best point is selected on that test frontier and is an **oracle screen**, not a validated production setting. All values below are target-verify layers 0–47 only.\n',
        '| W | workload | policy | D | delta D vs .08 | score sum mean | p95 | p99 | estimated us/step |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for width in (4,16):
        for workload in sorted({r['workload'] for r in rows}):
            local=[r for r in rows if r['width']==width and r['workload']==workload]
            for kind in ('baseline','singleton','layer','joint'):
                pool=[r for r in local if r['policy']==kind and (kind=='baseline' or r['delta_D']>=(3.7 if width==4 else 6.7))]
                if pool:
                    r=min(pool,key=lambda x:(x['p95'],x['mean']))
                    lines.append(f"| {width} | {workload} | {kind} | {r['D']:.3f} | {r['delta_D']:+.3f} | {r['mean']:.6f} | {r['p95']:.6f} | {r['p99']:.6f} | {r['step_saving_us']:+.2f} |")
                else: lines.append(f'| {width} | {workload} | {kind} | target unreachable | — | — | — | — | — |')
    lines+=['\n## Smallest p95 meeting the requested additional-D target\n',
        '| W | workload | winner | delta D | p95 | baseline p95 | p95 / baseline | us/step |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in selected:
        lines.append(f"| {r['width']} | {r['workload']} | {r['policy']} | {r['delta_D']:.3f} | {r['p95']:.6f} | {r['baseline_p95']:.6f} | {r['p95_ratio']:.3f} | {r['step_saving_us']:.2f} |")
    lines+=['\n## One common grid setting per width across all four workloads\n',
        'Among training-grid settings that meet the additional-D target in every test workload, minimize the worst workload p95/baseline ratio. These share one policy/threshold configuration per width; selection is still an oracle screen. Per-workload refined cuts above are excluded.\n',
        '| W | workload | common policy | grid index | delta D | p95 / baseline |',
        '|---|---|---|---:|---:|---:|']
    for r in common:
        lines.append(f"| {r['width']} | {r['workload']} | {r['policy']} | {r['index']} | {r['delta_D']:.3f} | {r['p95_ratio']:.3f} |")
    lines+=['\n## Held-out calibrated norm-table proxy\n',
        'Proxy = w * shrunk training mean expert norm[layer, expert] / current residual norm. Eight layer-mean pseudo-observations smooth each expert. No test norms enter fitting. Recall counts whole experts (a count-two group counts once). Matched-D cuts are test-time ranking diagnostics only.\n',
        '| W | workload | policy | same-cut expert recall | same-cut delta D | same-cut true p95 | matched-D recall | matched-D true p95 / oracle p95 | unseen routes |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in proxy_rows:
        p=r['same_cut'];m=r['matched_D']
        mr=f"{m['expert_recall']:.3f}" if m else '—'
        mp=f"{m['p95']/r['true_p95']:.3f}" if m else '—'
        lines.append(f"| {r['width']} | {r['workload']} | {r['policy']} | {p['expert_recall']:.3f} | {r['proxy_delta_D']:.3f} | {p['p95']:.6f} | {mr} | {mp} | {r['unseen_fraction']:.4f} |")
    lines+=['\n### Proxy generalization diagnostic: excluded workload\n',
        'Each norm table below trains on the temporal training halves of the other three workloads only. Threshold choice still uses the oracle frontier; this isolates norm-prediction generalization rather than validating a deployable policy.\n',
        '| W | workload | policy | excluded-workload same-cut recall | matched-D recall | matched-D p95 / oracle |',
        '|---|---|---|---:|---:|---:|']
    for r in proxy_rows:
        loo=r['leave_workload_out'];m=loo.get('matched_D')
        mr=f"{m['expert_recall']:.3f}" if m else '—'
        mp=f"{m['p95']/r['true_p95']:.3f}" if m else '—'
        lines.append(f"| {r['width']} | {r['workload']} | {r['policy']} | {loo['expert_recall']:.3f} | {mr} | {mp} |")
    lines+=['\nArtifacts: '+str(output)+'. `frontier.csv` holds every threshold, `pareto.csv` the D/p95 non-dominated subset, `calibration.json` every layer threshold, `proxy-evaluation.json` precision and route/expert recall, and `proxy-w*.npz` runtime table candidates. `layer-error.csv` gives per-layer calibration means and held-out contribution tails for each selected policy family.\n']
    with Path(report).open('a') as f: f.write('\n'.join(lines)+'\n')
    print(json.dumps(selected,indent=2))
    return rows,selected

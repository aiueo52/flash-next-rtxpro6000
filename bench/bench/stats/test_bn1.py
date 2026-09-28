"""CPU statistical and runner contract checks; no model or GPU required."""
import dataclasses
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

B=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(B));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from paired_ab import analyze, mde, power
from variance import analyze as variance
from fnbench.parsers import counter_acceptance
from fnbench.workloads import select_prompt_sets
from fnbench.runner import RunConfig, run_benchmark
from fnbench.gpu_guard import GuardResult
from fnbench.models import Sampling, Workload
from fnbench.http_client import StreamMeasurement


def counter(tokens, verifies):
    return f'sglang:generation_tokens_total{{model="x",is_streaming="true"}} {tokens}\nsglang:spec_verify_calls_total{{model="x"}} {verifies}\n'


def record(i,tps,accept=3):
    return dict(workload='code-edit',prompt_id=f'p{i}',repeat=1,prompt_sha256=f'hash{i}',max_tokens=4096,
                sampling={'temperature':0},model='fixed',engine='sglang',prompt_set='code-edit-v1',
                client={'decode_tps':tps},server={'acceptance':{'status':'ok','tokens_per_verify':accept}})


def write(path, values):
    path.write_text(''.join(json.dumps(record(i,float(x)))+'\n' for i,x in enumerate(values)))


class BN1Tests(unittest.TestCase):
    def test_counters(self):
        self.assertEqual(counter_acceptance(counter(20,5),counter(32,8),12)['tokens_per_verify'],4)
        self.assertEqual(counter_acceptance(counter(20,5),counter(0,0),12)['status'],'counter_reset')
        self.assertEqual(counter_acceptance('',counter(12,3),12)['status'],'missing_or_changed_series')
        self.assertEqual(counter_acceptance(counter(20,5),counter(20,5),12)['status'],'pending')
        self.assertEqual(counter_acceptance(counter(20,5),counter(40,10),12)['status'],'completion_mismatch_or_concurrent_requests')
        self.assertIsNone(counter_acceptance(None,None,12)['tokens_per_verify'])

    def test_sets(self):
        sets=select_prompt_sets('code-edit,prose-en,prose-ja,agent-loop',B/'workloads/sets')
        self.assertEqual(len(sets),32)
        self.assertEqual(len({s.prompt_id for s in sets}),32)
        self.assertEqual([s.name for s in sets[:4]],['code-edit','prose-en','prose-ja','agent-loop'])
        with self.assertRaises(ValueError): select_prompt_sets('code-edit',B/'workloads/sets',limit=16)

    def test_pairing_and_abba_linear_drift(self):
        with tempfile.TemporaryDirectory() as d:
            paths=[Path(d)/f'{i}.jsonl' for i in range(4)]
            base=np.exp(np.linspace(3,7,8))
            for i,p in enumerate(paths): write(p,base*math.exp(.1*i)*(1.05 if i in (1,2) else 1))
            result=analyze(paths,'ABBA',draws=1000)['domains']['code-edit']['tps']
            self.assertAlmostEqual(result['change_pct'],5,places=8)
            self.assertAlmostEqual(result['ci_pct'][0],5,places=8)
            self.assertTrue(result['A2_over_A1_drift']['flag'])
            self.assertAlmostEqual(result['block_means'][0]['log_ratio'],math.log(1.05)+.1)
            with self.assertRaises(ValueError): analyze([paths[0],paths[1],paths[1],paths[3]],'ABBA')
            # No silent intersection when a request is absent.
            write(paths[3],base[:-1])
            with self.assertRaises(ValueError): analyze(paths,'ABBA')

    def test_power_and_variance_components(self):
        self.assertLess(mde(.04/math.sqrt(16),15),mde(.04/math.sqrt(8),7))
        effect=mde(.02,7)/100
        self.assertAlmostEqual(power(math.log1p(effect),.02,7),.8,places=7)
        with tempfile.TemporaryDirectory() as d:
            paths=[Path(d)/f'{i}.jsonl' for i in range(3)]
            # Large parent difficulty with a pure shared restart shift.
            for i,p in enumerate(paths): write(p,np.exp(np.linspace(3,7,8)+[-.1,0,.1][i]))
            v=variance(paths)['code-edit']['tps']
            self.assertAlmostEqual(v['within_restart_residual_cv_pct'],0,places=6)
            self.assertGreater(v['between_restart_component_cv_pct'],9)
            self.assertGreater(v['planning_options'][0]['four_arm_cycles'],100)
            self.assertGreaterEqual(v['planning_options'][0]['power_plugin'],.80)
            self.assertAlmostEqual(v['mde']['8']['restart_aware_pct'],v['mde']['16']['restart_aware_pct'])

    def test_multiple_cycles_keep_restart_uncertainty(self):
        with tempfile.TemporaryDirectory() as d:
            paths=[Path(d)/f'{i}.jsonl' for i in range(8)]
            for i,p in enumerate(paths):
                effect=.05 if i<4 else -.03
                write(p,np.exp(np.linspace(3,7,8))*(math.exp(effect) if i%4 in (1,2) else 1))
            r=analyze(paths,'ABBA',draws=1000)['domains']['code-edit']['tps']
            self.assertAlmostEqual(r['log_mean'],.01)
            self.assertLess(r['crossed_cycle_prompt_interval']['lower_one_sided_pct'],0)
            self.assertGreater(r['lower_one_sided_pct'],0)

    def test_set_runner_does_not_warm_each_holdout(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'p.txt';path.write_text('heldout fixture')
            w=Workload('code-edit',path,32,Sampling(0),'test',prompt_id='p1',prompt_set='code-edit-v1')
            m=StreamMeasurement(.1,2,5.5,{'completion_tokens':12},[],[],12,0,'sample','hash')
            cfg=RunConfig('http://localhost:8001/v1','sglang',[w],1,Path(d)/'run.jsonl','A1',False,'greedy',model='x',require_acceptance=True)
            with patch('fnbench.runner.check_gpu_guard',return_value=GuardResult(True,None,())), \
                 patch('fnbench.runner.OpenAIStreamClient.complete',return_value=m) as complete, \
                 patch('fnbench.runner.fetch_metrics',side_effect=[{'raw':counter(20,5)},{'raw':counter(32,8)}]):
                self.assertEqual(run_benchmark(cfg),0)
                self.assertEqual(complete.call_count,1)
            row=json.loads(cfg.out.read_text())
            self.assertEqual(row['server']['accept_length'],4)
            self.assertEqual(row['derived']['effective_forward_per_second'],5.5/4)
            self.assertEqual(row['prompt_id'],'p1')

if __name__=='__main__': unittest.main()

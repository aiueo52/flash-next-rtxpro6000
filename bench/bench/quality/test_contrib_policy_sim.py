"""Meaningful P3 policy, calibration, loader and numerical-isolation regressions."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from contrib_policy_sim import (Capture,ContributionReplay,baseline_mask,calibrated_layers,
    fit_proxy,proxy_score,selection_recall,load_capture,matched_threshold)


def tiny(ids=None,w=None,norm=None,layer=0):
    if ids is None: ids=[[[0,1,2],[0,1,3]]]
    if w is None: w=[[[.8,.05,.15],[.8,.07,.13]]]
    ids=np.asarray(ids,dtype=np.int32);w=np.asarray(w,dtype=np.float32)
    if norm is None: norm=np.ones_like(w)
    return Capture(Path('tiny'),ids,w,np.asarray(norm),np.ones(ids.shape[:2]),
        np.full(len(ids),layer),np.arange(len(ids)),np.zeros(len(ids),bool),np.zeros(ids.shape[:2]))


class Policies(unittest.TestCase):
    def test_joint_cost_protection_and_strict_boundary(self):
        c=tiny();r=ContributionReplay(c)
        threshold=r.group_score[0,0,1]
        self.assertFalse(r.drop('joint',threshold)[0,0,1])
        drop=r.drop('joint',np.nextafter(threshold,np.inf))
        self.assertTrue(drop[0,0,1] and drop[0,1,1])
        self.assertFalse(drop[:,:,0].any())
        self.assertEqual(r.stats(drop)['D'],3)
        # Removing either of an expert's two routes alone is invalid.
        drop[0,1,1]=False
        with self.assertRaisesRegex(ValueError,'partial'):r.stats(drop)

    def test_score_changes_ranking_and_includes_zero_rows(self):
        c=tiny(norm=[[[1,1,.1],[1,1,10]]]);r=ContributionReplay(c)
        drop=r.drop('singleton',.05)
        expected=np.array([[float(c.w[0,0,2])*.1,0.]])
        np.testing.assert_allclose((c.score*drop).sum(2),expected)
        stat=r.stats(drop)
        self.assertEqual(stat['D'],3)
        np.testing.assert_allclose([stat['mean'],stat['p95'],stat['p99']],
            [expected.mean(),*np.percentile(expected,[95,99])])

    def test_protected_pair_cannot_partially_drop(self):
        c=tiny(w=[[[.05,.9,.05],[.85,.1,.05]]]);r=ContributionReplay(c)
        drop=r.drop('joint',99)
        self.assertFalse(drop[:,:,0].any())
        self.assertFalse(drop[:,:,1].any())
        self.assertEqual(r.stats(drop)['D'],2)

    def test_float32_baseline_and_stable_top1(self):
        c=tiny(ids=[[[1,2,3],[4,5,6]]],w=[[[.08,.08,.01],[.9,.08,.02]]])
        r=ContributionReplay(c);drop=baseline_mask(r.r)
        self.assertFalse(drop[0,0,0]);self.assertFalse(drop[0,0,1])
        self.assertTrue(drop[0,0,2]);self.assertTrue(drop[0,1,2])
        self.assertFalse(drop[0,1,1])

    def test_immutable_order_and_layer_budgets(self):
        a=tiny(ids=[[[1,2,3],[4,5,6]]],norm=[[[1,1,1],[1,1,1]]],layer=0)
        b=tiny(ids=[[[1,2,3],[4,5,6]]],norm=[[[1,2,2],[1,2,2]]],layer=1)
        replays=[ContributionReplay(a),ContributionReplay(b)]
        ai=a.ids.copy();aw=a.w.copy()
        table=calibrated_layers(replays,.07)
        for r in replays:
            mask=r.drop('layer',table)
            self.assertLessEqual((r.score*mask).sum()/r.data.width,.07+1e-12)
        self.assertEqual(table.shape,(48,))
        self.assertTrue((table[2:]==0).all())
        for kind in ('baseline','joint','singleton'):
            replays[0].stats(replays[0].drop(kind,.08))
        np.testing.assert_array_equal(a.ids,ai);np.testing.assert_array_equal(a.w,aw)

    def test_proxy_train_only_fallback_and_recall(self):
        train=tiny();table,counts=fit_proxy([train])
        test=tiny(norm=np.full((1,2,3),10000))
        before=proxy_score(test,table)
        test.norm[:]=20000
        np.testing.assert_array_equal(before,proxy_score(test,table))
        self.assertTrue(np.isfinite(table).all())
        self.assertTrue(counts[0,511]==0)
        truth=np.array([[[1,1],[1,0]]],bool)
        pred=np.array([[[1,0],[1,1]]],bool)
        mult=np.array([[[2,1],[2,1]]])
        result=selection_recall(truth,pred,mult)
        self.assertEqual(result['expert_recall'],.5)

    def test_exact_target_crossing_retains_strict_ties(self):
        r=ContributionReplay(tiny())
        threshold=matched_threshold(r,'joint',1.)
        drop=r.drop('joint',threshold)
        self.assertEqual(r.stats(drop)['D'],3)
        previous=np.nextafter(threshold,-np.inf)
        self.assertEqual(r.stats(r.drop('joint',previous))['D'],4)

    def test_invalid_norm_and_partial_capture(self):
        c=tiny();c.h[0,0]=0
        with np.errstate(divide='ignore'):
            with self.assertRaisesRegex(ValueError,'score'):ContributionReplay(c)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.npz'
            np.savez(path,schema_version=1,role='target_verify',call=np.arange(48),
                T=np.full(48,4),k=np.full(48,10),total_calls=49)
            with self.assertRaisesRegex(ValueError,'overflow'):load_capture(path)

if __name__=='__main__':unittest.main()

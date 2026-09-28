"""Run the unchanged corrected Q1 analyzer on timestamped P4 arms via private aliases."""
import argparse,json,math,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def tango_bounds(b, c, n, z):
    """Tango (1998) score interval for a paired difference theta = (b - c) / n, where b and c are the two
    discordant counts. Valid for small n and few (even zero) discordances, unlike the Wald interval.
    Returns (lower, upper): the theta values where the score statistic equals +z and -z."""
    def score(t):
        w = b + c - t * (2 * n - b + c)
        q = (w + math.sqrt(max(w * w + 8 * n * c * t * (1 - t), 0.0))) / (4 * n)  # constrained MLE of p(c-cell)
        var = n * (2 * q + t - t * t)
        num = b - c - n * t
        if var <= 0: return 0.0 if num == 0 else math.copysign(math.inf, num)
        return num / math.sqrt(var)
    def solve(target, lo, hi):  # score is decreasing in t
        for _ in range(200):
            mid = (lo + hi) / 2
            if score(mid) > target: lo = mid
            else: hi = mid
        return (lo + hi) / 2
    est = (b - c) / n; eps = 1e-12
    lower = -1.0 if score(-1 + eps) <= z else solve(z, -1 + eps, est)
    upper = 1.0 if score(1 - eps) >= -z else solve(-z, est, 1 - eps)
    return lower, upper
def main():
 ap=argparse.ArgumentParser();ap.add_argument('label');a=ap.parse_args()
 missing=[str(ROOT/'bench/quality/runs'/f'{a.label}-{arm}') for arm in ('prod','contrib','noprune','prod2')
          if not (ROOT/'bench/quality/runs'/f'{a.label}-{arm}'/'summary.json').exists()]
 if missing:raise SystemExit('p4_quality_report.py: missing P4 quality arms (not published in this repository): '+', '.join(missing))
 out=ROOT/'runs/p4'/f'{a.label}-quality-analysis';(out/'runs').mkdir(parents=True,exist_ok=False)
 for arm in ('prod','contrib','noprune','prod2'):
  source=ROOT/'bench/quality/runs'/f'{a.label}-{arm}'
  assert (source/'summary.json').exists(),source
  (out/'runs'/arm).symlink_to(source,target_is_directory=True)
 shutil.copy2(ROOT/'bench/quality/analyze.py',out/'analyze.py')
 text=subprocess.check_output([sys.executable,str(out/'analyze.py'),'--margin-pp','0.5','contrib','prod','contrib','noprune','prod2'],text=True)
 # Analyzer compares BASE minus comparator in its NI table. BASE=contrib is intentional.
 (out/'corrected-analysis.md').write_text(text)
 checks=[];significant=[]
 for bench in ('gsm8k','mmlu','humaneval','jcqa'):
  def read(arm):return {r['id']:r for r in map(json.loads,(out/'runs'/arm/f'{bench}.jsonl').read_text().splitlines())}
  b=read('prod');c=read('contrib');assert b.keys()==c.keys()
  n=len(b);loss=sum(b[i]['correct'] and not c[i]['correct'] for i in b);win=sum(c[i]['correct'] and not b[i]['correct'] for i in b)
  delta=(win-loss)/n
  lower,_=tango_bounds(win,loss,n,1.6448536269514722) # one-sided 95% Tango score bound; valid with zero discordances
  p=min(1.,2*sum(math.comb(win+loss,j) for j in range(min(win,loss)+1))/(1<<(win+loss)))
  checks.append(dict(bench=bench,n=n,loss=loss,win=win,diff_pp=delta*100,lower_pp=lower*100,p=p,noninferiority=lower>-.005))
  if loss>win and p<.05:significant.append(bench)
 result=dict(checks=checks,noninferiority=all(c['noninferiority'] for c in checks),significant_degradation=significant)
 (out/'gate.json').write_text(json.dumps(result,indent=2))
 with (ROOT/'specs/P4_CONTRIB_PRUNE_SHIP.md').open('a') as f:
  f.write('\n## Quality battery\n\nBASE is contrib; non-inferiority columns in the unchanged corrected analyzer are BASE minus comparator.\n\n'+text)
  f.write('\n### Candidate versus production gate with one-sided Tango score bound\n\n| Benchmark | n | loss / win | candidate minus prod pp | one-sided lower pp | p | NI |\n|---|---:|---:|---:|---:|---:|---|\n')
  for c in checks:f.write(f"| {c['bench']} | {c['n']} | {c['loss']} / {c['win']} | {c['diff_pp']:+.3f} | {c['lower_pp']:+.3f} | {c['p']:.5f} | {'PASS' if c['noninferiority'] else 'INCONCLUSIVE/FAIL'} |\n")
  f.write('\nSignificance verdict versus prod: '+(', '.join(significant) if significant else 'no significant degradation detected')+'. This does not imply non-inferiority.\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()

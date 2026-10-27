"""Mini Project 2.3: Partitioned multiprocessor scheduling heuristics.
CE80763 Real-Time Systems — Unit 2. Conservative per-core RM utilization-bound admission.
"""
from dataclasses import dataclass
from math import gcd
from functools import reduce
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class Task:
    name: str; C: int; T: int
    def __post_init__(self):
        if not isinstance(self.C,int) or not isinstance(self.T,int): raise TypeError("C and T must be integers")
        if self.C<=0 or self.T<=0: raise ValueError("C and T must be positive")
    @property
    def u(self): return self.C/self.T

def rm_bound(n): return n*(2**(1/n)-1) if n else 1.0
def hyperperiod(tasks): return reduce(lambda a,b:a*b//gcd(a,b),(t.T for t in tasks),1)
def core_admits(core_tasks,new_task):
    proposed=core_tasks+[new_task]
    return sum(t.u for t in proposed)<=rm_bound(len(proposed))
def first_fit(tasks,m):
    cores=[[] for _ in range(m)]
    for t in tasks:
        for core in cores:
            if core_admits(core,t): core.append(t); break
        else: return None
    return cores
def first_fit_decreasing(tasks,m): return first_fit(sorted(tasks,key=lambda t:-t.u),m)
def best_fit_decreasing(tasks,m):
    cores=[[] for _ in range(m)]
    for t in sorted(tasks,key=lambda t:-t.u):
        fits=[(rm_bound(len(c)+1)-sum(x.u for x in c)-t.u,c) for c in cores if core_admits(c,t)]
        if not fits: return None
        min(fits,key=lambda item:item[0])[1].append(t)
    return cores
def worst_fit(tasks,m):
    cores=[[] for _ in range(m)]
    for t in tasks:
        fits=[(rm_bound(len(c)+1)-sum(x.u for x in c)-t.u,c) for c in cores if core_admits(c,t)]
        if not fits: return None
        max(fits,key=lambda item:item[0])[1].append(t)
    return cores

def simulate_rm(tasks,horizon):
    """Per-core RM timeline; suitable for visualization, not a substitute for analysis."""
    if not tasks: return []
    if not isinstance(horizon,int) or horizon<0: raise ValueError("horizon must be a nonnegative integer")
    priority={t.name:i for i,t in enumerate(sorted(tasks,key=lambda t:t.T))}
    remaining={t.name:0 for t in tasks}; next_release={t.name:0 for t in tasks}; timeline=[]
    for tick in range(horizon):
        for t in tasks:
            if tick==next_release[t.name]: remaining[t.name]+=t.C; next_release[t.name]+=t.T
        ready=[t for t in tasks if remaining[t.name]>0]
        if not ready: timeline.append(None); continue
        chosen=min(ready,key=lambda t:priority[t.name]); timeline.append(chosen.name); remaining[chosen.name]-=1
    return timeline

def report(label,tasks,m,cores):
    U=sum(t.u for t in tasks); print(f"\n{label}: m={m}, total U={U:.3f}, aggregate capacity={m:.1f}")
    if cores is None: print(f"  FAILED to find a partition under the per-core RM sufficient-bound heuristic."); return
    for i,c in enumerate(cores):
        util=sum(t.u for t in c)
        print(f"  core {i}: {', '.join(f'{t.name}(u={t.u:.3f})' for t in c) or '(idle)'}; U_core={util:.3f}; RM bound={rm_bound(len(c)):.3f}")
    print(f"  cores used={sum(bool(c) for c in cores)}; average utilization per core={100*U/m:.1f}%")

def run_all(tasks,m,label):
    for name,fn in [("FF",first_fit),("FFD",first_fit_decreasing),("BFD",best_fit_decreasing),("WF",worst_fit)]: report(f"{label} [{name}]",tasks,m,fn(tasks,m))

if __name__=="__main__":
    m=4
    balanced=[Task(f"b{i}",1,20) for i in range(1,11)]
    run_all(balanced,m,"Balanced (10 small tasks, U=0.50)")
    skewed=[Task("s1",5,12),Task("s2",5,13),Task("s3",1,20),Task("s4",1,25),Task("s5",1,30)]
    run_all(skewed,m,"Skewed task set")
    adversarial=[Task(f"a{i}",6,11) for i in range(1,5)]
    run_all(adversarial,m,"Four tasks of utilization about 0.545")
    tight=[Task(f"t{i}",2,10+i) for i in range(1,9)]
    run_all(tight,m,"Tight set")
    # Full H for T=11..18 is 12,252,240 ticks. Never plot every tick by default.
    cores=first_fit_decreasing(tight,m)
    if cores is not None:
        horizon=min(hyperperiod(tight),300)
        fig,axes=plt.subplots(m,1,figsize=(12,5),sharex=True); cmap=plt.get_cmap("tab10")
        names=[t.name for t in tight]; colors={n:cmap(i%10) for i,n in enumerate(names)}
        from matplotlib.patches import Patch
        for i,c in enumerate(cores):
            tl=simulate_rm(c,horizon)
            for x,n in enumerate(tl):
                if n is not None: axes[i].broken_barh([(x,1)],(.2,.6),facecolors=colors[n],edgecolor="white",linewidth=.3)
            axes[i].set_yticks([.5]); axes[i].set_yticklabels([f"core {i}"])
        axes[-1].set_xlabel(f"Time tick (showing first {horizon} of H={hyperperiod(tight)} ticks)")
        fig.suptitle("FFD per-core RM schedule (illustrative prefix)")
        fig.legend(handles=[Patch(facecolor=colors[n],label=n) for n in names],loc="upper right",ncol=4)
        fig.tight_layout(); fig.savefig("mp2_3_gantt.png",dpi=160); plt.close(fig)

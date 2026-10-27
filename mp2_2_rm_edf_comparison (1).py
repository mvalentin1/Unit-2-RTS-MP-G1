"""Mini Project 2.2: RM/DM versus EDF schedulability comparison.
CE80763 Real-Time Systems — Unit 2. Integer-time job-level simulation.
"""
from dataclasses import dataclass
from math import ceil, gcd
from functools import reduce
import csv, sys
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class Task:
    name: str; C: int; T: int; D: int | None = None
    def __post_init__(self):
        if not isinstance(self.C,int) or not isinstance(self.T,int) or (self.D is not None and not isinstance(self.D,int)):
            raise TypeError("C, T and D must be integers for the discrete-time simulator")
        if self.C<=0 or self.T<=0 or (self.D is not None and self.D<=0): raise ValueError("C, T and D must be positive")
        if self.D is None: object.__setattr__(self,"D",self.T)
    @property
    def u(self): return self.C/self.T

def total_utilization(tasks): return sum(t.u for t in tasks)
def rm_bound(n): return n*(2**(1/n)-1) if n else 1.0
def hyperperiod(tasks): return reduce(lambda a,b:a*b//gcd(a,b),(t.T for t in tasks),1)
def rta(tasks, priority="RM"):
    """Exact fixed-priority response-time recurrence for this independent periodic model."""
    if priority not in {"RM","DM"}: raise ValueError("priority must be RM or DM")
    ordered=sorted(tasks,key=(lambda t:t.T) if priority=="RM" else (lambda t:t.D))
    out={}; hp=[]
    for t in ordered:
        R=t.C
        for iteration in range(1,10000):
            new_R=t.C+sum(ceil(R/h.T)*h.C for h in hp)
            if new_R==R:
                out[t.name]=(R,iteration,R<=t.D); break
            if new_R>t.D:
                out[t.name]=(new_R,iteration,False); break
            R=new_R
        else: raise RuntimeError("RTA failed to converge")
        hp.append(t)
    return out

def simulate(tasks,horizon,policy="RM"):
    jobs=[]; timeline=[]; misses=[]
    if policy not in {"RM","DM","EDF"}: raise ValueError("policy must be RM, DM or EDF")
    order=sorted(tasks,key=(lambda t:t.T) if policy=="RM" else (lambda t:t.D))
    pri={t.name:i for i,t in enumerate(order)}
    for tick in range(horizon):
        for j in jobs:
            if j["remaining"]>0 and j["deadline"]==tick and not j["miss"]:
                misses.append((j["task"].name,j["release"],j["deadline"])); j["miss"]=True
        for t in tasks:
            if tick%t.T==0: jobs.append({"task":t,"release":tick,"deadline":tick+t.D,"remaining":t.C,"miss":False})
        ready=[j for j in jobs if j["remaining"]>0]
        if not ready: timeline.append(None); continue
        chosen=min(ready,key=(lambda j:(j["deadline"],j["release"])) if policy=="EDF" else (lambda j:(pri[j["task"].name],j["release"])))
        timeline.append(chosen["task"].name); chosen["remaining"]-=1
    for j in jobs:
        if j["remaining"]>0 and j["deadline"]<=horizon and not j["miss"]:
            misses.append((j["task"].name,j["release"],j["deadline"])); j["miss"]=True
    return timeline,misses

def plot_schedules(tasks,horizon,path=None):
    rmtl,_=simulate(tasks,horizon,"RM" if all(t.D==t.T for t in tasks) else "DM")
    edftl,_=simulate(tasks,horizon,"EDF")
    fig,axes=plt.subplots(2,1,figsize=(12,4),sharex=True); cmap=plt.get_cmap("tab10")
    names=[t.name for t in tasks]; colors={n:cmap(i%10) for i,n in enumerate(names)}
    for ax,tl,title in [(axes[0],rmtl,"RM (or DM for constrained deadlines)"),(axes[1],edftl,"EDF")]:
        for x,n in enumerate(tl):
            if n is not None: ax.broken_barh([(x,1)],(.2,.6),facecolors=colors[n],edgecolor="white",linewidth=.3)
        ax.set_yticks([.5]); ax.set_yticklabels(["CPU"]); ax.set_title(title)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(facecolor=colors[n],label=n) for n in names],loc="upper right",ncol=max(1,len(names)))
    axes[-1].set_xlabel("Time tick"); axes[-1].set_xlim(0,horizon); fig.tight_layout()
    if path: fig.savefig(path,dpi=160)
    plt.close(fig)

def analyze(tasks,name="Task set",make_plot=True,max_plot_horizon=200):
    U=total_utilization(tasks); n=len(tasks); constrained=any(t.D<t.T for t in tasks)
    print(f"\n{name}: U={U:.4f}; n={n}; RM bound={rm_bound(n):.4f}; H={hyperperiod(tasks)}")
    print("Task parameters:")
    for t in tasks: print(f"  {t.name}: C={t.C}, T={t.T}, D={t.D}, u={t.u:.4f}")
    if constrained:
        rm_results=rta(tasks,"RM")
        dm_results=rta(tasks,"DM")
        print("Fixed-priority comparison for constrained deadlines:")
        print("  RM priority (period order):")
        for nm,(R,it,ok) in rm_results.items(): print(f"    {nm}: R={R}, iterations={it}, {'PASS' if ok else 'DEADLINE MISS'}")
        print("  DM priority (deadline order):")
        for nm,(R,it,ok) in dm_results.items(): print(f"    {nm}: R={R}, iterations={it}, {'PASS' if ok else 'DEADLINE MISS'}")
        fixed_ok=all(v[2] for v in dm_results.values())
        edf_note="U<=1 is not an exact EDF test for constrained deadlines; use processor-demand analysis."
        edf_ok=None
    else:
        if U<=rm_bound(n): fixed_ok=True; print("RM utilization sufficient test: PASS")
        else:
            fixed=rta(tasks,"RM"); fixed_ok=all(v[2] for v in fixed.values())
            print("RM utilization sufficient test: inconclusive; response-time analysis:")
            for nm,(R,it,ok) in fixed.items(): print(f"  {nm}: R={R}, iterations={it}, {'PASS' if ok else 'DEADLINE MISS'}")
        edf_ok=U<=1.0; edf_note=f"EDF utilization test: {'PASS' if edf_ok else 'FAIL'} (U {'<=' if edf_ok else '>'} 1)"
    print(f"Fixed-priority schedulable: {fixed_ok}")
    if edf_ok is not None: print(edf_note)
    else: print(edf_note)
    if make_plot:
        horizon=min(hyperperiod(tasks),max_plot_horizon)
        safe_name="".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")
        plot_schedules(tasks,horizon,f"schedule_{safe_name}.png")
        _,rm_misses=simulate(tasks,horizon,"DM" if constrained else "RM")
        _,edf_misses=simulate(tasks,horizon,"EDF")
        print(f"Simulation over first {horizon} ticks (illustrative horizon): fixed-priority deadline misses={len(rm_misses)}, EDF deadline misses={len(edf_misses)}")
    return fixed_ok,edf_ok

def load_csv(path):
    with open(path,newline="",encoding="utf-8") as f:
        return [Task(row["name"],int(row["C"]),int(row["T"]),int(row["D"]) if row.get("D") else None) for row in csv.DictReader(f)]

if __name__=="__main__":
    analyze([Task("t1",1,4),Task("t2",2,6),Task("t3",2,12)],"Worked example 2.2.2")
    analyze([Task("t1",1,4),Task("t2",2,6),Task("t3",2,12),Task("t4",1,8)],"Worked example 2.3.2")
    analyze([Task("t1",1,4),Task("t2",2,6),Task("t3",1,10),Task("t4",1,11)],"Boundary-style set")
    analyze([Task("t1",3,5),Task("t2",3,6)],"Overloaded set")
    analyze([Task("t1",3,6,D=6),Task("t2",2,8,D=3)],"Constrained-deadline counterexample")
    analyze([Task(f"t{i}",4,50+i) for i in range(1,11)],"Ten tasks near the RM utilization bound",make_plot=False)

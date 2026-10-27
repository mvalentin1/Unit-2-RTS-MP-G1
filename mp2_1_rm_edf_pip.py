"""Mini Project 2.1: RM, EDF, response-time analysis, and priority inheritance.
CE80763 Real-Time Systems — Unit 2
Discrete-time simulator: C, T, D and simulation horizon must be integers.
"""
from dataclasses import dataclass
from math import ceil, gcd
from functools import reduce
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class Task:
    name: str
    C: int
    T: int
    D: int | None = None
    def __post_init__(self):
        if self.C <= 0 or self.T <= 0 or (self.D is not None and self.D <= 0):
            raise ValueError("C, T and D must be positive integers")
        if not all(isinstance(x, int) for x in (self.C, self.T)) or (self.D is not None and not isinstance(self.D, int)):
            raise TypeError("This tick-based simulator requires integer C, T and D")
        if self.D is None:
            object.__setattr__(self, "D", self.T)
    @property
    def u(self): return self.C / self.T

def utilization(tasks): return sum(t.u for t in tasks)
def rm_bound(n): return n * (2 ** (1 / n) - 1) if n else 1.0
def hyperperiod(tasks): return reduce(lambda a,b: a*b//gcd(a,b), (t.T for t in tasks), 1)

def response_time_analysis(tasks, priority="RM"):
    """Fixed-priority RTA. RM uses period order; DM uses relative-deadline order."""
    if priority not in {"RM", "DM"}: raise ValueError("priority must be RM or DM")
    ordered = sorted(tasks, key=(lambda t: t.T) if priority == "RM" else (lambda t: t.D))
    results = {}
    hp = []
    for t in ordered:
        R = t.C
        for iteration in range(1, 10000):
            new_R = t.C + sum(ceil(R / h.T) * h.C for h in hp)
            if new_R == R:
                results[t.name] = {"R": R, "iterations": iteration, "schedulable": R <= t.D}
                break
            if new_R > t.D:
                results[t.name] = {"R": new_R, "iterations": iteration, "schedulable": False}
                break
            R = new_R
        else: raise RuntimeError("RTA failed to converge within iteration limit")
        hp.append(t)
    return results

def simulate(tasks, horizon, policy="RM"):
    """Job-level preemptive schedule with explicit deadlines and deadline-miss records."""
    if horizon < 0 or not isinstance(horizon, int): raise ValueError("horizon must be a nonnegative integer")
    if policy not in {"RM", "EDF"}: raise ValueError("policy must be RM or EDF")
    jobs, timeline, misses = [], [], []
    task_priority = {t.name:i for i,t in enumerate(sorted(tasks,key=lambda t:t.T))}
    for tick in range(horizon):
        # A job still unfinished at its deadline has missed it.
        for j in jobs:
            if j["remaining"] > 0 and j["deadline"] == tick and not j["miss_recorded"]:
                misses.append((j["task"].name, j["release"], j["deadline"]))
                j["miss_recorded"] = True
        for t in tasks:
            if tick % t.T == 0:
                jobs.append({"task":t,"release":tick,"deadline":tick+t.D,"remaining":t.C,"miss_recorded":False})
        ready = [j for j in jobs if j["remaining"] > 0]
        if not ready:
            timeline.append(None); continue
        if policy == "RM": chosen = min(ready, key=lambda j:task_priority[j["task"].name])
        else: chosen = min(ready, key=lambda j:(j["deadline"],j["release"]))
        timeline.append(chosen["task"].name)
        chosen["remaining"] -= 1
    # Check deadlines that occur exactly at horizon.
    for j in jobs:
        if j["remaining"] > 0 and j["deadline"] <= horizon and not j["miss_recorded"]:
            misses.append((j["task"].name,j["release"],j["deadline"]))
            j["miss_recorded"] = True
    return timeline, misses

def priority_inversion_demo(use_pip=True):
    """Small discrete event illustration: L holds R, H blocks on R, M is independent.
    Returns CPU timeline and H's resource-blocking interval. This is an educational model,
    not a general implementation of nested/multiple-resource PIP.
    """
    # Fixed scenario: L executes 6 ticks in its critical section; H arrives at 1; M at 2 for 4 ticks.
    remaining = {"L":6,"H":2,"M":4}; arrivals={"L":0,"H":1,"M":2}
    owns_R=True; waiting_H=False; release_tick=None; timeline=[]; inherited=[]
    for tick in range(20):
        if tick >= arrivals["H"] and owns_R: waiting_H=True
        ready=[]
        for n in remaining:
            if remaining[n] <= 0 or tick < arrivals[n]: continue
            if n == "H" and owns_R: continue
            ready.append(n)
        if not ready: timeline.append("IDLE"); continue
        priorities={"H":0,"M":1,"L":2}
        if use_pip and waiting_H and owns_R: priorities["L"]=-1
        chosen=min(ready,key=lambda n:priorities[n]); timeline.append(chosen)
        if chosen=="L" and use_pip and waiting_H and owns_R: inherited.append(tick)
        remaining[chosen]-=1
        if chosen=="L" and remaining["L"]==0 and owns_R:
            owns_R=False; release_tick=tick+1
    h_first=next((i for i,x in enumerate(timeline) if x=="H"),None)
    return timeline, release_tick, h_first, inherited

def plot_timeline(timeline, title, path=None):
    fig, ax=plt.subplots(figsize=(12,2.6))
    palette=plt.get_cmap("tab10")
    names=sorted(set(x for x in timeline if x not in (None,"IDLE")))
    colors={n:palette(i%10) for i,n in enumerate(names)}
    for i,n in enumerate(timeline):
        if n not in (None,"IDLE"):
            ax.broken_barh([(i,1)],(0.2,0.6),facecolors=colors[n],edgecolor="white",linewidth=.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=colors[n],label=n) for n in names],loc="upper right",ncol=max(1,len(names)))
    ax.set(xlim=(0,len(timeline)),yticks=[.5],yticklabels=["CPU"],xlabel="Time tick",title=title)
    ax.grid(axis="x",alpha=.2); fig.tight_layout()
    if path: fig.savefig(path,dpi=160)
    plt.close(fig)

if __name__ == "__main__":
    sets={"Worked example 1":[Task("t1",1,4),Task("t2",2,6),Task("t3",2,12)],
          "Worked example 2":[Task("t1",1,4),Task("t2",2,6),Task("t3",2,12),Task("t4",1,8)]}
    for label,tasks in sets.items():
        print(f"\n{label}: U={utilization(tasks):.4f}; RM bound={rm_bound(len(tasks)):.4f}; H={hyperperiod(tasks)}")
        for p in ("RM","EDF"):
            tl,miss=simulate(tasks,hyperperiod(tasks),p)
            print(f"  {p}: simulated {len(tl)} ticks; deadline misses={len(miss)}")
            plot_timeline(tl,f"{label} — {p}",f"mp2_1_{label.lower().replace(' ','_')}_{p.lower()}.png")
        print("  RM RTA:",response_time_analysis(tasks,"RM"))
    for pip in (False,True):
        tl,release,hstart,inh=priority_inversion_demo(pip)
        print(f"PIP={pip}: resource release={release}, H first runs={hstart}, inherited ticks={inh}")

import numpy as np, pandas as pd, json
from pathlib import Path
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

daily = pd.read_parquet("data/raw/daily_export.parquet")
CONTROLLABLE = [c for c in daily.columns if c.startswith("do_")]
# These journal items are SYMPTOMS/feelings, not actions you take -> exclude as treatments
SYMPTOMS = {"do_experienced_brain_fog","do_experienced_fatigue",
            "do_felt_emotionally_and_mentally_stable"}
ACTIONABLE = [c for c in CONTROLLABLE if c not in SYMPTOMS]

TARGET = "total_sleep_minutes"
SECONDARY = ["sleep_efficiency","sws_minutes","rem_minutes","hrv_rmssd","recovery_score"]

# Use a compact variable set for PCMCI: target + secondary + actionable + key physio
physio_keep = ["total_sleep_minutes","sws_minutes","rem_minutes","light_minutes",
               "sleep_efficiency","sleep_debt_minutes","recovery_score","hrv_rmssd",
               "resting_hr","strain","respiratory_rate"]
physio_keep = [c for c in physio_keep if c in daily.columns]
cols = list(dict.fromkeys(physio_keep + ACTIONABLE))
df = daily[cols].copy().dropna()
# standardize for ParCorr stability
dfz = (df - df.mean())/df.std(ddof=0)
dfz = dfz.loc[:, dfz.std() > 0]

var_names = list(dfz.columns)
controllable_in = [c for c in ACTIONABLE if c in var_names]
print(f"PCMCI on {len(var_names)} vars, {df.shape[0]} days. Controllable: {len(controllable_in)}")

MAX_LAG, ALPHA = 3, 0.10
dataframe = pp.DataFrame(dfz.values.astype(np.float64), var_names=var_names)
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
res = pcmci.run_pcmci(tau_max=MAX_LAG, pc_alpha=0.2)
graph = pcmci.get_graph_from_pmatrix(p_matrix=res["p_matrix"], alpha_level=ALPHA, tau_min=1, tau_max=MAX_LAG)
val, pm = res["val_matrix"], res["p_matrix"]

def parents_of(tgt):
    ti = var_names.index(tgt); out=[]
    for si in range(len(var_names)):
        for lag in range(1, MAX_LAG+1):
            if graph[si,ti,lag]=="-->":
                out.append({"source":var_names[si],"lag":lag,
                            "strength":float(val[si,ti,lag]),"p":float(pm[si,ti,lag]),
                            "controllable":var_names[si] in controllable_in})
    return sorted(out,key=lambda x:abs(x["strength"]),reverse=True)

for tgt in [TARGET]+[s for s in SECONDARY if s in var_names]:
    ps = parents_of(tgt)
    ctrl=[p for p in ps if p["controllable"]]
    print(f"\n### {tgt} — controllable causes (p<{ALPHA}):")
    if ctrl:
        for p in ctrl:
            arrow = "↑" if p["strength"]>0 else "↓"
            print(f"   {p['source'][3:]:38s} lag={p['lag']}  {arrow} strength={p['strength']:+.3f} p={p['p']:.3f}")
    else:
        print("   (none significant)")

# DoWhy ATE on TARGET's controllable parents, adjusting for physio confounders
from dowhy import CausalModel
tgt_ctrl = [p for p in parents_of(TARGET) if p["controllable"]]
physio_conf = [c for c in physio_keep if c != TARGET and c in df.columns]
print(f"\n=== DoWhy ATE on {TARGET} (adjusting for {len(physio_conf)} physio confounders) ===")
results=[]
for p in tgt_ctrl:
    d = df.copy()
    treat = p["source"]
    if p["lag"]>0:
        d[f"{treat}_lag{p['lag']}"]=d[treat].shift(p["lag"]); d=d.dropna(); treat=f"{treat}_lag{p['lag']}"
    conf=[c for c in physio_conf if c in d.columns]
    edges=[f'"{treat}" -> "{TARGET}"']+[f'"{c}" -> "{treat}"' for c in conf]+[f'"{c}" -> "{TARGET}"' for c in conf]
    nodes="; ".join(f'"{n}"' for n in set([treat,TARGET]+conf))
    g=f"digraph {{ {nodes}; {'; '.join(edges)} }}"
    try:
        m=CausalModel(data=d,treatment=treat,outcome=TARGET,common_causes=conf,graph=g)
        ie=m.identify_effect(proceed_when_unidentifiable=True)
        est=m.estimate_effect(ie,method_name="backdoor.linear_regression")
        ate=float(est.value)
        ref=m.refute_estimate(ie,est,method_name="placebo_treatment_refuter",placebo_type="permute",num_simulations=80)
        pp_=float(ref.refutation_result.get("p_value",0))
        robust = pp_>0.05
        results.append({"behavior":p["source"][3:],"lag":p["lag"],"ate_min":round(ate,1),"placebo_p":round(pp_,3),"robust":robust})
        print(f"   {p['source'][3:]:38s} lag={p['lag']}  ATE={ate:+.1f} min  placebo_p={pp_:.3f}  {'ROBUST' if robust else 'fragile'}")
    except Exception as e:
        print(f"   {p['source'][3:]}: ERROR {e}")

Path("data/raw").mkdir(exist_ok=True,parents=True)
json.dump({"target":TARGET,"n_days":int(df.shape[0]),"effects":results}, open("data/raw/_results.json","w"), indent=2, default=str)
print("\ndone")

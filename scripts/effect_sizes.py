import numpy as np, pandas as pd, json
daily = pd.read_parquet("data/raw/daily_export.parquet").sort_index()
CONTROLLABLE=[c for c in daily.columns if c.startswith("do_")]
SYMPTOMS={"do_experienced_brain_fog","do_experienced_fatigue","do_felt_emotionally_and_mentally_stable"}
ACTION=[c for c in CONTROLLABLE if c not in SYMPTOMS]
TARGETS={"total_sleep_minutes":"Total sleep","sws_minutes":"Deep (SWS)","rem_minutes":"REM",
         "sleep_efficiency":"Sleep efficiency %","hrv_rmssd":"HRV","recovery_score":"Recovery %"}

# Effect = mean(outcome | behavior on prior day=1) - mean(=0), for lag 0 and 1.
# Simple, interpretable, in native units. Welch t-test for significance.
from scipy import stats
rows=[]
for beh in ACTION:
    for lag in (0,1):
        b = daily[beh].shift(lag)
        for tgt,lbl in TARGETS.items():
            if tgt not in daily: continue
            y=daily[tgt]
            on=y[b==1].dropna(); off=y[b==0].dropna()
            if len(on)<8 or len(off)<8: continue
            t,p=stats.ttest_ind(on,off,equal_var=False)
            diff=on.mean()-off.mean()
            if p<0.05:
                rows.append({"behavior":beh[3:],"target":lbl,"lag":lag,
                             "delta":round(diff,1),"n_on":len(on),"p":round(p,4)})
res=pd.DataFrame(rows).sort_values(["target","p"])
pd.set_option("display.width",200)
for tgt in TARGETS.values():
    sub=res[res.target==tgt]
    if len(sub):
        print(f"\n### {tgt}")
        for _,r in sub.iterrows():
            unit="min" if "sleep" in tgt.lower() or tgt in("Deep (SWS)","REM") else ("%" if "%" in tgt else "")
            when="same day" if r.lag==0 else "next day"
            sign="+" if r.delta>=0 else ""
            print(f"   {r.behavior:38s} ({when}): {sign}{r.delta} {unit:3s}  (n={r.n_on}, p={r.p})")
json.dump(rows, open("data/raw/_effects_minutes.json","w"), indent=2, default=str)
print(f"\n{len(rows)} significant behavior→sleep effects (p<0.05)")

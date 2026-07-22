"""
Freshness-return ratio R and embodiment-cost channel decomposition
for the relay-allocation trade-off (Command_Layer repo).

R = g*beta/c
  g    : delay steps bought per relay          (relay sweep, exogenous k)
  beta : safe-delivery value per delay step    (delay grid, fixed composition)
  c    : safe-delivery price per relay         (relay sweep)
Relay allocation is net-beneficial for safety iff R > 1.

Channel decomposition: gross embodiment cost = g*beta - tau, mediated through
early attrition / early exposure (linear-probability, product-of-coefficients).
Deterministic: seeds fixed. Requires: data/ from
command_layer_relay_tradeoff_public_repo_v1.
"""
import pandas as pd, numpy as np, numpy.linalg as la, json, sys, pathlib

DATA = pathlib.Path(sys.argv[1] if len(sys.argv)>1 else "data")
NBOOT = 2000

def slope(df,xc,yc):
    x=df[xc].values.astype(float); y=df[yc].values.astype(float)
    x=x-x.mean(); return (x*y).sum()/(x*x).sum()
def ols(X,y):
    X1=np.column_stack([np.ones(len(X)),X]); return la.lstsq(X1,y,rcond=None)[0]

# ---------- beta: value of freshness (fixed composition, exogenous delay) ----
q=pd.read_csv(DATA/'q2_runs.csv')
grid=q[q['stress'].str.startswith('grid_m')].copy()
grid['map_delay']=grid['stress'].str.extract(r'grid_m(\d+)_c')[0].astype(int)
B=grid[(grid['team']=='no_relay_hetero')&(grid['command_mode']=='autonomous')&(grid['map_mode']=='delayed')]
beta=-slope(B,'map_delay','safe_delivery_success')

# ---------- g, c: what a relay buys and what it costs -----------------------
s=pd.read_csv(DATA/'relay_sweep_runs.csv')
kmap={'no_relay_hetero':0,'relay_sparse':1,'balanced_hetero':2,'relay_mid':3,'relay_rich':4}
s['k']=s['team'].map(kmap)
res={}
for st in ['degraded','severe']:
    ss=s[s['stress']==st]
    res[st]=dict(g=-slope(ss,'k','mean_effective_map_delay'),
                 c=-slope(ss,'k','safe_delivery_success'),
                 tau=slope(ss,'k','safe_delivery_success'))

# ---------- bootstrap R ------------------------------------------------------
rng=np.random.default_rng(2026)
Bg=[gr for _,gr in B.groupby('stress')]
Sg={st:[gr for _,gr in s[s['stress']==st].groupby('k')] for st in ['degraded','severe']}
Rb={'degraded':[], 'severe':[]}
for _ in range(NBOOT):
    Bb=pd.concat([gr.sample(len(gr),replace=True,random_state=rng.integers(1e9)) for gr in Bg])
    bb=-slope(Bb,'map_delay','safe_delivery_success')
    for st in ['degraded','severe']:
        sb=pd.concat([gr.sample(len(gr),replace=True,random_state=rng.integers(1e9)) for gr in Sg[st]])
        gb=-slope(sb,'k','mean_effective_map_delay'); cb=-slope(sb,'k','safe_delivery_success')
        Rb[st].append(gb*bb/cb if cb>0 else np.nan)

# ---------- channel decomposition (costs mediated; benefit external) --------
def decompose(ss):
    k=ss['k'].values.astype(float); y=ss['safe_delivery_success'].values.astype(float)
    M=ss[['early_attrition_rate','early_exposure_per_agent_step']].values.astype(float)
    tau=ols(k.reshape(-1,1),y)[1]
    a=[ols(k.reshape(-1,1),M[:,j])[1] for j in range(2)]
    bf=ols(np.column_stack([k,M]),y); return tau,bf[1],a[0]*bf[2],a[1]*bf[3]

out={'beta':beta,'beta_n':len(B)}
rng2=np.random.default_rng(2028)
for st in ['degraded','severe']:
    ss=s[s['stress']==st]
    tau,taup,ia,ie=decompose(ss)
    g,c=res[st]['g'],res[st]['c']; R=g*beta/c
    arr=np.array([x for x in Rb[st] if np.isfinite(x)])
    boots=[]
    groups=[gr for _,gr in ss.groupby('k')]
    for _ in range(NBOOT):
        sb=pd.concat([gr.sample(len(gr),replace=True,random_state=rng2.integers(1e9)) for gr in groups])
        boots.append(decompose(sb))
    Bm=np.array(boots)
    gross=g*beta - tau
    out[st]=dict(g=g,c=c,tau=tau,R=R,
        R_CI=list(np.percentile(arr,[2.5,97.5])), P_R_lt_1=float((arr<1).mean()),
        benefit_gbeta=g*beta, gross_cost=gross,
        cost_attrition=-ia, cost_attrition_CI=[float(-np.percentile(Bm[:,2],97.5)),float(-np.percentile(Bm[:,2],2.5))],
        cost_exposure=-ie,  cost_exposure_CI=[float(-np.percentile(Bm[:,3],97.5)),float(-np.percentile(Bm[:,3],2.5))],
        cost_residual=gross+ia+ie,
        share_attrition=float(-ia/gross), share_exposure=float(-ie/gross))

print(json.dumps(out,indent=2,default=float))

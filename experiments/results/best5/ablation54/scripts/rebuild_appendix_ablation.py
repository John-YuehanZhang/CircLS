import json,glob
CFGS=['full','no_reduce','no_fui','no_schedpar','no_place','no_live','reselect_only']
ROW={'full':'\\sys (full)','no_reduce':'w/o re-selection','no_fui':'w/o first-use initialization',
     'no_schedpar':'w/o reordering \\& parallel','no_place':'w/o mapping','no_live':'w/o last-use freeing',
     'reselect_only':'re-selection only'}
fm={}
for f in glob.glob('fullmetrics/*.json'):
    r=json.loads(open(f).read())
    if r.get('status')=='OK' and isinstance(r.get('metrics'),dict):
        fm.setdefault(r['name'],{})[r['config']]=r['metrics']
common=set(n for n in fm if all(c in fm[n] for c in CFGS))
print(f"# tab:ablation-full  (common-{len(common)} programs, OK in all 7 configs)")
def agg(panel,cfg):
    V1=sum(fm[n][cfg]['V1_volume_blocks'] for n in panel)
    V3=sum(fm[n][cfg]['V3_bbox_volume_blocks'] for n in panel)
    E =sum(fm[n][cfg]['T1_rounds'] for n in panel)
    PK=sum(fm[n][cfg]['S3_tiles_peak'] for n in panel)
    LV=sum(fm[n][cfg]['L1_live_tile_rounds'] for n in panel)
    denom=sum(fm[n][cfg]['S3_tiles_peak']*fm[n][cfg]['T1_rounds'] for n in panel)
    util=LV/denom if denom else 0
    return V1,V3,E,PK,LV,util
b=agg(common,'full')
print("cfg              |  V1    dV%  |  V3bbox  dV%  | exec dE% | peak dP% | live dL% | util")
lines=[]
for c in CFGS:
    V1,V3,E,PK,LV,util=agg(common,c)
    dV=(V1/b[0]-1)*100; d3=(V3/b[1]-1)*100; dE=(E/b[2]-1)*100; dP=(PK/b[3]-1)*100; dL=(LV/b[4]-1)*100
    print(f"{c:16s} | {V1:5.0f} {dV:+5.1f} | {V3:6.0f} {d3:+5.1f} | {E:4.0f} {dE:+5.1f} | {PK:4.0f} {dP:+5.1f} | {LV:5.0f} {dL:+5.1f} | {util:.2f}")
    if c=='full':
        lines.append(f"    \\sys (full) & {V1:.0f} & & {V3:.0f} & & {E:.0f} & & {PK:.0f} & & {LV:.0f} & & {util:.2f} \\\\")
    else:
        lines.append(f"    {ROW[c]} & {V1:.0f} & ${dV:+.1f}$ & {V3:.0f} & ${d3:+.1f}$ & {E:.0f} & ${dE:+.1f}$ & {PK:.0f} & ${dP:+.1f}$ & {LV:.0f} & ${dL:+.1f}$ & {util:.2f} \\\\")
print("\n---- LaTeX tab:ablation-full body ----")
print("\n".join(lines))

# ---- consumable ----
CONS=['bbpssw_4','bbpssw_8','teleport_4','teleport_8','twistedghz_4','twistedghz_8']
CCFG=['full','no_reduce','no_fui','no_schedpar','no_place','no_live']
cons_common=set(n for n in CONS if n in fm and all(c in fm[n] for c in CCFG))
print(f"\n# tab:ablation-consumable  (consumable programs OK in all 6 configs: {sorted(cons_common)})")
def agg2(panel,cfg):
    return sum(fm[n][cfg]['V1_volume_blocks'] for n in panel), sum(fm[n][cfg]['V2_qubit_rounds'] for n in panel)
b2=agg2(cons_common,'full')
clines=[]
for c in CCFG:
    V,Q=agg2(cons_common,c); dV=(V/b2[0]-1)*100; dQ=(Q/b2[1]-1)*100
    print(f"{c:16s} | V1={V:.0f} ({dV:+.1f})  V2={Q:.0f} ({dQ:+.1f})")
    if c=='full': clines.append(f"    \\sys (full) & {V:.0f} & & {Q:.0f} & \\\\")
    else: clines.append(f"    {ROW[c]} & {V:.0f} & ${dV:+.1f}$ & {Q:.0f} & ${dQ:+.1f}$ \\\\")
print("\n---- LaTeX tab:ablation-consumable body ----")
print("\n".join(clines))

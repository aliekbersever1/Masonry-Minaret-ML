#!/usr/bin/env python3
"""R1C6: sqrt(E/rho)-normalized geometry-only linear baseline (NO H**2 scaling).

Uses five original held-out GEOMETRY folds in data/geometry_fold_assignments.xlsx.
All six material realizations of one geometry remain in the same fold.
Each of the 191 unique geometries contributes one group-mean normalized target
when fitting the 6-parameter ordinary least-squares regression.

  c = sqrt((1e6 * E_MPa) / rho_kgm3) [m/s]
  z_i = f_i / c [1/m]
  zhat_i = beta0 + betaH*H + betaD0*D0 + betaDtop*Dtop + betat*t + betaB*B
  fhat_i = c * zhat_i [Hz].

Outer-test performance computed in Hz, on held-out realizations only.
No log transform, inner CV, H**2 scaling, geometric ratios, or test fitting.
"""
import argparse
import csv
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
GEOM = ('H_m', 'D0_m', 'Dtop_m', 't_m', 'B_kaide_m')
FOLD_GEOM = ('H_m', 'D0_m', 'Dtop_m', 't_m', 'B_m')
FREQS = (('f1', 'f1 (Hz)'), ('f2', 'f2 (Hz)'))


def load_sheet1_xlsx(path):
    """Read only sheet 1 from OOXML; no external spreadsheet dependency."""
    with zipfile.ZipFile(path) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            root = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for item in root.findall('m:si', NS):
                strings.append(''.join(t.text or '' for t in item.iter('{'+NS['m']+'}t')))
        root = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        rows = []
        for row in root.findall('.//m:sheetData/m:row', NS):
            obj = {}
            for cell in row.findall('m:c', NS):
                val = cell.find('m:v', NS)
                if val is None:
                    continue
                col = re.match(r'[A-Z]+',cell.attrib['r']).group()
                raw = val.text
                obj[col] = strings[int(raw)] if cell.attrib.get('t') == 's' else raw
            if obj: rows.append(obj)
    headers = rows[0]
    return [{header: row.get(col) for col,header in headers.items()} for row in rows[1:]]


def save_csv(path, rows):
    if not rows:return
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def metrics(y, yp):
    y=np.asarray(y,dtype=float); yp=np.asarray(yp,dtype=float)
    den=np.abs(y)+np.abs(yp)
    return dict(R2=float(r2_score(y,yp)),
        MAE_Hz=float(mean_absolute_error(y,yp)),
        RMSE_Hz=float(np.sqrt(mean_squared_error(y,yp))),
        SMAPE_pct=float(100*np.mean(np.divide(2*np.abs(y-yp),den,out=np.zeros_like(y),where=den>0))),
        Within_20pct_pct=float(100*np.mean(np.abs(yp-y)/np.abs(y)<=.20)))


def draw_paper_style_parity(out,y_pred,fold_ids,summary):
    # Match original manuscript parity plots: fold colors, dashed 1:1,
    # dotted +/-20% bounds, upper-left pooled-OOF metric box, lower-right legend.
    fig,axes=plt.subplots(1,2,figsize=(12.9,6.3),constrained_layout=True)
    cmap=plt.get_cmap('tab10')
    for ax,(i,(name,(_,y,yp))) in zip(axes,enumerate(y_pred.items())):
        for fi in range(1,6):
            mask=fold_ids==fi
            ax.scatter(y[mask],yp[mask],s=23,alpha=.72,color=cmap(fi-1),
                       label=f'Outer fold {fi}',edgecolors='none',rasterized=True)
        # Keep *all* OOF points visible: unconstrained linear OLS can yield negative
        # predicted frequencies for a small number of held-out geometries.
        lower_x=0.0
        lower_y=min(0.0,float(yp.min())*1.10)
        upper=max(float(y.max()),float(yp.max()))*1.05
        xx=np.linspace(lower_x,upper,250)
        ax.plot(xx,xx,linestyle='--',linewidth=1.6,color='black',label='1:1 line')
        ax.plot(xx,1.2*xx,linestyle=':',linewidth=1.35,color='grey',label='±20% bounds')
        ax.plot(xx,.8*xx,linestyle=':',linewidth=1.35,color='grey')
        m=metrics(y,yp)
        text=(f"Pooled OOF R² = {m['R2']:.4f}\n"
              f"MAE = {m['MAE_Hz']:.4f} Hz\n"
              f"RMSE = {m['RMSE_Hz']:.4f} Hz\n"
              f"SMAPE = {m['SMAPE_pct']:.2f}%\n"
              f"Within ±20% = {m['Within_20pct_pct']:.1f}%")
        ax.text(.03,.97,text,transform=ax.transAxes,ha='left',va='top',fontsize=10.5,
                bbox=dict(facecolor='white',alpha=.88,edgecolor='#dedede'))
        ax.set_xlabel(rf'FE reference $f_{{{i+1}}}$ (Hz)',fontsize=12)
        ax.set_ylabel(rf'Grouped outer-test prediction $f_{{{i+1}}}$ (Hz)',fontsize=12)
        ax.set_title(f"({'a' if i==0 else 'b'}) {name} – normalized geometry baseline",fontsize=12)
        ax.set_xlim(lower_x,upper);ax.set_ylim(lower_y,upper)
        ax.set_aspect('equal',adjustable='box')
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=8.5,loc='lower right')
    fig.savefig(out/'R1C6_B_parity_figure.png',dpi=300,bbox_inches='tight')
    fig.savefig(out/'R1C6_B_parity_figure.svg',bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/minaret_frequency_database.xlsx'))
    p.add_argument('--folds',type=Path,default=Path('data/geometry_fold_assignments.xlsx'))
    p.add_argument('--output',type=Path,default=Path('results/mechanical_baseline'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    raw=load_sheet1_xlsx(a.data); groups_file=load_sheet1_xlsx(a.folds)
    require=list(GEOM)+['E (MPa)','Ro (kg/m3)']+[t for _,t in FREQS]
    assert raw and all(k in raw[0] for k in require), 'Input columns mismatch'
    groups=defaultdict(list)
    for index,r in enumerate(raw):
        g=tuple(round(float(r[k]),8) for k in GEOM)
        groups[g].append(index)
    fold_map={};gid_map={}
    for r in groups_file:
        g=tuple(round(float(r[k]),8) for k in FOLD_GEOM)
        assert g not in fold_map, 'Duplicate geometry in assignments'
        fold_map[g]=int(float(r['Outer_Fold']));gid_map[g]=r['Geometry_Group']
    assert len(raw)==1146 and len(groups)==len(fold_map)==191,'Expected 1146 cases/191 geometries'
    assert set(groups)==set(fold_map),'Fold file geometry mismatch'
    assert all(len(ix)==6 for ix in groups.values()),'Each geometry must have six realizations'
    assert all(1<=f<=5 for f in fold_map.values())
    geo_keys=sorted(groups)
    X=np.array([[1]+list(g) for g in geo_keys],dtype=float)
    gfold=np.array([fold_map[g] for g in geo_keys],dtype=int)
    fold_ids=np.array([fold_map[tuple(round(float(r[k]),8) for k in GEOM)] for r in raw])
    assert sorted(np.bincount(gfold)[1:])==[38,38,38,38,39]
    assert all(np.unique(fold_ids[groups[g]]).size==1 for g in geo_keys)
    c=np.sqrt(np.array([float(r['E (MPa)']) for r in raw])*1e6 /
              np.array([float(r['Ro (kg/m3)']) for r in raw]))
    assert np.all(np.isfinite(c)) and np.all(c>0)
    preds={}; normal_rows=[]; fold_rows=[]; coefficient_rows=[]; oof_rows=[]; summary_rows=[]
    for name,col in FREQS:
        y=np.array([float(r[col]) for r in raw]);z=y/c
        yg=np.array([np.mean(z[groups[g]]) for g in geo_keys])
        oof_scaled=np.full(len(geo_keys),np.nan)
        for fi in range(1,6):
            train=gfold!=fi;test=gfold==fi
            b=np.linalg.lstsq(X[train],yg[train],rcond=None)[0]
            oof_scaled[test]=X[test]@b
            for k,v in zip(('intercept','H_m','D0_m','Dtop_m','t_m','B_kaide_m'),b):
                coefficient_rows.append(dict(frequency=name,fit=f'outer_fold_{fi}',term=k,coefficient=float(v)))
        assert not np.isnan(oof_scaled).any()
        b=np.linalg.lstsq(X,yg,rcond=None)[0]
        for k,v in zip(('intercept','H_m','D0_m','Dtop_m','t_m','B_kaide_m'),b):
            coefficient_rows.append(dict(frequency=name,fit='final_all_191_geometries',term=k,coefficient=float(v)))
        zg=dict(zip(geo_keys,oof_scaled))
        yp=np.array([zg[tuple(round(float(r[k]),8) for k in GEOM)] for r in raw])*c
        preds[name]=(col,y,yp)
        cv=[]
        for g in geo_keys:
            sub=z[groups[g]]
            cvv=100*np.std(sub,ddof=0)/np.mean(sub)
            cv.append(cvv)
            normal_rows.append(dict(frequency=name,geometry_group=gid_map[g],outer_fold=fold_map[g],
                **dict(zip(GEOM,g)),n_material_realizations=6,normalized_f_mean=float(np.mean(sub)),
                normalized_f_cv_percent=float(cvv),normalized_f_range_percent=float(100*np.ptp(sub)/np.mean(sub))))
        for fi in range(1,6):
            train=fold_ids!=fi;test=fold_ids==fi
            fold_rows.append(dict(frequency=name,outer_fold=fi,n_test_geometries=int((gfold==fi).sum()),
                n_test_realizations=int(test.sum()),
                **{f'test_{k}':v for k,v in metrics(y[test],yp[test]).items()}))
            # For train diagnostics, fit only on the training groups
            b=np.linalg.lstsq(X[gfold!=fi],yg[gfold!=fi],rcond=None)[0]
            pred_train=np.array([np.dot(np.array([1]+[float(r[k]) for k in GEOM]),b)
                for i,r in enumerate(raw) if train[i]])*c[train]
            fold_rows[-1].update({f'train_{k}':v for k,v in metrics(y[train],pred_train).items()})
        frows=[r for r in fold_rows if r['frequency']==name]
        sm=dict(frequency=name,material_norm_CV_median_pct=float(np.median(cv)),
                material_norm_CV_max_pct=float(np.max(cv)))
        for k in metrics(y,yp):
            vv=np.array([r[f'test_{k}'] for r in frows])
            sm[k+'_mean']=float(np.mean(vv));sm[k+'_std']=float(np.std(vv,ddof=1));
            sm[k+'_pooled']=metrics(y,yp)[k]
        summary_rows.append(sm)
        for idx,r in enumerate(raw):
            g=tuple(round(float(r[k]),8) for k in GEOM)
            oof_rows.append(dict(source_id=r.get('id',''),geometry_group=gid_map[g],
                outer_fold=fold_map[g],frequency=name,**dict(zip(GEOM,g)),
                E_MPa=float(r['E (MPa)']),density_kgm3=float(r['Ro (kg/m3)']),
                wave_speed_sqrt_EPa_over_rho_m_per_s=float(c[idx]),
                FE_reference_Hz=float(y[idx]),normalized_frequency_inverse_m=float(z[idx]),
                predicted_frequency_Hz=float(yp[idx]),
                relative_error_pct=float(100*(yp[idx]-y[idx])/y[idx])))
    save_csv(a.output/'R1C6_B_summary_metrics.csv',summary_rows)
    save_csv(a.output/'R1C6_B_fold_metrics.csv',fold_rows)
    save_csv(a.output/'R1C6_B_group_normalization.csv',normal_rows)
    save_csv(a.output/'R1C6_B_oof_predictions.csv',oof_rows)
    save_csv(a.output/'R1C6_B_final_and_fold_coefficients.csv',coefficient_rows)
    draw_paper_style_parity(a.output,preds,fold_ids,summary_rows)
    for r in summary_rows:print(r)
    print('Completed; results:', a.output)

if __name__=='__main__':main()

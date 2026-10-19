"""Plot empirical figures at publication size; export RGB PNG and LZW TIFF."""
from io import BytesIO
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
OUT = ROOT / 'figures'
CASES = ['UNSW official', 'UNSW random', 'KDDCup99', 'CSE infiltration']
ATTACKS = ['constraint_aware', 'constraint_greedy', 'unconstrained_surrogate']
COLORS = ['#3478a4', '#e3942f', '#2d966f', '#ae5b91', '#666666']


def case_column(frame):
    return ['UNSW official' if d == 'UNSW-NB15' and s == 'temporal_official'
            else 'UNSW random' if d == 'UNSW-NB15'
            else 'CSE infiltration' if d.startswith('cse_cic') else d
            for d, s in zip(frame.dataset, frame.split)]


def save(fig, number):
    OUT.mkdir(exist_ok=True)
    fig.tight_layout(pad=1.0)
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    im = Image.open(buffer).convert('RGB')
    assert 789 <= im.width <= 2250 and im.height <= 2625
    im.save(OUT / f'Fig{number}.png', dpi=(300, 300))
    im.save(OUT / f'Fig{number}.tif', dpi=(300, 300), compression='tiff_lzw')
    plt.close(fig)


def validity_bars(ax, frame, cases, attacks):
    frame = frame[frame.epsilon == .15].copy()
    frame['case'] = case_column(frame)
    means = frame.groupby(['case', 'attack']).constraint_violation_rate.mean()
    x = np.arange(len(cases)); width = .23
    names = dict(zip(ATTACKS, ['Projected', 'Top-ten', 'Unconstrained']))
    for j, attack in enumerate(attacks):
        ax.bar(x+(j-(len(attacks)-1)/2)*width, [means.loc[c, attack] for c in cases],
               width=width, label=names[attack], color=COLORS[j])
    ax.set_xticks(x, cases, rotation=20, ha='right')
    ax.set_ylabel('Constraint violation rate'); ax.set_ylim(0, 1.36); ax.set_yticks(np.arange(0, 1.01, .2))
    ax.legend(frameon=False, loc='upper center', ncol=2)


def panels(axes):
    for label, ax in zip(['(a)', '(b)'], axes):
        ax.text(-.13, 1.05, label, transform=ax.transAxes, fontweight='bold')


def main():
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update({'font.family': 'Arial' if 'Arial' in fonts else 'Times New Roman',
                         'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'legend.fontsize': 8,
                         'axes.spines.top': False, 'axes.spines.right': False})
    full = pd.read_csv(RESULTS/'full_experiment/adversarial_robustness.csv')
    repeated = pd.read_csv(RESULTS/'q1_upgrade/q1_adversarial_by_seed.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    subset = full[(full.dataset == 'UNSW-NB15') & (full.split == 'temporal_official')
                  & (full.attack == 'constraint_aware')]
    for j, (model, part) in enumerate(subset.groupby('model')):
        axes[0].plot(part.epsilon, part.f1, marker='o', ms=3, color=COLORS[j], label=model)
    axes[0].set(xlabel=r'Final coordinate budget $\epsilon$', ylabel='F1 under projected attack', ylim=(0,1.04))
    axes[0].legend(frameon=False, ncol=2, loc='center right', bbox_to_anchor=(1, .30))
    validity_bars(axes[1], full, CASES[:3], [ATTACKS[0], ATTACKS[2]])
    panels(axes); save(fig, 1)

    cross = pd.read_csv(RESULTS/'full_experiment/cross_dataset_transfer.csv')
    pivot = cross.pivot_table(index='train_dataset', columns='test_dataset', values='f1')
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('#ededed')
    im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap=cmap)
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i,j]
            ax.text(j,i,f'{value:.3f}' if np.isfinite(value) else 'Not evaluated',
                    color='white' if np.isfinite(value) else '#555555',ha='center',va='center')
    ax.set_xticks(range(len(pivot.columns)),pivot.columns)
    ax.set_yticks(range(len(pivot)),pivot.index)
    ax.set(xlabel='Test corpus',ylabel='Training corpus')
    fig.colorbar(im,ax=ax,label='Mean transfer F1'); save(fig,2)

    timing = pd.read_csv(RESULTS/'r2_checks/latency_repetitions.csv')
    metrics = pd.read_csv(RESULTS/'r2_checks/metrics.csv')
    metrics = metrics[(metrics.attack=='clean') & (metrics.threshold_rule=='fixed_050')]
    fig, ax = plt.subplots(figsize=(6.8,3.8))
    for j,model in enumerate(['HistGB','RandomForest','ExtraTrees']):
        for defense,marker in [('standard','o'),('adv_aug','s')]:
            values=timing[(timing.model==model)&(timing.defense==defense)&(timing['mode']=='scale_and_predict')].ms_per_flow
            y=metrics[(metrics.model==model)&(metrics.defense==defense)].f1.mean()
            mid=values.median();lo=values.quantile(.05);hi=values.quantile(.95)
            ax.errorbar(mid,y,xerr=[[mid-lo],[hi-mid]],marker=marker,ms=5,capsize=3,
                        color=COLORS[j],label=f"{model}, {'standard' if defense=='standard' else 'augmented'}")
    ax.set(xscale='log',xlabel='Scaler + prediction time (ms/flow; batch averages)',
           ylabel='Clean F1 (balanced UNSW subset)')
    ax.legend(frameon=False,ncol=2,loc='best');ax.grid(alpha=.2);save(fig,3)

    fig,axes=plt.subplots(1,2,figsize=(7.4,3.6))
    data=repeated[(repeated.epsilon==.15)&repeated.attack.isin(ATTACKS[:2])].copy()
    data['case']=case_column(data);means=data.groupby(['case','defense']).f1.mean();x=np.arange(4);width=.19
    for j,(defense,label) in enumerate([('standard','Standard'),('feature_squeeze','Squeezing'),('reject_option','Alert threshold'),('adv_aug','Augmentation')]):
        axes[0].bar(x+(j-1.5)*width,[means.loc[c,defense] for c in CASES],width=width,color=COLORS[j],label=label)
    axes[0].set_xticks(x,CASES,rotation=20,ha='right')
    axes[0].set(ylabel='Mean F1 under constrained attacks',ylim=(0,1.36))
    axes[0].set_yticks(np.arange(0, 1.01, .2))
    axes[0].legend(frameon=False,ncol=2,loc='upper center')
    validity_bars(axes[1],repeated,CASES,ATTACKS);panels(axes);save(fig,4)


if __name__=='__main__':
    main()

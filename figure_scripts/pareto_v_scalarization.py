from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt 
from read_results_utils import parse_config, calc_true_hv, get_hvs, get_scores, pareto_profile, true_pf, top_k_smiles, get_top_k_profile, calc_front_extent
from organize_results import extract_data, create_run_dicts
from plot_utils import set_size, set_style, labels, acq_labels, method_colors, it_colors, it_labels, shapes
import pickle 
import matplotlib.patches as mpatches
from matplotlib.ticker import NullFormatter
import seaborn as sns 
import pandas as pd 

set_style()

filetype='.pdf'
target = 'DRD3'
offt = 'DRD2'

base_dir = Path('results') / f'selective_{target}_dockinglookup'
save_dir = Path(f'figure_scripts/fig_1_{target}_dockinglookup')
save_dir.mkdir(parents=False, exist_ok=True)
obj_configs =  [Path(f'moo_runs/objective/{target}_min.ini'), Path(f'moo_runs/objective/{offt}_max.ini')]
pool_sep = "\t"

k = 0.01

if not (save_dir/'results.pickle').exists(): 
    run_dicts, true_front = create_run_dicts(base_dir, obj_configs, pool_sep=pool_sep, k=k)
    with open(save_dir/'results.pickle', 'wb') as file: 
        pickle.dump(run_dicts, file)
    with open(save_dir/'true_front.pickle', 'wb') as file: 
        pickle.dump(true_front, file)
    print('done')
elif not (save_dir/'true_front.pickle').exists(): 
    _, reference_min, objs, true_scores = calc_true_hv(obj_configs, base_dir, pool_sep)
    true_front = true_pf(true_scores, reference_min)
    with open(save_dir/'true_front.pickle', 'wb') as file: 
        pickle.dump(true_front, file)
    with open(save_dir/'results.pickle', 'rb') as file: 
        run_dicts = pickle.load(file)
else: 
    with open(save_dir/'results.pickle', 'rb') as file: 
        run_dicts = pickle.load(file)
    with open(save_dir/'true_front.pickle', 'rb') as file: 
        true_front = pickle.load(file)


def scatter_top_1():
    _, _, objs, true_scores, all_smiles = calc_true_hv(obj_configs, base_dir, pool_sep)
    df1 = pd.DataFrame(true_scores, columns=[f'-{target}', offt])
    df1['kind'] = 'All Data'
    top_k_smis, top_percent = top_k_smiles(k, all_smiles, true_scores)
    top_smis = [str(smi) for smi in top_k_smis]
    scores = [list(obj(top_smis).values()) for obj in objs]
    scores = np.array(scores).T
    df2 = pd.DataFrame(scores, columns=[f'-{target}', offt])
    df2['kind'] = f'Top {100*top_percent:0.03}%'
    df = pd.concat([df1, df2])

    fig, ax = plt.subplots(1,1, )
    plt.tight_layout()
    

    g = sns.JointGrid(height=1.5, data=df, ratio=4)
    sns.scatterplot(
        data=df, s=1.5, alpha=0.3, 
        ax = g.ax_joint, x=f'-{target}', 
        y=f'{offt}', hue='kind', linewidth=0, 
        palette=['#B4B5B4', method_colors['nds'] ], )
    handles, labels = g.ax_joint.get_legend_handles_labels()
    g.ax_joint.legend(handles=handles[1:], labels=labels[1:], handletextpad=0.1, loc='lower left')
    g.ax_joint.yaxis.get_major_locator().set_params(integer=True)
    g.ax_joint.yaxis.labelpad = 0
    g.ax_joint.xaxis.labelpad = 0

    g.ax_joint.xaxis.get_major_locator().set_params(integer=True)

    sns.kdeplot(
        data=df,
        x=f'-{target}',
        hue='kind',
        palette=['#B4B5B4', method_colors['nds'] ],
        ax=g.ax_marg_x, 
        fill=True, common_norm=False, legend=False
    )
    sns.kdeplot(
        data=df,
        y=offt,
        hue='kind',
        palette=['#B4B5B4', method_colors['nds'] ],
        ax=g.ax_marg_y, 
        fill=True, common_norm=False, legend=False,
    )
    set_size(1.6, 1.6, ax=g.ax_joint)
    g.savefig(save_dir/f'top_k_all.png',bbox_inches='tight', dpi=500, transparent=True)

def hv_profile(run_dicts): 
    pareto_data, scal_data, rand_data = extract_data(run_dicts, key='hv_profile')
    
    x = range(len(pareto_data['ei']['mean']))

    fig, ax = plt.subplots(1,1)
    for metric, entry in pareto_data.items(): 
        ax.plot(x, entry['mean'], label=acq_labels[metric][0], color=method_colors[metric],)
        ax.fill_between(x, entry['mean'] - entry['std'], entry['mean'] + entry['std'], 
                facecolor = method_colors[metric], alpha=0.3
            )

    ax.plot(x, rand_data['mean'], label='Random', color=method_colors['random'],)
    ax.fill_between(x, rand_data['mean'] - rand_data['std'], rand_data['mean'] + rand_data['std'], 
            facecolor = method_colors['random'], alpha=0.3
        )
        
    legend = ax.legend(frameon=False, facecolor="none", fancybox=False, loc='upper left', labelspacing=0.3)
    legend.get_frame().set_facecolor("none")
    # ax.set_ylim([0.8, 1])
    ax.set_ylabel('Hypervolume fraction')
    ax.locator_params(axis='y', nbins=5)
    ylim = ax.get_ylim()
    set_size(1.5,1.5, ax=ax)    
    fig.savefig(save_dir/f'A_1{filetype}',bbox_inches='tight', dpi=200, transparent=True)

    fig, ax = plt.subplots(1,1)
    for metric, entry in scal_data.items(): 
        ax.plot(x, entry['mean'],'--', label=acq_labels[metric][-1], color=method_colors[metric],)
        ax.fill_between(x, entry['mean'] - entry['std'], entry['mean'] + entry['std'], 
                facecolor = method_colors[metric], alpha=0.3
            )
    
    ax.plot(x, rand_data['mean'], label='Random', color=method_colors['random'],)
    ax.fill_between(x, rand_data['mean'] - rand_data['std'], rand_data['mean'] + rand_data['std'], 
            facecolor = method_colors['random'], alpha=0.3
        )    
    legend = ax.legend(frameon=False, facecolor="none", fancybox=False, loc='upper left', labelspacing=0.3)
    legend.get_frame().set_facecolor("none")
    # ax.set_ylim([0.8, 1])
    ax.set_ylim(ylim)
    ax.locator_params(axis='y', nbins=5)
    set_size(1.5,1.5, ax=ax)
    fig.savefig(save_dir/f'A_2{filetype}',bbox_inches='tight', dpi=200, transparent=True)

def igd(run_dicts): 

    pareto_data, scal_data, rand_data = extract_data(run_dicts, 'overall_igd')
    
    w = 0.12
    x = np.array(range(4))/2
    x[3] = x[3] - w/2

    fig, ax = plt.subplots(1,1)

    for i, (p_af, s_af) in enumerate([['nds', 'greedy'], ['ei', 'ei'], ['pi','pi'], ]): 
        ax.bar(x[i]-w/2, pareto_data[p_af]['mean'], width=w, color=method_colors[p_af], align='center', edgecolor=method_colors[p_af], linewidth=0.8)
        ax.errorbar(x[i]-w/2, pareto_data[p_af]['mean'], yerr=pareto_data[p_af]['std'], color='black', capsize=2, capthick=0.8, elinewidth=0.8)
        ax.bar(x[i]+w/2, scal_data[s_af]['mean'], width=w, color=method_colors[s_af], alpha = 1, align='center', hatch='///', edgecolor='black', linewidth=0.8)
        ax.errorbar(x[i]+w/2, scal_data[s_af]['mean'], yerr=scal_data[s_af]['std'], color='black', capsize=2, capthick=0.8, elinewidth=0.8)

    ax.bar(x[3], rand_data['mean'], width=w, color=method_colors['random'], align='center', edgecolor=method_colors['random'], linewidth=0.8)
    ax.errorbar(x[3], rand_data['mean'], yerr=rand_data['std'], color='black', capsize=2, elinewidth=0.8, capthick=0.8)

    ax.set_xticks(x, ['NDS/\nGreedy', 'EHI/\nEI', 'PHI/\nPI',   'Random'])
    ax.locator_params(axis='y', nbins=5)
    ax.set_ylabel('IGD')
    handles = [mpatches.Patch( facecolor='k',edgecolor='black', linewidth=0.4), mpatches.Patch( facecolor='white',hatch='////////',edgecolor='black', linewidth=0.4)]
    legend = plt.legend(handles, ['Pareto', 'Scalarized'], frameon=False, labelspacing=0.3)

    set_size(1.5,1.5, ax=ax)    
    fig.savefig(save_dir/f'B_{target}{filetype}',bbox_inches='tight', dpi=200, transparent=True)

def moving_front(run_dicts, true_front, seed=59, model_seed=37):
    nds_data = [entry for entry in run_dicts if (entry['metric']=='pi' and entry['seed']==str(seed) and entry['model_seed']==str(model_seed))][0]
    greedy_data = [entry for entry in run_dicts if (entry['metric']=='greedy' and entry['seed']==str(seed) and entry['model_seed']==str(model_seed))][0]

    fig, ax = plt.subplots(1,1)
    ax.plot(true_front[:,0], true_front[:,1], marker=shapes[-1], color=it_colors[-1], label=it_labels[-1], linewidth=0.5)

    for i, front in enumerate(nds_data['pf_profile']): 
        if i in [0, 2, 4, 6]:
            continue
        ax.plot(front[:,0], front[:,1], marker=shapes[i], color=it_colors[i], label=it_labels[i], linewidth=0.5)

    ax.set_ylabel(f'{offt}')
    ax.set_xlabel(f'-{target}')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    set_size(2, 2, ax=ax)
    fig.savefig(save_dir/f'C_1_{seed}_{model_seed}{filetype}',bbox_inches='tight', dpi=200, transparent=True)
    
    fig, ax = plt.subplots(1,1)
    ax.plot(true_front[:,0], true_front[:,1], marker=shapes[-1], color=it_colors[-1], label=it_labels[-1], linewidth=0.5)
    
    for i, front in enumerate(greedy_data['pf_profile']): 
        if i in [0, 2, 4, 6]:
            continue
        ax.plot(front[:,0], front[:,1], linestyle='dashed', marker=shapes[i], color=it_colors[i], label=it_labels[i], linewidth=0.5)
    
    ax.set_ylabel(f'{offt}')
    ax.set_xlabel(f'-{target}')

    ax.legend()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    set_size(2, 2, ax=ax)
    fig.savefig(save_dir/f'C_2_{seed}_{model_seed}{filetype}',bbox_inches='tight', dpi=200, transparent=True) 

def number_pf_points(run_dicts):
    nds_data = np.array([[pf.shape[0] for pf in entry['pf_profile']]
                for entry in run_dicts if (entry['metric']=='pi')])
    greedy_data = np.array([[pf.shape[0] for pf in entry['pf_profile']]
                for entry in run_dicts if (entry['metric']=='greedy')])
    
    nds_means = nds_data.mean(axis=0)
    nds_stds = nds_data.std(axis=0)
    greedy_means = greedy_data.mean(axis=0)
    greedy_stds = greedy_data.std(axis=0)

    fig, ax = plt.subplots(1,1)
    x = range(len(greedy_means))
    
    ax.plot(x, greedy_means, label='Greedy', color=it_colors[0],)
    ax.fill_between(x, greedy_means - greedy_stds, greedy_means + greedy_stds, 
            facecolor = it_colors[0], alpha=0.3
        )
    ax.plot(x, nds_means, label='PI', color=it_colors[1],)
    ax.fill_between(x, nds_means - nds_stds, nds_means + nds_stds, 
            facecolor =it_colors[1], alpha=0.3
        )

    set_size(1.5, 1.5, ax=ax)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of PF points')
    ax.legend()
    fig.savefig(save_dir/f'D{filetype}',bbox_inches='tight', dpi=200, transparent=True)

def fraction_top_1_profile(run_dicts): 
    pareto_data, scal_data, rand_data = extract_data(run_dicts, 'top-k-profile')

    x = range(len(pareto_data['ei']['mean']))


    fig, ax = plt.subplots(1,1)
    for metric, entry in pareto_data.items(): 
        ax.plot(x, entry['mean'], label=acq_labels[metric][0], color=method_colors[metric],)
        ax.fill_between(x, entry['mean'] - entry['std'], entry['mean'] + entry['std'], 
                facecolor = method_colors[metric], alpha=0.3
            )

    ax.plot(x, rand_data['mean'], label='Random', color=method_colors['random'],)
    ax.fill_between(x, rand_data['mean'] - rand_data['std'], rand_data['mean'] + rand_data['std'], 
            facecolor = method_colors['random'], alpha=0.3
        )
        
    legend = ax.legend(frameon=False, facecolor="none", fancybox=False, loc='upper left', labelspacing=0.3)
    legend.get_frame().set_facecolor("none")
    ylim = ax.get_ylim()
    ax.set_ylim([ylim[0], ylim[1]+0.05])
    ax.set_ylabel('Fraction of top ~1%')
    ax.locator_params(axis='y', nbins=5)
    ax.set_xticks(x)
    set_size(1.5,1.5, ax=ax)    
    fig.savefig(save_dir/f'top_k_1{filetype}',bbox_inches='tight', dpi=200, transparent=True)

    fig, ax = plt.subplots(1,1)
    for metric, entry in scal_data.items(): 
        ax.plot(x, entry['mean'],'--', label=acq_labels[metric][-1], color=method_colors[metric],)
        ax.fill_between(x, entry['mean'] - entry['std'], entry['mean'] + entry['std'], 
                facecolor = method_colors[metric], alpha=0.3
            )
    
    ax.plot(x, rand_data['mean'], label='Random', color=method_colors['random'],)
    ax.fill_between(x, rand_data['mean'] - rand_data['std'], rand_data['mean'] + rand_data['std'], 
            facecolor = method_colors['random'], alpha=0.3
        )    
    legend = ax.legend(frameon=False, facecolor="none", fancybox=False, loc='upper left', labelspacing=0.3)
    legend.get_frame().set_facecolor("none")
    ax.set_ylim([ylim[0], ylim[1]+0.05])
    ax.set_xticks(x)
    ax.yaxis.set_tick_params(left=False, labelleft=False)
    set_size(1.5,1.5, ax=ax)
    fig.savefig(save_dir/f'top_k_2{filetype}',bbox_inches='tight', dpi=200, transparent=True)

def final_top_1(run_dicts): 

    pareto_data, scal_data, rand_data = extract_data(run_dicts, 'top-k-profile')
    for entry in [*pareto_data.values(), *scal_data.values(), rand_data]: 
        entry['all'] = [profile[-1] for profile in entry['all']]
        entry['mean'] = entry['mean'][-1]
        entry['std'] = entry['std'][-1]

    w = 0.12
    x = np.array(range(4))/2
    x[3] = x[3] - w/2

    fig, ax = plt.subplots(1,1)

    for i, (p_af, s_af) in enumerate([ ['nds', 'greedy'], ['ei', 'ei'], ['pi','pi']]): 
        ax.bar(x[i]-w/2, pareto_data[p_af]['mean'], width=w, color=method_colors[p_af], align='center', edgecolor=method_colors[p_af], linewidth=0.8)
        ax.errorbar(x[i]-w/2, pareto_data[p_af]['mean'], yerr=pareto_data[p_af]['std'], color='black', capsize=2, capthick=0.8, elinewidth=0.8)
        ax.bar(x[i]+w/2, scal_data[s_af]['mean'], width=w, color=method_colors[s_af], alpha = 1, align='center', hatch='///', edgecolor='black', linewidth=0.8)
        ax.errorbar(x[i]+w/2, scal_data[s_af]['mean'], yerr=scal_data[s_af]['std'], color='black', capsize=2, capthick=0.8, elinewidth=0.8)

    ax.bar(x[3], rand_data['mean'], width=w, color=method_colors['random'], align='center', edgecolor=method_colors['random'])
    ax.errorbar(x[3], rand_data['mean'], yerr=rand_data['std'], color='black', capsize=2, elinewidth=0.8, capthick=0.8)

    ax.set_xticks(x, ['NDS/\nGreedy','EHI/\nEI', 'PHI/\nPI',   'Random'])
    ax.locator_params(axis='y', nbins=5)
    ax.set_ylabel('Fraction of top ~1%')
    ylim = ax.get_ylim()
    ax.set_ylim([0, ylim[1]+0.1])
    handles = [mpatches.Patch( facecolor='k',), mpatches.Patch( facecolor='white',hatch='////////',edgecolor='black', linewidth=0.4)]
    legend = plt.legend(handles, ['Pareto', 'Scalarized'], frameon=False)

    set_size(1.5,1.5, ax=ax)    
    fig.savefig(save_dir/f'end_top_k{filetype}',bbox_inches='tight', dpi=200, transparent=True)

def final_front(run_dicts, true_front, scal_metric='greedy', pareto_metric='pi', seed=59, model_seed=37):
    nds_data = [entry for entry in run_dicts if (entry['metric']==pareto_metric and entry['seed']==str(seed) and entry['model_seed']==str(model_seed)) and not entry['scalarization']][0]
    greedy_data = [entry for entry in run_dicts if (entry['metric']==scal_metric and entry['seed']==str(seed) and entry['model_seed']==str(model_seed) and entry['scalarization'])][0]
    
    fig, ax = plt.subplots(1,1)
    ax.plot(true_front[:,0], true_front[:,1], marker=shapes[-1], color=it_colors[-1], linewidth=0.5)

    front = nds_data['pf_profile'][-1]
    ax.plot(front[:,0], front[:,1], marker=shapes[0], color=method_colors[pareto_metric], label=acq_labels[pareto_metric][0], linewidth=0.5)

    ax.set_ylabel(f'{offt}')
    ax.set_xlabel(f'-{target}')
    ax.legend()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()    
    set_size(1.3, 1.3, ax=ax)
    fig.savefig(save_dir/f'final_front_{pareto_metric}_{model_seed}{filetype}',bbox_inches='tight', dpi=200, transparent=True)
    
    fig, ax = plt.subplots(1,1)
    ax.plot(true_front[:,0], true_front[:,1], marker=shapes[-1], color=it_colors[-1], linewidth=0.5)
    
    front = greedy_data['pf_profile'][-1]
    ax.plot(front[:,0], front[:,1],  '--', marker=shapes[3], color=method_colors[scal_metric], label=acq_labels[scal_metric][-1], linewidth=0.5)

    ax.set_ylabel(f'{offt}')
    ax.set_xlabel(f'-{target}')

    ax.legend()    

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    set_size(1.3, 1.3, ax=ax)
    fig.savefig(save_dir/f'final_front_{scal_metric}_scal_{model_seed}{filetype}',bbox_inches='tight', dpi=200, transparent=True) 

def end_extents(run_dicts):
    
    pareto_data, scal_data, rand_data = extract_data(run_dicts, 'final_extent')
    w = 0.12
    x = np.array(range(4))/2
    x[3] = x[3] - w/2

    fig, ax = plt.subplots(1,1)

    for i, (p_af, s_af) in enumerate([['nds', 'greedy'], ['ei', 'ei'], ['pi','pi'] ]): 
        ax.bar(x[i]-w/2, pareto_data[p_af]['mean'], width=w, color=method_colors[p_af], align='center', edgecolor=method_colors[p_af])
        ax.errorbar(x[i]-w/2, pareto_data[p_af]['mean'], yerr=pareto_data[p_af]['std'], color='black', capsize=2, capthick=0.8, elinewidth=0.8)
        ax.bar(x[i]+w/2, scal_data[s_af]['mean'], width=w, color=method_colors[s_af], alpha = 0.8, align='center', hatch='////', edgecolor='black', linewidth=0.5)
        ax.errorbar(x[i]+w/2, scal_data[s_af]['mean'], yerr=scal_data[s_af]['std'], color='black', capsize=2, capthick=0.8, elinewidth=0.8)

    ax.bar(x[3], rand_data['mean'], width=w, color=method_colors['random'], align='center', edgecolor=method_colors['random'])
    ax.errorbar(x[3], rand_data['mean'], yerr=rand_data['std'], color='black', capsize=2, elinewidth=0.8, capthick=0.8)

    ax.set_xticks(x, ['NDS/Greedy', 'EI', 'PI', 'Random'])
    ax.locator_params(axis='y', nbins=5)
    ax.set_ylabel('Extent')
    ylim = ax.get_ylim()
    ax.set_ylim([0, ylim[1]+0.1])
    handles = [mpatches.Patch( facecolor='k',), mpatches.Patch( facecolor='white',hatch='////////',edgecolor='black', linewidth=0.4)]
    legend = plt.legend(handles, ['Pareto', 'Scalarized'], frameon=False)

    set_size(1.5,1.5, ax=ax)    
    fig.savefig(save_dir/f'end_extent{filetype}',bbox_inches='tight', dpi=200, transparent=True)

def fraction_true_front(run_dicts, true_front): 
    
    for entry in run_dicts: 
        pf = entry['final_pf']
        frac = sum([point.tolist() in true_front.tolist() for point in pf])/len(true_front)
        entry['fraction_first_rank'] = frac

    pareto_data, scal_data, rand_data = extract_data(run_dicts, 'fraction_first_rank')

    w = 0.12
    x = np.array(range(4))/2
    x[3] = x[3] - w/2

    fig, ax = plt.subplots(1,1)

    ax.barh(x[3], rand_data['mean'], height=w, color=method_colors['random'], align='center', edgecolor=method_colors['random'])
    ax.errorbar(rand_data['mean'], x[3],  xerr=rand_data['std'], color='black', capsize=2, elinewidth=0.8, capthick=0.8)

    for i, (p_af, s_af) in enumerate([['pi','pi'], ['ei', 'ei'],  ['nds', 'greedy'], ]): 
        ax.barh(x[i]-w/2, pareto_data[p_af]['mean'], height=w, color=method_colors[p_af], align='center', edgecolor=method_colors[p_af], linewidth=0.8)
        ax.errorbar(pareto_data[p_af]['mean'], x[i]-w/2, xerr=pareto_data[p_af]['std'], color='black', capsize=2, capthick=0.8, elinewidth=0.8)
        ax.barh(x[i]+w/2, scal_data[s_af]['mean'], height=w, color=method_colors[s_af], alpha = 1, align='center', hatch='///', edgecolor='black', linewidth=0.8)
        ax.errorbar(scal_data[s_af]['mean'], x[i]+w/2, xerr=scal_data[s_af]['std'], color='black', capsize=2, capthick=0.8, elinewidth=0.8)

    ax.set_yticks(x, ['Greedy\nNDS', 'EI\nEHI','PI\nPHI',   'Random',  ])
    ax.locator_params(axis='x', nbins=5)
    ax.set_xlabel('Fraction of non-dominated points') 
    handles = [mpatches.Patch( facecolor='k',edgecolor='k',), mpatches.Patch( facecolor='white',hatch='////////',edgecolor='black', linewidth=0.4)]
    legend = plt.legend(handles, ['Pareto', 'Scalarized'], frameon=False)

    set_size(1.5,1.5, ax=ax)    
    fig.savefig(save_dir/f'fraction_non_dominated{filetype}',bbox_inches='tight', dpi=200, transparent=True)

def violin_plots(run_dicts, scalarization=False, metric='pi', seed='59', model_seed='37', score_type='all_scores'): 
    run = [entry for entry in run_dicts if entry['metric']==metric and entry['scalarization']==scalarization and entry['model_seed']==model_seed and entry['seed']==seed][0]
    
    data = []
    for it, scores in enumerate(run[score_type]): 
        if it in {0, 2, 4, 6}:
            for score in scores: 
                data.append({
                    'Iteration': it, 
                    'Objective': target,
                    'Value': -score[0],
                })
                data.append({
                    'Iteration': it, 
                    'Objective': offt,
                    'Value': score[1],
                })

    df = pd.DataFrame.from_dict(data)

    fig,ax = plt.subplots(1,1)

    sns.violinplot(ax=ax, x="Iteration", y="Value", hue="Objective",
        data=df, size=0.5, linewidth=0.5,
        palette="Set2", dodge=True,)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='upper left')
    ax.set_ylim([-18, -0])
    ax.set_ylabel('Docking Score')
    ax.set_yticks([-3, -6, -9, -12, -15])
    set_size(1.5, 1.5, ax)
    
    fig.savefig(save_dir/f'violin_{metric}_{scalarization}_{seed}_{score_type}{filetype}',bbox_inches='tight', dpi=200, transparent=True)

def all_hist_2d():
    true_hv, reference_min, objs, true_scores, all_smiles = calc_true_hv(obj_configs, base_dir, pool_sep)

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    nullfmt = NullFormatter()         # no labels

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    axScatter.hist2d(true_scores[:,0], true_scores[:,1], bins=120, cmap=plt.cm.CMRmap.reversed())
    lims = axScatter.get_xlim()

    # now determine nice limits by hand:
    binwidth = 0.25

    bins = np.arange(lims[0], lims[1] + binwidth, binwidth)
    axHistx.hist(true_scores[:,0], bins=bins)
    axHisty.hist(true_scores[:,1], bins=bins, orientation='horizontal')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    axHisty.set_xlim(axHistx.get_ylim())

    
    plt.savefig(save_dir/f'2d_hist.png',bbox_inches='tight', dpi=500)



if __name__ == '__main__': 
    scatter_top_1()
    hv_profile(run_dicts)
    igd(run_dicts)
    # moving_front(run_dicts, true_front, seed=47, model_seed=29)
    # number_pf_points(run_dicts)
    fraction_top_1_profile(run_dicts)
    # for s1, s2 in zip([47, 53, 59, 61, 67], [29, 31, 37, 41, 43]):
    #     final_front(run_dicts, true_front, seed=s1, model_seed=s2)
    # final_front(run_dicts, true_front, pareto_metric='ei', scal_metric='pi', seed=47, model_seed=29)
    final_top_1(run_dicts)
    fraction_true_front(run_dicts, true_front)
    # end_extents(run_dicts)
    # score_type='all_scores'
    # violin_plots(run_dicts, scalarization='True', metric='random', score_type=score_type)
    # violin_plots(run_dicts, scalarization=False, metric='pi', score_type=score_type)
    # violin_plots(run_dicts, scalarization='True', metric='greedy', score_type=score_type)
    # all_hist_2d()
    # final_front(run_dicts, true_front, seed=61, model_seed=41, pareto_metric='ei', scal_metric='ei')



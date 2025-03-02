import os, sys
from typing import List
from collections import Counter

import RNA
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf
import pandas as pd
import seaborn as sns
import numpy as np

import genseq.tools.stats as stats
import genseq.tools.utils as utils
from genseq.tools.dataset import load_msa, DatasetDCA
from genseq.secondary_structure.make_struct import walk_seq

sns.set_theme('paper')

def kde_boxplot(df : pd.DataFrame, df_col : List[str], indel : str, freq : pd.DataFrame, fig_dir : str):
    path_pdf = os.path.join(fig_dir, f'EDA{indel}.pdf')
    pdf = bpdf.PdfPages(path_pdf)

    for col in df_col:
        if col == 'generated':
            continue
        fig, axes = plt.subplots(1, 2, figsize=(18, 5))
        if col == 'ngaps':
            sns.histplot(df, x = col, hue = 'generated', kde = True, multiple="dodge", ax=axes[0])
            sns.boxplot(df, y = col, hue="generated", ax=axes[1])
            fig.suptitle(f'Numeric Feature : {col}', fontsize=16, fontweight='bold')
            fig.subplots_adjust(wspace=0.2)

            fig.savefig(pdf, format='pdf')
            plt.close(fig)
            continue

        if col == 'energy':

            alignment = ['edge', 'center', 'edge']
            width_val = [-1, 1, 1]
            for ic, fcol in enumerate(freq.columns[:-1]):
                for ii in freq[fcol].index:
                    if ic < 1:
                        axes[1].bar(fcol, freq.iat[ii, ic], width=(1/len(freq[fcol].index))*width_val[ii], alpha=0.5, align=alignment[ii], color=f'C{ii}', label=f"{freq.iat[ii, -1]}")
                    else:
                        axes[1].bar(fcol, freq.iat[ii, ic], width=(1/len(freq[fcol].index))*width_val[ii], alpha=0.5, align=alignment[ii], color=f'C{ii}')
            
            sns.histplot(df, x = col, hue = 'generated', kde = True, multiple="dodge", ax=axes[0])

            axes[1].legend(bbox_to_anchor=(1, 1), title="generated")

            fig.suptitle(f'Numeric Feature : Sequences Energy and Nucleic acid average frequency', fontsize=16, fontweight='bold')
            fig.subplots_adjust(wspace=0.2)
            fig.savefig(pdf, format='pdf')
            plt.close(fig)
            continue

        sns.histplot(df, x = col, hue = 'generated', kde = True, multiple="dodge", ax=axes[0])
        sns.scatterplot(df, x = 'ngaps', y = col, hue="generated", ax=axes[1], alpha=0.2)
        
        fig.suptitle(f'Numeric Feature : {col}', fontsize=16, fontweight='bold')
        fig.subplots_adjust(wspace=0.2)
        
        fig.savefig(pdf, format='pdf')
        plt.close(fig)
    
    return pdf


def homology_vs_gaps(chains_file_ref : str, 
                     infile_path : str,
                     chains_file_bias : str,
                     fig_dir : str,
                     indel : bool = False):
    """ Computes the variation of homology depending on the number of gaps in the sequence """
    if indel:
        index = DatasetDCA(infile_path, alphabet='rna').get_indels(threshold_method="mean")
        indel = "_indel"
    else:
        indel = "_gaps"

    df_dico = {'D_MFE' : [], 'ngaps':[], "energy" : [], 'generated' : []}
    freq_gen = Counter()
    freq_data = Counter()
    freq_ref = Counter()
    for seq, (ref_mfe, tmp_mfe), description in walk_seq(infile_path=infile_path, genseqs_path=chains_file_ref):
        if indel == "_indel":
            seq = "".join([seq[i] for i in index])
        freq_gen += Counter(str(seq))
        df_dico['ngaps'].append(str(seq).count('-'))
        df_dico['D_MFE'].append(round(ref_mfe-tmp_mfe, 3))
        df_dico['energy'].append(round(float(description.split('DCAenergy: ')[1].strip()), 3))
        df_dico['generated'].append('yes_unbiased')
    
    for seq, (ref_mfe, tmp_mfe), description in walk_seq(infile_path=infile_path, genseqs_path=infile_path):
        if indel == "_indel":
            seq = "".join([seq[i] for i in index])
        freq_ref += Counter(str(seq))
        df_dico['ngaps'].append(str(seq).count('-'))
        df_dico['D_MFE'].append(round(ref_mfe-tmp_mfe, 3))
        df_dico['energy'].append(0)

        df_dico['generated'].append('no')

    for seq, (ref_mfe, tmp_mfe), description in walk_seq(infile_path=infile_path, genseqs_path=chains_file_bias):
        if indel == "_indel":
            seq = "".join([seq[i] for i in index])
        freq_data += Counter(str(seq))
        df_dico['ngaps'].append(str(seq).count('-'))
        df_dico['D_MFE'].append(round(ref_mfe-tmp_mfe, 3))
        df_dico['energy'].append(round(float(description.split('DCAenergy: ')[1].strip()), 3))
        df_dico['generated'].append('yes_biased')

    df = pd.DataFrame(data=df_dico, index=list(range(len(df_dico['ngaps']))))
    df_freq = pd.DataFrame(data=[freq_gen,freq_ref, freq_data]).apply(lambda x : x.apply(lambda y : y/x.sum() ), axis=1)
    df_freq["generated"] = ['yes_unbiased', 'no','yes_biased']
    pdf = kde_boxplot(df, df.columns,indel=indel, freq=df_freq, fig_dir=fig_dir)
    gaps_coupling_heatmap(chains_file_ref=chains_file_ref, infile_path=infile_path, chains_file_bias=chains_file_bias, pdf=pdf)

    pdf.close()

def gaps_coupling_heatmap(
                     chains_file_ref : str, 
                     infile_path : str,
                     chains_file_bias : str,
                     pdf):
    translate = {0: '-', 1: 'A', 2: 'C', 3: 'G', 4: 'U'}

    dataset_ref = DatasetDCA(infile_path, alphabet='rna')
    dataset_biased = DatasetDCA(chains_file_bias, alphabet='rna')
    dataset_unbiased = DatasetDCA(chains_file_ref, alphabet='rna')

    seqref, wref = utils.one_hot(dataset_ref.data), dataset_ref.weights
    seqbiased, wbiased = utils.one_hot(dataset_biased.data), dataset_biased.weights
    sequnbiased, wunbiased = utils.one_hot(dataset_unbiased.data), dataset_unbiased.weights

    M, L, q = seqref.shape

    f2p_ref = stats.get_freq_two_points(data=seqref, weights=wref)
    f2p_biased = stats.get_freq_two_points(data=seqbiased, weights=wbiased)
    f2p_unbiased = stats.get_freq_two_points(data=sequnbiased, weights=wunbiased)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(f2p_unbiased[:, 0, :, 0], cmap="magma", cbar=True, ax=axes[0], vmin=0,  vmax=1)
    sns.heatmap(f2p_ref[:, 0, :, 0], cmap="magma", cbar=True, ax=axes[1], vmax=1, vmin=0)
    sns.heatmap(f2p_biased[:, 0, :, 0], cmap="magma", cbar=True, ax=axes[2], vmax=1, vmin=0)

    axes[0].set_title("Unbiased generated gaps' couplings")
    axes[1].set_title("Reference data gaps' couplings")
    axes[2].set_title("Biased generated data gaps' couplings")
    
    fig.subplots_adjust(wspace=0.2)
    fig.savefig(pdf, format='pdf')
    plt.close(fig)

    dico_f2p = dict(zip(["Unbiased", 'ref', "biased"],[f2p_unbiased, f2p_ref, f2p_biased]))
    
    for name, f2p in dico_f2p.items():
        fig, axes = plt.subplots(1, 4, figsize=(19, 5))
        for i in range(1, q):
            sns.heatmap(f2p[:, 0, :, i], cmap="magma", cbar=True, ax=axes[i-1], vmin=0, vmax=1)
            axes[i-1].set_title(f"Coupling heatmap -/{translate[i]}")
            axes[i-1].set_xlabel(f"{translate[i]} Position")
            axes[i-1].set_ylabel(f"Gaps (-) position")

            fig.suptitle(f'Couplings heatmap for {name}', fontsize=16, fontweight='bold')

        fig.tight_layout()
        fig.subplots_adjust(wspace=0.2)
        fig.savefig(pdf, format='pdf')
        plt.close(fig)

def heatmap_homology_vs_gaps(chains_file : str, 
                            infile_path : str,
                            bias : bool,
                            indel : bool = False,
                            ):
    if bias:
        bias = '_biased'
    else:
        bias = '_ref'
    
    if indel:
        index = DatasetDCA(infile_path, alphabet='rna').get_indels(threshold_method="mean")
        indel = "_indel"
    else:
        indel = "_gaps"
    fig = plt.figure()
    ax1, ax2, ax3 = fig.subplots(3, 1)
    fig.set_figheight(15)
    fig.set_figwidth(15)

    df_dico = {'D_MFE' : [], 'PPV': [], 'MCC':[], 'ngaps':[]}
    for seq, (ref_mfe, tmp_mfe), dist in walk_seq(infile_path=infile_path, genseqs_path=chains_file, indel_mode=indel):
        if indel== "_indel":
            seq = "".join([seq[i] for i in index])
        df_dico['ngaps'].append(str(seq).count('-'))
        df_dico['D_MFE'].append(round(abs(ref_mfe-tmp_mfe), 3))
        df_dico['MCC'].append(round(dist.MCC, 3))
        df_dico['PPV'].append(round(dist.PPV, 3))

    df = pd.DataFrame(data=df_dico, index=list(range(len(df_dico['ngaps']))))
    MFE_df = pd.crosstab(df['D_MFE'], df['ngaps']).apply(lambda x : x.apply(lambda y : np.log2(y) if y != 0 else 0)) #contingency table
    PPV_df = pd.crosstab(df['PPV'], df['ngaps']).apply(lambda x : x.apply(lambda y : np.log2(y) if y != 0 else 0)) #contingency table
    MCC_df = pd.crosstab(df['MCC'], df['ngaps']).apply(lambda x : x.apply(lambda y : np.log2(y) if y != 0 else 0)) #contingency table

    sns.heatmap(MFE_df, cmap='jet', fmt='d', cbar_kws={'label': 'Number of Occurrences'}, ax=ax1)
    sns.heatmap(PPV_df, cmap='jet', fmt='d', cbar_kws={'label': 'Number of Occurrences'}, ax=ax2)
    sns.heatmap(MCC_df, cmap='jet', fmt='d', cbar_kws={'label': 'Number of Occurrences'}, ax=ax3)

    plt.savefig(rf'/home/mbettiati/LBE_MatteoBettiati/tests/Artificial_rybo_gen_jl/own_dca/output/figures/heatmap_Homology_vs{indel}_gen{bias}.png')
# Plot as a heatmap ?
import os, sys
from typing import List
from collections import Counter

import RNA
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf
import pandas as pd
import seaborn as sns
import numpy as np

from genseq.tools.dataset import load_msa, DatasetDCA
from genseq.secondary_structure.make_struct import walk_seq

sns.set_theme('paper')

def kde_boxplot(df : pd.DataFrame, df_col : List[str], bias : str, indel : str, freq : pd.DataFrame):
    pdf = bpdf.PdfPages(rf'/home/mbettiati/LBE_MatteoBettiati/tests/Artificial_rybo_gen_jl/own_dca/output/figures/EDA{indel}{bias}.pdf')

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

        sns.histplot(df, x = col, hue = 'generated', kde = True, multiple="dodge", ax=axes[0])
        sns.scatterplot(df, x = 'ngaps', y = col, hue="generated", ax=axes[1], alpha=0.4)
        
        fig.suptitle(f'Numeric Feature : {col}', fontsize=16, fontweight='bold')
        fig.subplots_adjust(wspace=0.2)
        
        fig.savefig(pdf, format='pdf')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    for ic, col in enumerate(freq.columns[:-1]):
        for ii in freq[col].index:
            if ic < 1:
                axes[0].bar(col, freq.iat[ii, ic], width=0.5, alpha=0.5, align='center', color=f'C{ii}', label=f"{freq.iat[ii, -1]}")
            else:
                axes[0].bar(col, freq.iat[ii, ic], width=0.5, alpha=0.5, align='center', color=f'C{ii}')

    
    axes[0].legend(bbox_to_anchor=(0, 1.1), title="generated")
    fig.suptitle(f'Numeric Feature : Nucleic acid frequency and Sequences Energy', fontsize=16, fontweight='bold')
    fig.subplots_adjust(wspace=0.2)

    
    fig.savefig(pdf, format='pdf')
    plt.close(fig)
    pdf.close()


def homology_vs_gaps(chains_file : str, 
                     infile_path : str,
                     bias : bool,
                     indel : bool = False):
    """ Computes the variation of homology depending on the number of gaps in the sequence """
    if bias:
        bias = '_biased'
    else:
        bias = '_ref'    
    if indel:
        index = DatasetDCA(infile_path, alphabet='rna').get_indels(threshold_method="mean")
        indel = "_indel"
    else:
        indel = "_gaps"

    df_dico = {'D_MFE' : [], 'ngaps':[], "energy" : [], 'generated' : []}
    freq_gen = Counter()
    freq_data = Counter()
    for seq, (ref_mfe, tmp_mfe), description in walk_seq(infile_path=infile_path, genseqs_path=chains_file):
        if indel == "_indel":
            seq = "".join([seq[i] for i in index])
        freq_gen += Counter(str(seq))
        df_dico['ngaps'].append(str(seq).count('-'))
        df_dico['D_MFE'].append(round(abs(ref_mfe-tmp_mfe), 3))
        df_dico['generated'].append('yes')
    
    for seq, (ref_mfe, tmp_mfe), dist in walk_seq(infile_path=infile_path, genseqs_path=infile_path):
        if indel == "_indel":
            seq = "".join([seq[i] for i in index])
        freq_data += Counter(str(seq))
        df_dico['ngaps'].append(str(seq).count('-'))
        df_dico['D_MFE'].append(round(abs(ref_mfe-tmp_mfe), 3))
        df_dico['generated'].append('no')

    df = pd.DataFrame(data=df_dico, index=list(range(len(df_dico['ngaps']))))
    df_freq = pd.DataFrame(data=[freq_gen, freq_data]).apply(lambda x : x.apply(lambda y : y/x.sum() ), axis=1)
    df_freq["generated"] = ['yes', 'no']
    kde_boxplot(df, df.columns, bias= bias, indel=indel, freq=df_freq)

    

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
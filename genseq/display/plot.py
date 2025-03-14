import os
from collections import Counter
from typing import List

from Bio import SeqIO
import matplotlib.backends.backend_pdf as bpdf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from Bio import SeqIO



def get_summary(all_fam_dir : str):
    template = "{0:<30} {1:<50}"
    for path, directories, files in os.walk(all_fam_dir):
        if files != []:
            for f in files:
                print(f.split('.')[0])
                path_file = os.path.join(path, f)
                seqs = Counter("".join([str(record.seq) for record in SeqIO.parse(path_file, 'fasta')]))
                seqs = {key : val/sum(seqs.values()) for key, val in seqs.items()}

                for key, freq in seqs.items():
                    print(template.format(key, freq))


def plot_energy_vs_gaps(base_outdir : str, path_save : str):
    """ Plots the statistical energy depending on the number of gaps """
    fig = plt.figure()
    ax1, ax2 = fig.subplots(2, 1)
    fig.set_figheight(15)

    for x_vector, y_vector, nuc_count, _dir, length in load_seq(family_dir=base_outdir):
        ax1.scatter(x_vector, y_vector, marker='.', alpha=0.8)
        ax2.bar(list(nuc_count.keys()), list(nuc_count.values()), width=0.5, alpha=0.5, align='center', label=_dir)

    ax1.set_xlabel("Number of gaps")
    ax1.set_ylabel("Statistical energy")
    ax1.set_title("Evolution of the statistical energy depending on the number of gaps")

    ax2.set_ylabel("Overall nucleotide count")
    ax2.set_title("Nucleotide's frequencies in generated VS data sequences")
    
    plt.subplots_adjust(hspace=0.8)
    plt.legend(bbox_to_anchor=(1.1, 1.8))
    
    plt.savefig(path_save)

def load_seq(family_dir : str):
    """ 
    Load the sequences generated from the DCA model, and parse 
    - the number of gaps
    - the DCA statistical energy computed
    """
    for _dir in os.listdir(family_dir):
        if os.listdir(os.path.join(family_dir, _dir)) == []:
            continue

        path_to_seq = os.path.join(family_dir, _dir, "chains_energies.fasta")
        gaps, energy, length = [], [], []
        nuc_count = Counter()
        for index, record in enumerate(SeqIO.parse(path_to_seq, 'fasta')):
            tmp_count = Counter(record.seq)
            energy.append(float(record.description.split('DCAenergy: ')[1].strip()))
            gaps.append(tmp_count['-'])
            nuc_count += tmp_count
            length.append(len(record.seq))

        total = sum(nuc_count.values())
        nuc_count = {key : val/total for key, val in nuc_count.items()}        
        yield gaps, energy, nuc_count, _dir, length


def _plot_scatter_labels(
    ax: plt.Axes,
    data1: np.ndarray,
    pc1: int = 0,
    pc2: int = 1,
    data2: np.ndarray | None = None,
    labels: List[str] | str | None = "Data",
    colors: List[str] | str = "black",
) -> plt.Axes:

    if isinstance(labels, str):
        labels = [labels,]
    if isinstance(colors, str):
        colors = [colors,]
    if labels is None:
        labels = [None, None]
    ax.scatter(data1[:, pc1], data1[:, pc2], color=colors[0], s=50, label=labels[0], zorder=0, alpha=0.3)
    if data2 is not None:
        if len(labels) == 1:
            labels.append("Generated")
        if len(colors) == 1:
            colors.append("red")
        ax.scatter(data2[:, pc1], data2[:, pc2], color=colors[1], label=labels[1], s=20, zorder=2, edgecolor='black', marker='o', alpha=1, linewidth=0.4)
    
    return ax
    
def _plot_hist(
    ax: plt.Axes,
    data1: np.ndarray,
    pc: int,
    data2: np.ndarray | None = None,
    colors: List[str] | str = "black",
    labels: List[str] | str | None = "Data",
    orientation='vertical',
) -> plt.Axes:
    if isinstance(labels, str):
        labels = [labels,]
    if labels is None:
        labels = [None, None]
    if isinstance(colors, str):
        colors = [colors,]
    ax.hist(data1[:, pc], bins=40, color=colors[0], histtype='step', label=labels[0], zorder=0, density=True, orientation=orientation, lw=1)
    if data2 is not None:
        if len(labels) == 1:
            labels.append("Generated")
        if len(colors) == 1:
            colors.append("red")
    if data2 is not None:
        ax.hist(data2[:, pc], bins=40, color=colors[1], histtype='step', label=labels[1], zorder=1, density=True, orientation=orientation, lw=1.5)
    ax.axis('off')
    
    return ax
    
def plot_PCA(
    fig: plt.figure,
    data1: np.ndarray,
    pc1: int = 0,
    pc2: int = 1,
    data2: np.ndarray | None = None,
    labels: List[str] | str = "Data",
    colors: List[str] | str = "black",
    title: str | None = None,
) -> plt.figure:
    """Makes the scatter plot of the components (pc1, pc2) of the input data and shows the histograms of the components.

    Args:
        fig (plt.figure): Figure to plot the data.
        data1 (np.ndarray): Data to plot.
        pc1 (int, optional): First principal direction. Defaults to 0.
        pc2 (int, optional): Second principal direction. Defaults to 1.
        data2 (np.ndarray | None, optional): Data to be superimposed to data1. Defaults to None.
        labels (List[str] | str, optional): Labels to put in the legend. Defaults to "Data".
        colors (List[str] | str, optional): Colors to be used. Defaults to "black".
        title (str | None, optional): Title of the plot. Defaults to None.

    Returns:
        plt.figure: Updated figure.
    """
    
    gs = GridSpec(4, 4)
    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
    ax_hist_x = fig.add_subplot(gs[0, 0:3])
    ax_hist_y = fig.add_subplot(gs[1:4, 3])
        
    ax_scatter = _plot_scatter_labels(
        ax=ax_scatter,
        data1=data1,
        pc1=pc1,
        pc2=pc2,
        data2=data2,
        labels=None,
        colors=colors,
    )
    ax_hist_x = _plot_hist(
        ax=ax_hist_x,
        data1=data1,
        pc=pc1,
        data2=data2,
        colors=colors,
        labels=labels,
        orientation='vertical',
    )
    ax_hist_y = _plot_hist(
        ax=ax_hist_y,
        data1=data1,
        pc=pc2,
        data2=data2,
        colors=colors,
        labels=None,
        orientation='horizontal',
    )
    
    ax_scatter.set_xlabel(f"PC {pc1 + 1}")
    ax_scatter.set_ylabel(f"PC {pc2 + 1}")
    fig.suptitle(title)
    fig.legend(fontsize=12, bbox_to_anchor=(1, 1));
    h, l = ax_scatter.get_legend_handles_labels()
    
    return fig
    
    
def plot_pearson_sampling(
    ax: plt.Axes,
    checkpoints: np.ndarray,
    pearsons: np.ndarray,
    pearson_training: np.ndarray | None = None
):
    
    if pearson_training:
        ax.axhline(y=pearson_training, ls="dashed", color="red", label="Training", lw=1, zorder=0)
        annotation_text = f"Training: {pearson_training:.3f}\nSampling: {pearsons[-1]:.3f}"
    else:
        annotation_text = annotation_text = f"Sampling: {pearsons[-1]:.3f}"
    ax.plot(checkpoints, pearsons, "-o", label="Resampling", lw=0.5, color="royalblue", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("Sampling time [sweeps]")
    ax.set_ylabel(r"Pearson $C_{ij}(a,b)$")
    ax.annotate(annotation_text, xy=(0.95, 0.53), xycoords='axes fraction',
                verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    ax.legend()
    return ax


def plot_autocorrelation(
    ax: plt.Axes,
    checkpoints: np.ndarray,
    autocorr: np.ndarray,
    gen_seqid: np.ndarray,
    data_seqid: np.ndarray
) -> plt.Axes:
    """Plots the time-autocorrelation curve of the sequence identity and the generated and data sequence identities.
    
    Args:
        ax (plt.Axes): Axes to plot the data.
        checkpoints (np.ndarray): Checkpoints of the sampling.
        autocorr (np.ndarray): Time-autocorrelation of the sequence identity.
        gen_seqid (np.ndarray): Sequence identity of the generated data.
        data_seqid (np.ndarray): Sequence identity of the data.
        
    Returns:
        plt.Axes: Updated axes.
    """
    ax.plot(checkpoints, autocorr, "-o", c="royalblue", lw=0.5, label="Time-autocorrelation", zorder=2)
    ax.axhline(y=gen_seqid, color="navy", lw=1, ls="dashed", label="Generated seqID", zorder=1)
    ax.axhline(y=data_seqid, color="red", lw=1, ls="dashed", label="Data seqID", zorder=0)
    ax.set_xlabel(r"$\tau$ [sweeps]")
    ax.set_ylabel(r"$\langle \mathrm{SeqID}(t, t-\tau) \rangle$")
    ax.legend();

    if np.any(checkpoints[autocorr <= gen_seqid]):
        mixing_time = checkpoints[autocorr <= gen_seqid][0]
        print(f"Mixing time: {mixing_time} sweeps")
        annotation_text = r"$\tau_{\mathrm{mix}}$" + f": {mixing_time} sweeps"
        plt.annotate(annotation_text, xy=(0.95, 0.65), xycoords='axes fraction',
                verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    else:
        print(f"The mixing time could not be computed within {checkpoints[-1]} sweeps")
    
    return ax


def plot_scatter_correlations(
    ax: plt.Axes,
    Cij_data: np.ndarray,
    Cij_gen: np.ndarray,
    Cijk_data: np.ndarray,
    Cijk_gen: np.ndarray,
    pearson_Cij: float,
    pearson_Cijk: float
) -> plt.Axes:
    """Plots the scatter plot of the data and generated Cij and Cijk values.
    
    Args:
        ax (plt.Axes): Axes to plot the data. Must have 2 subplots.
        Cij_data (np.ndarray): Data Cij values.
        Cij_gen (np.ndarray): Generated Cij values.
        Cijk_data (np.ndarray): Data Cijk values.
        Cijk_gen (np.ndarray): Generated Cijk values.
        pearson_Cij (float): Pearson correlation coefficient of Cij.
        pearson_Cijk (float): Pearson correlation coefficient of Cijk.
        
    Returns:
        plt.Axes: Updated axes.
    """
    if len(ax) != 2:
        raise ValueError("The axes must have 2 subplots")
    
    color_line = "#50424F"
    color_scatter = "#FF6275"

    x = np.linspace(Cij_data.min(), Cij_data.max(), 100)
    ax[0].scatter(Cij_data, Cij_gen, alpha=0.5, color=color_scatter)
    ax[0].plot(x, x, ls="dashed", color=color_line)
    ax[0].set_xlabel(r"$C_{ij}$ data")
    ax[0].set_ylabel(r"$C_{ij}$ generated")

    x = np.linspace(Cijk_data.min(), Cijk_data.max(), 100)
    ax[1].scatter(Cijk_data, Cijk_gen, alpha=0.5, color=color_scatter)
    ax[1].plot(x, x, ls="dashed", color=color_line)
    ax[1].set_xlabel(r"$C_{ijk}$ data")
    ax[1].set_ylabel(r"$C_{ijk}$ generated")

    ax[0].annotate(r"$\rho=$" + f"{pearson_Cij:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                    verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))

    ax[1].annotate(r"$\rho=$" + f"{pearson_Cijk:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                    verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))
    
    return ax
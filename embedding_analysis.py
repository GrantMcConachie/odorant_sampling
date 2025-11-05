"""
This script does PCA on the graph or the mordred features to visualize the
data.

@author: grant
"""

import math
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

import umap
from sklearn import decomposition
from sklearn.cluster import KMeans

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


def get_rd_labels(smiles_list):
    """
    Gets the rdkit descriptors for labeling molecules by their functional
    groups

    Args:
        smiles_list (list) - List of all the smiles

    Returns:
        rd_desc (list) - List of all the rd descriptors
    """
    rd_desc = []
    list_rd_desc = [x[0] for x in Descriptors._descList if "fr" in x[0]]
    calc_rd = MoleculeDescriptors.MolecularDescriptorCalculator(
        [x[0] for x in Descriptors._descList if "fr" in x[0]]
    )
    for smiles in tqdm(smiles_list, desc="Getting rd descriptors"):
        mol = Chem.MolFromSmiles(smiles)
        rd_result = np.array(calc_rd.CalcDescriptors(mol))
        rd_desc.append(rd_result)

    return rd_desc, list_rd_desc


def z_score(data_matrix):
    """
    Throws away any feature with 0 standard deviation then z-scores the data

    Args:
        data_matrix (numpy.ndarray) - raw data matrix

    Return:
        Z scored version of the datamatrix with no 0 std features
    """
    zero_std = np.where(data_matrix.std(axis=0) == 0)[0]
    data_matrix = np.delete(data_matrix, zero_std, axis=1)
    return (data_matrix - data_matrix.mean(axis=0)) / data_matrix.std(axis=0)


def pca(data_matrix):
    """
    Runs PCA on the data matrix

    Args:
        data_matrix (numpy.ndarray) - mean centerd data matrix

    Returns:
        transformed_data (numpy.ndarray) - data projected onto its principal 
          components
        clf.explained_variance_ratio_ (list?) - variance explaiend by each PC
    """
    clf = decomposition.PCA()
    transformed_data = clf.fit_transform(data_matrix)
    return transformed_data, clf.explained_variance_ratio_


def umap_proj(data_matrix, three_d):
    """
    Does a UMAP dimensionality reduction

    Args:
        data_matrix (numpy.ndarray) - mean centerd data matrix

    Returns:
        umap projections
    """
    if three_d:
        n_components = 3
    else:
        n_components = 2

    reducer = umap.UMAP(random_state=12345, n_components=n_components)
    return reducer.fit_transform(data_matrix)


def gen_data_labels(rd_desc, list_rd_desc):
    """
    Generates labels for plotting

    Args:
        rd_desc (list) - List of all the molecules functional groups
        list_rd_desc (list) - Names of each of the functional groups
    """
    # init
    data_labels = []

    # Loop through all of the data
    for i in rd_desc:
        desc_count = len(i) - (i == False).sum()
        if desc_count == 0:
            data_labels.append('None')
        elif desc_count == 1:
            data_labels.append(list_rd_desc[np.argmax(i)])
        else:
            data_labels.append("multi")

    return data_labels


def plot_components(projections, variance_explained, data_labels):
    """
    Plots the first two PCs / UMAP dimensions
    """
    # PCA
    if variance_explained is not None:
        fig, axs = plt.subplots(1, 2, layout="constrained")
        for label in np.unique(data_labels):
            points = [i for i, j in enumerate(data_labels) if label == j]
            axs[0].scatter(projections[points,0], projections[points,1], alpha=0.5, label=label)

        axs[0].set_xlabel(f"PC 1 ({variance_explained[0]*100:0.3f}%)")
        axs[0].set_ylabel(f"PC 2 ({variance_explained[1]*100:0.3f}%)")
        axs[1].bar(np.arange(len(variance_explained[:10])), variance_explained[:10])
        axs[1].set_xlabel("PC")
        axs[1].set_ylabel("Variance Explained")
        axs[0].legend()

    # UMAP
    else:
        fig, axs = plt.subplots(layout="constrained")
        for label in np.unique(data_labels):
            points = [i for i, j in enumerate(data_labels) if label == j]
            axs.scatter(projections[points,0], projections[points,1], alpha=0.5, label=label)
        axs.set_xlabel("umap 1")
        axs.set_ylabel("umap 2")
        axs.legend()


def plot_components_3d(projections, variance_explained, rd_desc, list_rd_desc, data_labels):
    """
    Plots the first two PCs / UMAP dimensions
    """
    # PCA
    if variance_explained is not None:
        fig = plt.figure()
        ax1 = fig.add_subplot(projection='3d')
        for label in list_rd_desc:
            points = [i for i, j in enumerate(data_labels) if label == j]
            ax1.scatter(projections[points,0], projections[points, 1], projections[points, 2], alpha=0.5, label=label)

        ax1.set_xlabel(f"PC 1 ({variance_explained[0]*100:0.3f}%)")
        ax1.set_ylabel(f"PC 2 ({variance_explained[1]*100:0.3f}%)")
        ax1.set_zlabel(f"PC 3 ({variance_explained[2]*100:0.3f}%)")
        # ax2 = fig.add_subplot()
        # ax2.bar(np.arange(len(variance_explained[:10])), variance_explained[:10])
        # ax2.set_xlabel("PC")
        # ax2.set_ylabel("Variance Explained")
        # ax1.legend()

    # UMAP
    else:
        fig = plt.figure()
        axs = fig.add_subplot(projection='3d')
        for label in list_rd_desc:
            points = [i for i, j in enumerate(data_labels) if label == j]
            axs.scatter(projections[points,0], projections[points,1], projections[points,2], alpha=0.3, label=label)
        axs.set_xlabel("umap 1")
        axs.set_ylabel("umap 2")
        axs.set_zlabel("umap 3")
        # fig.colorbar(axs, orientation='vertical')


def plot_hist_feats(rd_desc):
    """
    Plots a histogram of the number a features each molecule has.

    Args:
        rd_desc (list) - Matrix of all the descriptors in the dataset.
    """
    desc_count = [len(i) - (i == False).sum() for i in rd_desc]
    plt.hist(desc_count, edgecolor="black", bins=np.arange(max(desc_count)))
    plt.xlabel("Number of present physiochemical descritors")
    plt.ylabel("Features")
    plt.show() 


def plot_all_within_one_class(projections, rd_desc, list_rd_desc, frags, labels, three_d=False):
    """
    Plots every point in grey except for one class that you want to show
    """
    if three_d:
        subplot_kw = dict(projection='3d')
    else:
        subplot_kw = None

    fig, axs = plt.subplots(3, 3, subplot_kw=subplot_kw)
    for group, ax, label in zip(frags, axs.reshape(-1), labels):
        label_nums = [i for i, j in enumerate(list_rd_desc) if j in group]
        in_group_idx = np.unique(np.where(np.array(rd_desc)[:,label_nums] != 0)[0]) # subset data correspoding to labels
        data_in_group = projections[in_group_idx]
        data_out_group = np.delete(projections, in_group_idx, axis=0) 
        if three_d:
            ax.scatter(data_in_group[:,0], data_in_group[:,1], data_in_group[:,2], c="#CB2027", s=8, alpha=0.8)
            ax.scatter(data_out_group[:,0], data_out_group[:,1], data_out_group[:,2], c="#464F51", s=3, alpha=0.3)
            ax.zaxis.set_major_locator(ticker.NullLocator())

        else:
            ax.scatter(data_in_group[:,0], data_in_group[:,1], c="#CB2027", s=8, alpha=0.5)
            ax.scatter(data_out_group[:,0], data_out_group[:,1], c="#464F51", s=3, alpha=0.1)

        ax.set_title(label)
        ax.spines[['right', 'top']].set_visible(False)
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.set_ylabel('umap 2')
        ax.set_xlabel('umap 1')

    fig.suptitle("RDKit embedding Goodscents and human odorants")
    # fig.set_size_inches(14, 12)

    axs[-1, -1].axis('off')


def plot_two_datasets(data_matrix, dataset_labels, three_d=False):
    """
    Combines datasets and plots them in one space Goodscents first
    """
    # loop thorugh labels
    labels = []
    for i in dataset_labels:
        if i == "['human', 'gslf']":
            labels.append("#53B3CB")
        elif 'gslf' in i:
            labels.append("#464F51")
        elif 'human' in i:
            labels.append("#EA638C")

    # z-score data
    data_matrix_z = z_score(data_matrix)

    # project in PC space and UMAP space
    projections, variance_explained = pca(data_matrix_z)
    projections_umap = umap_proj(projections, three_d) # NOTE: This is the PC space rotation in UMAP

    # plotting
    if three_d:
        fig, axs = plt.subplots(subplot_kw=dict(projection='3d'))
        axs.scatter(projections[:,0], projections[:,1], projections[:,2], c=labels, s=10, alpha=0.5)
        p1 = mpatches.Patch(color="#EA638C", label='Human')
        p2 = mpatches.Patch(color="#464F51", label='Goodscents')
        p3 = mpatches.Patch(color="#53B3CB", label="Both")
        axs.set_xlabel("dim 1")
        axs.set_ylabel("dim 2")
        axs.set_zlabel("dim 3")
        axs.legend(handles=[p1,p2,p3])
        axs.set_title("Mordred embeddings (PC space)")

    else:
        fig, axs = plt.subplots()
        axs.scatter(projections_umap[:,0], projections_umap[:,1], c=labels, s=10, alpha=0.5)
        p1 = mpatches.Patch(color="#EA638C", label='Human')
        p2 = mpatches.Patch(color="#464F51", label='Goodscents')
        p3 = mpatches.Patch(color="#53B3CB", label="Both")
        axs.set_xlabel("dim 1")
        axs.set_ylabel("dim 2")
        axs.legend(handles=[p1,p2,p3])
        axs.set_title("RDkit embeddings")


def pick_n_odors(projections, frags, labels, list_rd_desc, rd_desc, smiles, n, dataset_labels):
    """
    Picks n odors that are maximally far apart
    """
    n_clusters = 5
    centroid_dat = {}
    fig, axs = plt.subplots(3, 3)

    # Loop through labels
    for group, ax, label in zip(frags, axs.reshape(-1), labels):
        label_nums = [i for i, j in enumerate(list_rd_desc) if j in group]
        in_group_idx = np.unique(np.where(np.array(rd_desc)[:,label_nums] != 0)[0]) # subset data correspoding to labels
        human_odors_idx = np.array([i for i in in_group_idx if 'human' in dataset_labels[i]])
        gslf_odors_idx = np.array([i for i in in_group_idx if 'gslf' in dataset_labels[i]])
        all_odors = projections[in_group_idx]
        human_odors = projections[human_odors_idx]
        gslf_odors = projections[gslf_odors_idx]
        
        # fit a kmeans and find centroids
        kmeans = KMeans(n_clusters=n_clusters, random_state=12345).fit(all_odors)
        centroids = kmeans.cluster_centers_

        # find n points closest to each cluster
        # For each centroid, get indices of closest n human_odors
        dists_human = np.linalg.norm(human_odors[None,:,:] - centroids[:,None,:], axis=2)
        closest_human_indices = np.argsort(dists_human, axis=1)[:, :n]  # (n_clusters, n)

        # For each centroid, get indices of closest n non-human_odors
        dists_gslf = np.linalg.norm(gslf_odors[None,:,:] - centroids[:,None,:], axis=2)
        closest_gslf_indices = np.argsort(dists_gslf, axis=1)[:, :n]

        # Now grab the 2D coordinates for those points
        chosen_human_pts = human_odors[closest_human_indices]
        chosen_gslf_pts = gslf_odors[closest_gslf_indices]

        # print closest smiles per centroid and put into csv file
        print(f'Chosen {label}')
        for i in range(len(centroids)):
            final_human_odors = [smiles[human_odors_idx[j]] for j in closest_human_indices[i]]
            final_gslf_odors = [smiles[gslf_odors_idx[j]] for j in closest_gslf_indices[i]]
            
            print(f'centroid {i}')
            print('Human odorants:')
            print(final_human_odors)
            print('gslf odorants:')
            print(final_gslf_odors)
            print('\n')

            # append to a data dict
            if 'centroid' not in centroid_dat.keys():
                centroid_dat['centroid'] = []
            centroid_dat['centroid'].append(f'{label} {i}')

            for j in range(n):
                if f'pick {j} human' not in centroid_dat.keys():
                    centroid_dat[f'pick {j} human'] = []
                if f'pick {j} gslf' not in centroid_dat.keys():
                    centroid_dat[f'pick {j} gslf'] = []
                centroid_dat[f'pick {j} human'].append(final_human_odors[j])
                centroid_dat[f'pick {j} gslf'].append(final_gslf_odors[j])

        # Assuming n_clusters is known
        palette = plt.get_cmap('tab10')  # you can choose any colormap, e.g. 'tab20', 'Set2', etc.

        # plot all points
        ax.scatter(projections[:,0], projections[:,1], c="#464F51", s=2, alpha=0.08)

        for idx in range(n_clusters):
            # Pick a color for this centroid
            color = palette(idx % palette.N)
            
            # Each centroid, its human points, and its gslf points get the same color
            centroid = centroids[idx]
            human_pts = chosen_human_pts[idx]  # shape (n, 2) for this centroid
            gslf_pts = chosen_gslf_pts[idx]    # shape (n, 2) for this centroid
            
            ax.scatter(human_pts[:,0], human_pts[:,1], c=[color], s=45, marker='*', alpha=1, label=f'human (cluster {idx+1})')
            ax.scatter(gslf_pts[:,0], gslf_pts[:,1], c=[color], s=45, marker='o', alpha=1, label=f'gslf (cluster {idx+1})')
            ax.scatter(centroid[0], centroid[1], c=[color], s=45, marker='x', alpha=1, label=f'centroid {idx+1}')

        # Optional: to avoid legend duplication, create custom legend handles
        legend_elements = [
            Line2D([0], [0], marker='*', color='w', label='Human', markerfacecolor='k', markersize=12),
            Line2D([0], [0], marker='o', color='w', label='GSLF', markerfacecolor='k', markersize=8),
            Line2D([0], [0], marker='x', color='w', label='Centroid', markerfacecolor='k', markeredgecolor='k', markersize=10)
        ]
        ax.legend(handles=legend_elements)

        # Axis formatting
        ax.set_title(label)
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())

    axs[-1, -1].axis('off')

    # save dataframe
    df = pd.DataFrame(centroid_dat)
    df.to_csv('/Users/grantmcconachie/Desktop/research/TORI_PROJECTS/kmeans_on_odor_space/chosen_odors/chosen_odors_kmeans.csv', index=False)


def main(data_matrix, rd_desc, list_rd_desc, smiles, dataset_labels, three_d=False):
    """
    Run the entire pipeline
    """
    # z-score data
    data_matrix_z = z_score(data_matrix)

    # project in PC space and UMAP space
    projections, variance_explained = pca(data_matrix_z)
    projections_umap = umap_proj(projections, three_d) # NOTE: This is the PC space rotation in the UMAP

    # Generate data labels
    data_labels = gen_data_labels(rd_desc, list_rd_desc)

    # plot
    # plot_components(projections, variance_explained, data_labels)
    # plot_components(projections_umap, None, data_labels)

    important_fragments = [
        ["fr_NH0", "fr_NH1", "fr_NH2", "fr_Ar_NH", "fr_N_O"], # Primary, secondary, and terciary amines, aromatic amines, hydroxylamines
        ["fr_lactone"], # lactones
        ["fr_Al_COO", "fr_Ar_COO", "fr_COO", "fr_COO2"], # aliphatic, aromatic, and 2 versions of carboxylic acids
        ["fr_prisulfonamd", "fr_sulfide", "fr_sulfonamd", "fr_sulfone"], # sulfur compounds
        ["fr_ketone"], # ketones
        ["fr_ArN", "fr_Ar_COO", "fr_Ar_N", "fr_Ar_NH", "fr_Ar_OH"], # aromatics
        ["fr_Al_OH", "fr_Ar_OH", "fr_N_O"], # hydroxyls (alcohols)
        ["fr_ester"], # esters
    ]
    labels = ["Amines", "Lactones", "Carboxylic Acids", "Sulfur", "Ketones", "Aromatics", "Hydroxyls", "Esters"]

    # does the kmeans
    pick_n_odors(projections_umap, important_fragments, labels, list_rd_desc, rd_desc, smiles, 5, dataset_labels)

    # generate the functional group plots
    plot_all_within_one_class(projections_umap, rd_desc, list_rd_desc, important_fragments, labels, three_d)

    # generate the plot with the two datasets
    plot_two_datasets(data_matrix, dataset_labels, three_d=False)

    plt.show()


if __name__ == '__main__':
    # load data
    df = pd.read_csv('/Users/grantmcconachie/Desktop/research/TORI_PROJECTS/kmeans_on_odor_space/best_smiles.csv')
    smiles_list = list(df['top ranked smiles'])
    label = list(df['label'])

    # remove first odor
    smiles_list.pop(0)
    label.pop(0)

    # embed odors into rdkit descriptor vectors
    data_matrix = []
    functional_groups = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        descriptor_dict = Descriptors.CalcMolDescriptors(mol)
        desc_vector = list(descriptor_dict.values())
        data_matrix.append(desc_vector)
    
    data_matrix = np.array(data_matrix)

    # remove nans
    nan_cols = np.unique(np.where(np.isnan(data_matrix))[1])
    data_matrix = np.delete(data_matrix, nan_cols, axis=1)

    # get functional group information
    rd_desc, list_rd_desc = get_rd_labels(smiles_list)

    main(data_matrix, rd_desc, list_rd_desc, smiles_list, label, three_d=False)

    plt.show()

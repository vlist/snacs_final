import igraph as ig
import leidenalg
import louvain

import datasets.cosnology as cosnology
import datasets.email_enron as email_enron
import datasets.facebook_combined as facebook_combined
import datasets.brightkite_edges as brightkite
import datasets.roadnet_CA as roadnet_CA
import datasets.wiki_vote as wiki_vote

import networkx as nx

import matplotlib.pyplot as plt

SEED = 42
import random
random.seed(SEED)

graphs = [cosnology.get_graph(), email_enron.get_graph(), facebook_combined.get_graph(), brightkite.get_graph(), wiki_vote.get_graph() ]
graphs = [ig.Graph.from_networkx(g) for g in graphs]
graph_names = ['Cosnology', 'Email Enron', 'Facebook', 'Brightkite', 'Wiki Vote' ]

from datasets import cosnology
from datasets import email_enron
from tqdm import tqdm



def run(graph, graph_name):
    G = graph
    resolutions_mod = [0.40 + 0.02 * d for d in range(61)] # 0.40 ... 1.60
    resolutions_cpm = [0.04 + 0.02 * d for d in range(48)] # 0.04 ... 0.98


    def leiden_fn(*args, **kwargs):
        partition = leidenalg.find_partition(*args, **kwargs)
        return partition

    def louvain_fn(*args, **kwargs):
        partition = louvain.find_partition(*args, **kwargs)
        return partition


    # Run louvain and leiden algorithms for all resolutions with the Modularity quality function
    coms_louvain_mod = list([louvain_fn(G, louvain.RBConfigurationVertexPartition, resolution_parameter=res, seed=SEED) for res in tqdm(resolutions_mod, desc="Louvain Modularity")])
    coms_leiden_mod  = list([leiden_fn(G, leidenalg.RBConfigurationVertexPartition, resolution_parameter=res, seed=SEED) for res in tqdm(resolutions_mod, desc="Leiden Modularity")])


    # Run louvain and leiden algorithms for all resolutions with the CPM quality function
    coms_louvain_cpm = list([louvain_fn(G, louvain.CPMVertexPartition, resolution_parameter=res, seed=SEED) for res in tqdm(resolutions_cpm, desc="Louvain CPM")])
    coms_leiden_cpm  = list([leiden_fn(G, leidenalg.CPMVertexPartition, resolution_parameter=res, seed=SEED) for res in tqdm(resolutions_cpm, desc="Leiden CPM")])


    mod_louvain_mod = [com.modularity for com in coms_louvain_mod]
    mod_leiden_mod = [com.modularity for com in coms_leiden_mod]
    mod_louvain_cpm = [com.modularity for com in coms_louvain_cpm]
    mod_leiden_cpm = [com.modularity for com in coms_leiden_cpm]


    from numpy import argmax
    idx_louvain_mod = argmax(mod_louvain_mod)
    idx_leiden_mod  = argmax(mod_leiden_mod)
    idx_louvain_cpm = argmax(mod_louvain_cpm)
    idx_leiden_cpm  = argmax(mod_leiden_cpm)



    # Set up two graphs, side by side, to plot the results in
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 4))
    ax2.yaxis.set_tick_params(labelbottom=True)

    ## Left graph, for modularity
    # ax1.set_title("Modularity")
    ax1.set_xlabel('Resolution $γ$ of quality function')
    ax1.set_ylabel('Quality of resulting partition')

    ## Right graph, for CPM
    # ax2.set_title("CPM")
    ax2.set_xlabel('Resolution $γ$ of quality function')
    ax2.set_ylabel('Quality of resulting partition')

    # Draw the results for the modularity quality function
    ## Plot the modularity determined
    ln_mod_louvain_mod, = ax1.plot(resolutions_mod, mod_louvain_mod, label='Quality Louvain')
    ln_mod_leiden_mod,  = ax1.plot(resolutions_mod, mod_leiden_mod,  label='Quality Leiden')

    ## Draw the x marks where maximum quality is attained
    ax1.set_prop_cycle(None)
    ax1.plot(resolutions_mod[idx_louvain_mod], mod_louvain_mod[idx_louvain_mod], 'x')
    ax1.plot(resolutions_mod[idx_leiden_mod],  mod_leiden_mod[idx_leiden_mod],   'x')

    # Draw the results for the CPM quality function
    ## Plot the modularity determined
    ln_mod_louvain_cpm, = ax2.plot(resolutions_cpm, mod_louvain_cpm, label='Louvain CPM')
    ln_mod_leiden_cpm,  = ax2.plot(resolutions_cpm, mod_leiden_cpm,  label='Leiden CPM')

    ## Draw the x marks where maximum quality is attained
    ax2.set_prop_cycle(None)
    ax2.plot(resolutions_cpm[idx_louvain_cpm], mod_louvain_cpm[idx_louvain_cpm], 'x')
    ax2.plot(resolutions_cpm[idx_leiden_cpm],  mod_leiden_cpm[idx_leiden_cpm],   'x')


    ## Left graph, for modularity
    # ax3.set_title("Modularity")
    ax3.set_xlabel('Resolution $γ$ of quality function')
    ax3.set_ylabel('Number of detected communities')

    ## Plot the number of found communities
    ln_cnt_louvain_mod, = ax3.plot(resolutions_mod, list(map(len, coms_louvain_mod)), label='Community count Louvain')
    ln_cnt_leiden_mod,  = ax3.plot(resolutions_mod, list(map(len, coms_leiden_mod)), label='Community count Leiden')



    ## Right graph, for CPM
    # ax4.set_title("CPM")
    ax4.set_xlabel('Resolution $γ$ of quality function')
    ax4.set_ylabel('Number of detected communities')

    ## Plot the number of found communities
    ln_cnt_louvain_cpm, = ax4.plot(resolutions_cpm, list(map(len, coms_louvain_cpm)), label='Community count')
    ln_cnt_leiden_cpm,  = ax4.plot(resolutions_cpm, list(map(len, coms_leiden_cpm)), label='Community count')

    # Put a legend there
    # lgd = fig.legend(handles=[ln_mod_louvain_mod, ln_mod_leiden_mod], ncol=4,
    #                  loc='upper center', bbox_to_anchor=(0.5, 0.0))

    # Draw the plot
    fig.tight_layout()
    # fig.subplots_adjust(wspace=0.4)
    # plt.show()
    fig.savefig(f"figures/{graph_name}.png")

if __name__ == "__main__":
    for graph, graph_name in zip(graphs, graph_names):
        try:
            print(f"Running {graph_name}")
            run(graph, graph_name)
        except Exception as e:
            print(f"Failed to run {graph_name}")
            print(e)
            continue
import pandas as pd
import numpy as np
import h5py
import os
import logging
from pathlib import Path
import torch

def get_edges_and_unconnected_nodes(node_coords):
    e_start=[]
    e_to=[]
    unconnected_nodes=[]
    for idx in range(len(node_coords)):
        i, j = node_coords[idx]
        all_nodes = np.where((node_coords[:,0]>=(i-1)) & (node_coords[:,0]<=(i+1)) & (node_coords[:,1]>=(j-1)) & (node_coords[:,1]<=(j+1)))
        adj_nodes=all_nodes[0][all_nodes[0]!=idx]
        if len(adj_nodes)>0:
            start_adj=np.ones(len(adj_nodes))*idx
            end_adj=adj_nodes
            e_start=np.append(e_start, start_adj)
            e_to=np.append(e_to, end_adj)
        else:
            #print(idx)
            unconnected_nodes=np.append(unconnected_nodes, idx) 
    e_start=e_start.astype(int)
    e_to=e_to.astype(int)
    if len(unconnected_nodes)>0:
        unconnected_nodes=unconnected_nodes.astype(int)
    e_idx=[e_start,e_to]
    return e_idx, unconnected_nodes

def main(city, in_dir, out_dir, file_type, tstval=None, vol_filt=0):
    """ Calcualte the 
    """
    logger = logging.getLogger(__name__)

    logger.info(f'Loading {city}{file_type}.npy')
    mvol_all = np.load(os.path.join(in_dir, f'{city}{file_type}.npy'))
    all_map= mvol_all>vol_filt

    graph_coverage = all_map.sum()/(495*436)
    logger.info(f'Ratio of Image Covered by Roads: {graph_coverage}')

    all_node_coords = np.array(np.where(all_map)).transpose()
    #all_node_coords = all_node_coords[all_node_coords[:,0]<=255]
    e_idx, unc_nodes = get_edges_and_unconnected_nodes(all_node_coords)
    logger.info(f'Number of Nodes: {len(all_node_coords)}')
    logger.info(f'Number of Edges: {len(e_idx[0])}')
    logger.info(f'Number of Unconnected Nodes: {len(unc_nodes)}')

    logger.info(f'Rerunning with only connected nodes')
    logger.info(f'Must be faster ways of doing this.....')
    connected_nodes = all_node_coords[np.unique(e_idx[0])]
    e_idx, unc_nodes = get_edges_and_unconnected_nodes(connected_nodes)
    logger.info(f'Number of Nodes: {len(all_node_coords)}')
    logger.info(f'Number of Edges: {len(e_idx[0])}')
    logger.info(f'Number of Unconnected Nodes: {len(unc_nodes)}')

    logger.info(f'Saving...')
    if tstval is None:
        node_file = os.path.join(out_dir, f'{city}_nodes_{vol_filt}.npy')
        edge_file = os.path.join(out_dir, f'{city}_edges_{vol_filt}.npy')
        unc_node_file = os.path.join(out_dir, f'{city}_unc_nodes_{vol_filt}.npy')
        mask_file = os.path.join(out_dir, f'{city}_Mask_{vol_filt}.pt')
    else:
        node_file = os.path.join(out_dir, f'{city}_nodes_{vol_filt}_{tstval}.npy')
        edge_file = os.path.join(out_dir, f'{city}_edges_{vol_filt}_{tstval}.npy')
        unc_node_file = os.path.join(out_dir, f'{city}_unc_nodes_{vol_filt}_{tstval}.npy') 
        mask_file = os.path.join(out_dir, f'{city}_Mask_{vol_filt}.pt')     
    np.save(node_file, connected_nodes)
    np.save(edge_file, e_idx)
    np.save(unc_node_file, unc_nodes)
    mask = torch.zeros([495,436]).byte()
    mask[connected_nodes[:,0], connected_nodes[:,1]]=1
    torch.save(mask, mask_file)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[1]
    data_dir = os.path.join(project_dir, 'data')
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    #main('Istanbul', os.path.join(data_dir, 'interim'), os.path.join(data_dir, 'processed'), '_roads_max_vol')
    city='Berlin'
    main(city, os.path.join(data_dir, 'interim'), os.path.join(data_dir, 'processed/'+city), '_roads_max_vol', None, 5)

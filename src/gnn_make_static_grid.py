import pandas as pd
import numpy as np
import h5py
import os
import logging
from pathlib import Path
from numba import jit

#@jit() # Set "nopython" mode for best performance, equivalent to @njit
def get_road_network(dir, image_size, testing, data_type):
    grid = np.zeros(image_size)
    i=0
    tot = len(os.listdir(dir))
    for f in os.listdir(dir):
        print(f'Processed {i}/{tot}')
        #if i%10 ==0:
        #    print(i)
        #    #print(f'Processed {i}/{tot}')
        fr = h5py.File(os.path.join(dir, f), 'r')
        data = fr[list(fr.keys())[0]]
        if testing:
            if data_type=='avg_tot_vol':
                data=np.mean(data, axis=0)
            else:
                data=np.max(data, axis=0)
        elif data_type=='max_vol':
            d_max = np.array(data)[:,:,:,[0,2,4,6]].max(3).max(0)
            grid = np.maximum(grid, d_max)
            #for j in [0,2,4,6]:
            #    d_max = np.max(data[:,:,:,j], axis=0)
            #    grid = np.maximum(grid, d_max)
        elif data_type=='avg_tot_vol':
            d_max = np.array(data)[:,:,:,[0,2,4,6]].mean(3).mean(0)
            grid+=d_max/tot
        else:
            #d_max = np.max(np.max(data, axis=0), axis=2)
            d_max = np.array(data).reshape(-1,495,436).sum(0)
            grid+=d_max
            #grid = np.maximum(grid, d_max)
        print(f'This slice size {(d_max>0.1).sum()/(495*436)} of image')
        print(f'Current size {(grid>0.1).sum()/(495*436)} of image')
        i=i+1
        #print(fr)
    #if data_type!='max_vol':
    #    grid = (grid>0).astype('uint8')
    return grid


def process_grid(city, image_size, trn_val_test, raw_dir, save, proc_dir, logger, data_type):
    fname_full = os.path.join(proc_dir, f'{city}_{trn_val_test}_roads_{data_type}.npy')
    raw=os.path.join(raw_dir, city, trn_val_test)
    #grid = get_road_network(raw, image_size, trn_val_test=='testing', data_type)
    if os.path.isfile(fname_full):
        logger.info(f'Reading already processed {trn_val_test}_{data_type} file')
        grid = np.load(fname_full)
    else:
        raw=os.path.join(raw_dir, city, trn_val_test)
        grid = get_road_network(raw, image_size, trn_val_test=='testing', data_type)
        if save:
            np.save(fname_full, grid)
    return grid

def combined_grid(train_grid, val_grid, test_grid, save, city, proc_dir, data_type):
    grid = np.maximum(train_grid, val_grid)
    grid = np.maximum(grid, test_grid)
    if save:
        fname = f'{city}_roads_{data_type}.npy'
        np.save(os.path.join(proc_dir, fname), grid)
    return grid

def main(city, image_size, raw_dir, proc_dir, data_type):
    """ For the Graph-NN approach static grid from all the train,test,val data showing where roads are
    """
    logger = logging.getLogger(__name__)

    logger.info('Calculating & saving the road network for training data')
    trn_grid = process_grid(city, image_size, 'training', raw_dir, True, proc_dir, logger,data_type)
    road_pc = (trn_grid!=0).sum()/(image_size[0]*image_size[1])
    logger.info(f'Training images shows a road network covers {road_pc} of image')

    logger.info('Calculating & saving  the road network for validation data')
    val_grid = process_grid(city, image_size, 'validation', raw_dir, True, proc_dir, logger, data_type)
    road_pc = (val_grid!=0).sum()/(image_size[0]*image_size[1])
    logger.info(f'Val images shows a road network covers {road_pc} of image')

    logger.info('Calculating & saving the road network for testing data')
    tst_grid = process_grid(city, image_size, 'testing', raw_dir, True, proc_dir, logger, data_type)
    road_pc = (tst_grid!=0).sum()/(image_size[0]*image_size[1])
    logger.info(f'test images shows a road network covers {road_pc} of image')

    logger.info('Combining the grid to calculate overall')
    #comb_grid = np.maximum(tst_grid, val_grid)
    comb_grid = combined_grid(trn_grid, val_grid, tst_grid, True, city, proc_dir, data_type)
    road_pc = (comb_grid!=0).sum()/(image_size[0]*image_size[1])
    logger.info(f'Combined images shows a road network covers {road_pc} of image')
    assert (np.subtract((tst_grid!=0)*1, (comb_grid!=0)*1)==1).sum()==0, "Oh no! Seems like there is activity in image areas in the test that isnt in the train or val"
    return comb_grid
    return comb_grid

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[1]
    data_dir = os.path.join(project_dir, 'data')
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main('Berlin', [495, 436], os.path.join(data_dir, 'raw'), os.path.join(data_dir, 'interim'), 'max_vol')

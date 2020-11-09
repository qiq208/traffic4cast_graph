from torch_geometric.data import Dataset, Data, DataLoader
from pathlib import Path
import torch
import numpy as np
import h5py
import os
import math
from scipy import stats

class CityGraphDataset(Dataset):
    def __init__(self,raw_dir, base_dir, city, forward_mins, window=12, mode='training', overlap=True, normalise=None, full_val=False, pca_static=False):
        self.window = window
        self.forward_mins = forward_mins
        self.mode = mode
        self.full_val=full_val
        self.raw_data_dir = Path(raw_dir + '/' +city)
        if base_dir is None:
            self.data_dir = Path(city)
        else:
            self.data_dir = Path(base_dir + '/' +city)
        self.data_file = self.data_dir / (city+'_'+mode+'_5.h5')
        self.file_list = list(h5py.File(self.data_file, 'r').keys())
        self.overlap=overlap
        self.forward_steps = forward_mins//5
        self.single_forward = (len(self.forward_steps)==1)
        self.normalise=normalise
        self.pca_static = pca_static
        print(f'Normalising by: {normalise}')

        if (normalise=='Active'):
            with open(self.data_dir / (city + '_norm_'+ normalise + '.npy'), 'rb') as f:           
              self.mean = np.load(f)
              self.std = np.load(f)
            with open(self.data_dir / (city + '_norm_'+ normalise + '_static.npy'), 'rb') as f: 
              self.mean_static = np.load(f)
              self.std_static = np.load(f)  

        if mode=='validation':
            self.idxs = np.load(self.data_dir /'ivan_val.npy')
            self.len=len(self.idxs)
        elif mode=='training':
            self.idxs = np.load(self.data_dir /'ivan_train.npy')
            self.len=len(self.idxs)
        else:
            #forward_steps = [int(i/5) for i in forward_mins]
            #self.total_window = window + max(forward_steps)
            self.total_window = window + self.forward_steps.max()
            if self.overlap:
              self.data_slice_per_file = 288 - self.total_window
              self.len = len(self.file_list)*self.data_slice_per_file
            else:
              self.data_slice_per_file = math.floor(288/self.total_window)
              self.len = len(self.file_list)*self.data_slice_per_file            
        
        node_coords_file = self.data_dir / (city +'_nodes_5.npy')
        edge_file = self.data_dir / (city +'_edges_5.npy')
        self.node_coords = torch.tensor(np.load(node_coords_file), dtype=torch.long)
        self.edges = torch.tensor(np.load(edge_file), dtype=torch.long)
        
        if self.pca_static:
            self.city_static = np.load(self.data_dir / (city + '_static_pca.npy'))
            self.city_static=self.city_static[:,:4]
        else:
            static_file = self.raw_data_dir / (city + '_static_2019.h5')
            static = h5py.File(static_file, 'r')
            self.city_static = static[list(static.keys())[0]]
            self.city_static = np.array(self.city_static)[self.node_coords[:,0], self.node_coords[:,1], :]
        self.len_total = self.len
        self.scale=1
        super().__init__(self.data_dir)

    def __len__(self):
        return self.len

    def set_subset_len(self, subset_len):
        self.len=subset_len
        self.scale = math.floor(self.len_total/self.len)

    def get(self, idx, debug=False):
        idx = self.scale*idx
        fileId =self.idxs[idx, 0]
        dayId = self.idxs[idx, 1]+12

        if debug:
            print(f'index: {idx}')
            print(f'File_ID: {fileId}')
            print(f'time_id: {dayId}')

        label_idx = dayId-1+self.forward_steps
        with h5py.File(self.data_file, 'r') as fr:
            train_data = fr[self.file_list[fileId]][(dayId-self.window):dayId]
            label_data=fr[self.file_list[fileId]][label_idx.min():(label_idx.max()+1)]
            label_data=label_data[:, :,:8]
        fr.close()
        

        x = self.get_training_data(train_data, self.city_static, self.window, slice_id=dayId)
        #x=x/255
        y = self.get_label_data(label_data, dayId, self.forward_steps)
        #y=y/255
        data = Data(x=x, y=y, edge_index = self.edges)

        if (self.mode=='validation') & (self.full_val):
            val_file = self.raw_data_dir / ('validation')/self.file_list[fileId]
            with h5py.File(val_file, 'r') as fr:
                y_image = fr[list(fr.keys())[0]][label_idx.min():(label_idx.max()+1)]
            y_image = y_image[:, :, :,:8]
            fr.close()
            y_image = torch.tensor(y_image, dtype=torch.float)
            y_zeros = torch.zeros(y_image.shape,dtype=torch.float)
            return data, y_image, y_zeros, torch.tensor(fileId),torch.tensor(dayId)
        return data
    
    def get_fileId_sliceNo_dayId(self, idx, overlap, debug=False):
        fileId, sliceNo = divmod(idx+1+self.data_slice_per_file,self.data_slice_per_file)
        fileId=fileId-1
        if debug:
            print(f'orig file id: {fileId}')
            print(f'orig slice no: {sliceNo}')
        if sliceNo==0: 
            fileId=fileId-1
            sliceNo=self.data_slice_per_file
            if debug:
                print(f'mod file id: {fileId}')
                print(f'mod slice no: {sliceNo}')
        if overlap:
          dayId=sliceNo+self.window-1
        else:
          dayId= self.window + (sliceNo-1)*self.total_window
        return fileId, sliceNo, dayId
    
    def add_static_data(self, slice_data, static_data):
        return np.concatenate([slice_data, static_data], axis=1)

    def process_slice_train(self, sliced_window):
        data=sliced_window[6:]
        dmean = np.expand_dims(sliced_window[:6].mean(axis=0), axis=0)
        dmin = np.expand_dims(sliced_window[:6].min(axis=0), axis=0)
        dmax = np.expand_dims(sliced_window[:6].max(axis=0), axis=0)
        data=np.concatenate([data, dmean, dmin, dmax], axis=0)
        self.channels = data.shape[-1]
        #print(self.channels)
        sliced_window=data
        s = np.moveaxis(sliced_window, 0, -1)
        s = s.reshape(len(self.node_coords),-1)
        return s
    
    def process_slice(self, sliced_window):
        s = np.moveaxis(sliced_window, 0, -1)
        s = s.reshape(len(self.node_coords),-1)
        return s

    def get_training_data(self, full_data, static_data, window=12, slice_id=None):
        #slice_window = full_data[(day_id-window):day_id]
        slice_window=full_data
        no_timesteps = slice_window.shape[0]
        assert (no_timesteps==window), f'Expected data to be for {window} timesteps, but got {no_timesteps} timesteps. day_id probably<window'
        
        slice_data=self.process_slice_train(slice_window)
        alive_nodes=(slice_data.sum(1)>0)
        
        channels=self.channels
        slice_data = self.add_static_data(slice_data, static_data)

        if (self.normalise=='Active'):
          mean = np.concatenate((self.mean.repeat(channels), self.mean_static))
          std = np.concatenate((self.std.repeat(channels), self.std_static))
          slice_data=(slice_data-mean)/std
            
        slice_id_feat = slice_id*np.ones(slice_data.shape[0]).T
        slice_id_feat=(slice_id_feat-144)/82.849
        coord_feat = (self.node_coords-np.array([247,217.5]))/np.array([142.8939,125.862])
        slice_data = np.column_stack([slice_data, slice_id_feat, coord_feat, alive_nodes])
        slice_data = torch.tensor(slice_data, dtype=torch.float)
        return slice_data

    def get_label_data(self, slice_window_label, day_id, forward_steps):
        slice_data = self.process_slice(slice_window_label)
        slice_data = torch.tensor(slice_data, dtype=torch.float)
        return slice_data

    def convert_graph_minibatch_y_to_image(self, graph_y, image_zeros):
        #graph_y=graph_y.reshape(-1,len(self.node_coords), 8, len(self.forward_steps))
        graph_y=graph_y.reshape(-1,len(self.node_coords), 8, 12)
        graph_y=graph_y.permute(0,3,1,2)
        image_zeros[:,:,self.node_coords[:,0], self.node_coords[:,1], :]=graph_y
        return image_zeros

    def _download(self):
        pass

    def _process(self):
        pass
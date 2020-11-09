from torch_geometric.data import Dataset, Data
from pathlib import Path
import torch
import numpy as np
import h5py
import os
import math
from scipy import stats
#'data/processed/'

class CityGraphDataset(Dataset):
    def __init__(self, city, forward_mins, window=12, mode='training', overlap=True, normalise=None, full_val=False, pca_static=False):
        self.window = window
        self.forward_mins = forward_mins
        self.mode = mode
        self.full_val=full_val
        self.data_dir = Path('data/processed/'+city)
        self.data_file = self.data_dir / (city+'_'+mode+'.h5')
        self.file_list = list(h5py.File(self.data_file, 'r').keys())
        self.overlap=overlap
        self.forward_steps = forward_mins//5
        self.single_forward = (len(self.forward_steps)==1)
        self.normalise=normalise
        self.pca_static = pca_static
        print(f'Normalising by: {normalise}')

        if (normalise=='Active') or (normalise=='noZeros'):
            with open(self.data_dir / (city + '_norm_'+ normalise + '.npy'), 'rb') as f:           
              self.mean = np.load(f)
              self.std = np.load(f)
            with open(self.data_dir / (city + '_norm_'+ normalise + '_static.npy'), 'rb') as f: 
              self.mean_static = np.load(f)
              self.std_static = np.load(f)  
        if (normalise=='lmdas'):
            with open(self.data_dir / (city + '_norm_'+ normalise + '.npy'), 'rb') as f:           
              self.lmdas = np.load(f)   
            with open(self.data_dir / (city + '_norm_'+ normalise + '_static.npy'), 'rb') as f: 
              self.lmdas_static = np.load(f)

        if mode=='testing':
            self.len = len(self.file_list)
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
        
        node_coords_file = self.data_dir / (city +'_nodes.npy')
        edge_file = self.data_dir / (city +'_edges.npy')
        self.node_coords = torch.tensor(np.load(node_coords_file), dtype=torch.long)
        self.edges = torch.tensor(np.load(edge_file), dtype=torch.long)
        
        if self.pca_static:
            self.city_static = np.load(self.data_dir / (city + '_static_pca.npy'))
            self.city_static=self.city_static[:,:4]
        else:
            static_file = self.data_dir / (city + '_static_2019.h5')
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
        #print(idx)

        #start = time.time()
        fileId, sliceNo, dayId = self.get_fileId_sliceNo_dayId(idx, self.overlap)
        #fr = h5py.File(self.data_dir.parent /self.mode/ self.file_list[fileId], 'r')
        #full_data = fr[list(fr.keys())[0]]
        #print(full_data)
        #print(time.time()-start)
        #forward_steps = [int(i/5) for i in self.forward_mins]
        if debug:
            print(idx)
            print(fileId)
            print(sliceNo)
            print(dayId)
            print(forward_steps)
        
        print(f'Qi File: {self.file_list[fileId]}')
        print(f'Qi frame: {dayId-self.window}')
        #label_idx = [(dayId-1+i) for i in forward_steps]
        label_idx = dayId-1+self.forward_steps
        #print(time.time()-start)
        with h5py.File(self.data_file, 'r') as fr:
          #t_d = range((dayId-self.window):dayId)
          #data = fr[self.file_list[fileId]][list(t_d)+label_idx]
          train_data = fr[self.file_list[fileId]][(dayId-self.window):dayId]
          #print(f'train slice start: {dayId-self.window}')
          #print(f'train slice end: {dayId}')
          #label_data = fr[self.file_list[fileId]][label_idx]
          #a = fr[self.file_list[fileId]][:]
          #print(label_idx)
          if self.single_forward:
            #print(f'label slice : {label_idx.max()}')
            label_data=fr[self.file_list[fileId]][label_idx.max()]
            #print(label_data.shape)
            label_data = np.expand_dims(label_data, axis=0)
            #print(label_data.shape)
          else:
            label_data=fr[self.file_list[fileId]][label_idx.min():(label_idx.max()+1)]
            #print(label_data.shape)
            #label_data=label_data[label_idx-label_idx.min()]
            label_data=label_data[self.forward_steps-1, :,:8]
            #print(label_data.shape)
            #label_data1 = fr[self.file_list[fileId]][label_idx]
            #print(label_data1.shape)
            #print((label_data-label_data1).max())
        fr.close()
        
        #train_data=data[:len(t_d)]
        #label_data=data[len(t_d):]
        #print(time.time()-start)
        #with h5py.File(self.data_dir.parent /self.mode/ self.file_list[fileId], 'r') as fr:
        #    train_data = fr[list(fr.keys())[0]][(dayId-self.window):dayId]
        #    label_data = fr[list(fr.keys())[0]][label_idx]
        x = self.get_training_data(train_data, self.city_static, dayId, self.window, slice_id=None)
        #x=x/255
        y = self.get_label_data(label_data, dayId, self.forward_steps)
        #y=y/255
        data = Data(x=x, y=y, edge_index = self.edges)
        #data = Data(x=x, y=y, edge_index = self.edges, edge_attr = torch.ones(self.edges.shape[1]).unsqueeze(1))
        #data = Data(x=x, y=y, edge_index = self.edges, x_coords=self.node_coords)
        #print(time.time()-start)
        if (self.mode=='validation') & (self.full_val):
          val_file = self.data_dir / ('validation')/self.file_list[fileId]
          #print(val_file)
          with h5py.File(val_file, 'r') as fr:
            y_image = fr[list(fr.keys())[0]][label_idx.min():(label_idx.max()+1)]
            #print(label_data.shape)
            y_image = y_image[self.forward_steps-1, :, :,:8]
          fr.close()
          y_image = torch.tensor(y_image, dtype=torch.float)
          y_zeros = torch.zeros(y_image.shape,dtype=torch.float)
          return data, y_image, y_zeros
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
        #data=sliced_window[9:]
        #dmean = np.expand_dims(sliced_window[:9].mean(axis=0), axis=0)
        #dmin = np.expand_dims(sliced_window[:9].min(axis=0), axis=0)
        #dmax = np.expand_dims(sliced_window[:9].max(axis=0), axis=0)
        #data=np.concatenate([data, dmean, dmin, dmax], axis=0)
        #sliced_window=data
        s = np.moveaxis(sliced_window, 0, -1)
        s = s.reshape(len(self.node_coords),-1)
        return s
    
    def process_slice(self, sliced_window):
        s = np.moveaxis(sliced_window, 0, -1)
        s = s.reshape(len(self.node_coords),-1)
        return s

    def get_training_data(self, full_data, static_data, day_id, window=12, slice_id=None):
        #slice_window = full_data[(day_id-window):day_id]
        slice_window=full_data
        no_timesteps = slice_window.shape[0]
        assert (no_timesteps==window), f'Expected data to be for {window} timesteps, but got {no_timesteps} timesteps. day_id probably<window'
        
        slice_data=self.process_slice_train(slice_window)

        channels = 6
        if self.pca_static:
          if (self.normalise=='Active') or (self.normalise=='noZeros'):
            mean = self.mean.repeat(channels)
            std = self.std.repeat(channels)
            slice_data=(slice_data-mean)/std

          if (self.normalise=='lmdas'):
            lmbdas = self.lmdas.repeat(self.window)
            for i in range(len(lmbdas)):
              slice_data[:,i]=stats.yeojohnson(slice_data[:,i], lmbda=lmbdas[i])
        
        #slice_data =slice_data[:,54:]
        slice_data = self.add_static_data(slice_data, static_data)

        if not self.pca_static:
          if (self.normalise=='Active') or (self.normalise=='noZeros'):
            mean = np.concatenate((self.mean.repeat(channels), self.mean_static))
            std = np.concatenate((self.std.repeat(channels), self.std_static))
            slice_data=(slice_data-mean)/std

          if (self.normalise=='lmdas'):
            lmbdas = np.concatenate((self.lmdas.repeat(self.window), self.lmdas_static))
            for i in range(len(lmbdas)):
              slice_data[:,i]=stats.yeojohnson(slice_data[:,i], lmbda=lmbdas[i])

        if slice_id is not None:
          #print(slice_id)
          slice_id_feat = slice_id*np.ones(slice_data.shape[0]).T
          if self.normalise is not None:
            slice_id_feat=(slice_id_feat-144)/82.849
          slice_data = np.column_stack([slice_data, slice_id_feat])
        
        slice_data = torch.tensor(slice_data, dtype=torch.float)
        return slice_data

    def get_label_data(self, slice_window_label, day_id, forward_steps):
        #label_idx = [(day_id-1+i) for i in forward_steps]
        #slice_window_label = full_data[label_idx]
        #slice_window_label = full_data[:,:,:8]
        #print(slice_window_label.shape)
        slice_data = self.process_slice(slice_window_label)
        slice_data = torch.tensor(slice_data, dtype=torch.float)
        return slice_data

    def convert_graph_minibatch_y_to_image(self, graph_y, image_zeros):
        graph_y=graph_y.reshape(-1,len(self.node_coords), 8, len(self.forward_steps))
        graph_y=graph_y.permute(0,3,1,2)
        image_zeros[:,:,self.node_coords[:,0], self.node_coords[:,1], :]=graph_y
        return image_zeros

    def _download(self):
        pass

    def _process(self):
        pass
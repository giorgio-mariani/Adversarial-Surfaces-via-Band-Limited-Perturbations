import os

import numpy as np
import torch
import torch_geometric.data
import torch_geometric.io as gio
import torch_geometric.transforms as transforms
import tqdm

import mesh.decimation
import dataset.downscale

class ComaDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, root:str):
        super(ComaDataset, self).__init__(root=root)
        data_list = torch.load(self.processed_paths[0])
        self.cat2id = IndexDict()
        self.sub2id = IndexDict()

        for mesh in data_list:
            mesh.y = self.cat2id[mesh.category]
            mesh.subj = self.sub2id[mesh.subject]
            mesh.category = None #necessary in order to collate the object
            mesh.subject = None #necessary in order to collate the object
        self.data, self.slices = self.collate(data_list=data_list)

        # create downscale matrices
        self.downscale = dataset.downscale.DownscaleDelegate(self,ds_model_index=0)

    @property
    def raw_file_names(self):
        # for each folder
        subtree = os.walk(top=self.root, topdown=True)
        raw_filenames = []
        for path, dirs, fnames in subtree:
            if len(dirs) == 0:
                for fname in fnames:
                    filename = os.path.join(path, fname)
                    relfname = os.path.relpath(filename, self.raw_dir)
                    raw_filenames.append(relfname)
        return raw_filenames

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        
        # TODO


    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        raise RuntimeError('Dataset not found.')
    
    # delegate methods
    @property
    def downscale_matrices(self): return self.downscale.downscale_matrices

    @property
    def downscaled_edges(self): return self.downscale.downscaled_edges

    @property
    def downscaled_faces(self): return self.downscale.downscaled_faces

class IndexDict(dict):
    def __init__(self, iterable=[]):
        super().__init__(iterable)
        self._i = 0

    def __getitem__(self, key):
        if key not in self:
            self[key] = self._i
            self._i += 1
        return super().__getitem__(key)


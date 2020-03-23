import os.path as osp
from glob import glob
import random

import torch
from torch_geometric.data import InMemoryDataset, extract_zip
from torch_geometric.datasets import CoMA
from torch_geometric.io import read_ply

import dataset.downscale

class CoMADataset(CoMA):
    def __init__(self, root, train=True, category_samples=50, test_categories=[0,1]):
        self.test_categories = test_categories
        self.category_samples = category_samples
        super(CoMA, self).__init__(root, None, None, None)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)
        self.data.y = self.data.subject
        self.slices["y"] = self.slices["subject"]
        
        # create downscale matrices
        self.downscale = dataset.downscale.DownscaleDelegate(self, ds_model_index=0)

    def process(self):
        folders = sorted(glob(osp.join(self.raw_dir, 'FaceTalk_*')))
        if len(folders) == 0:
            extract_zip(self.raw_paths[0], self.raw_dir, log=False)
            folders = sorted(glob(osp.join(self.raw_dir, 'FaceTalk_*')))

        train_data_list, test_data_list = [], []
        for fi, folder in enumerate(folders):
            for ci, category in enumerate(self.categories):
                files = sorted(glob(osp.join(folder, category, '*.ply')))
                files = random.sample(files, k=min(self.category_samples, len(files)))
                for j, f in enumerate(files):
                    data = read_ply(f)
                    data.category = torch.tensor([ci], dtype=torch.long)
                    data.subject = torch.tensor([fi], dtype=torch.long)
                    if self.pre_filter is not None and\
                       not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    if not ci in self.test_categories:
                        train_data_list.append(data)
                    else:
                        test_data_list.append(data)

        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(test_data_list), self.processed_paths[1])

'''
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
'''



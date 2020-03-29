import os

import numpy as np
import torch
import torch_geometric
import torch_geometric.data
import torch_geometric.io as gio
import torch_geometric.transforms as transforms
import tqdm

import mesh.decimation
import dataset.downscale as dscale
from dataset.transforms import Move, Rotate

class FaustDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, 
        root:str, 
        device:torch.device=torch.device("cpu"),
        train:bool=True, test:bool=True,
        transform_data:bool=True):
        self.url = 'http://faust.is.tue.mpg.de/'
        def to_device(mesh:torch_geometric.data.Data):
            mesh.pos = mesh.pos.to(device)
            mesh.y = mesh.y.to(device)
            return mesh

        if transform_data:
            # rotate and move
            transform = transforms.Compose([
                Move(mean=[0,0,0], std=[0.05,0.05,0.05]), 
                Rotate(dims=[0,1,2]), 
                to_device])

            # center each mesh into its centroid
            pre_transform = Move(mean=[0,0,0], std=[0.0,0.0,0.0])
            super().__init__(root=root, transform=transform, pre_transform=pre_transform)
        else:
            super().__init__(root=root, transform=to_device)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.ds_delegate = dscale.DownscaleDelegate(self)

        if train and not test:
            self.data, self.slices = self.collate([self.get(i) for i in range(20, 100)])
        elif not train and test:
            self.data, self.slices = self.collate([self.get(i) for i in range(0, 20)])

    @property
    def raw_file_names(self):
        tofilename =  lambda x : "tr_reg_"+str(x).zfill(3)+".ply"
        return [tofilename(fi) for fi in range(100)]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download {} from {} and move it to {}'.format(self.raw_file_names, self.url, self.raw_dir))
    
    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        f2e = transforms.FaceToEdge(remove_faces=False)
        for i, path in enumerate(tqdm.tqdm(self.raw_paths)):
            mesh = torch_geometric.io.read_ply(path)
            mesh.y = i%10 # set the mesh class (note that FAUST models are ordered by class)
            f2e(mesh)
            data_list.append(mesh)
        data, slices = self.collate(data_list)
        if not os.path.exists(self.processed_dir):
            os.mkdir(path)
        torch.save( (data, slices), self.processed_paths[0])

    @property
    def downscale_matrices(self): return self.ds_delegate.downscale_matrices

    @property
    def downscaled_edges(self): return self.ds_delegate.downscaled_edges

    @property
    def downscaled_faces(self): return self.ds_delegate.downscaled_faces


class FaustAugmented(FaustDataset):
  def __init__(self,
    root:str,
    device:torch.device = torch.device("cpu"),
    train:bool=True, test:bool=True,
    transform_data:bool=True):
    super().__init__(root=root, 
        device=device, 
        transform_data=transform_data, 
        train=True, test=True)
    data_aug, slices_aug = torch.load(self.processed_paths[1])
    data, slices = self.data, self.slices
    
    keys = data.keys
    key_cat_dim = {"pos":0, "edge_index":1, "y":0,"face":1}
    for k in keys:
      data_k = getattr(data, k)
      data_k_aug = getattr(data_aug, k)
      slice_k = slices[k]
      slice_k_aug =  slices_aug[k]

      tmp_data = torch.cat([data_k, data_k_aug], dim=key_cat_dim[k])
      tmp_slices = torch.cat([slice_k[:-1], slice_k[-1] + slice_k_aug[:]], dim=0)

      self.data[k] = tmp_data
      self.slices[k] = tmp_slices

    if train and not test:
        I = list(range(20,80))+ list(range(120,200))+ list(range(212,260))
        self.data, self.slices = self.collate([self.get(i) for i in I])
    elif not train and test:
        I = list(range(0,20))+ list(range(100,120))+ list(range(200,212))
        self.data, self.slices = self.collate([self.get(i) for i in I])

  @property
  def processed_file_names(self):
      return ['data.pt','data_amass.pt']



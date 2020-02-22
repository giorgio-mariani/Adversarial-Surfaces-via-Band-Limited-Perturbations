import os

import numpy as np
import torch
import torch_geometric.data
import torch_geometric.io as gio
import torch_geometric.transforms as transforms
import tqdm

import mesh.decimation
import dataset.downscale as dscale

class FaustDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, root:str):
        super().__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.url = 'http://faust.is.tue.mpg.de/'
        self.ds_delegate = dscale.DownscaleDelegate(self)


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
        torch.save( (data, slices), self.processed_paths[0])

    
    @property
    def downscale_matrices(self): return self.ds_delegate.downscale_matrices

    @property
    def downscaled_edges(self): return self.ds_delegate.downscaled_edges

    @property
    def downscaled_faces(self): return self.ds_delegate.downscaled_faces


class FaustAugmented(FaustDataset):
  def __init__(self, root:str):
    super().__init__(root=root)
    data_aug, slices_aug = torch.load(self.processed_paths[1])
    data, slices = self.data, self.slices
    
    #num_subj = 10
    #num_class = 10
    #data.subj = torch.repeat_interleave(torch.arange(start=0, end=num_subj), num_class, dim=0)
    #slices["subj"] = torch.range(start=0, end=num_class*num_subj)

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

  @property
  def processed_file_names(self):
      return ['data.pt','data_amass.pt']

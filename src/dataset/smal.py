from os import listdir, mkdir
from os.path import isfile, join, exists, split
import tarfile

import tqdm
import torch
import torch_geometric.data
import torch_geometric.transforms as transforms

import dataset.downscale as dscale
from utils.transforms import Move, Rotate


class SmalDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, 
        root:str, 
        device:torch.device=torch.device("cpu"),
        train:bool=True, 
        test:bool=True,
        transform_data:bool=True):

        self.url = 'http://smal.is.tue.mpg.de'
        self.categories = ["big_cats","cows","dogs","hippos","horses"]

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
            super().__init__(root=root, transform=to_device, pre_transform=pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.ds_delegate = dscale.DownscaleDelegate(self)
        '''
        if train and not test:
            self.data, self.slices = self.collate([self.get(i) for i in range(20, 100)])
        elif not train and test:
            self.data, self.slices = self.collate([self.get(i) for i in range(0, 20)])'''

    @property
    def raw_file_names(self):
        categ_files = []
        for category in self.categories:
            category_dir = join(self.raw_dir, category)
            files = listdir(category_dir)
            categ_files += [join(category,f) for f in files if isfile(join(category_dir, f)) and f.split(".")[-1]=="ply" and f[0]!="."]
        return categ_files

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
            category = split(split(path)[0])[1]
            mesh.y = self.categories.index(category)
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

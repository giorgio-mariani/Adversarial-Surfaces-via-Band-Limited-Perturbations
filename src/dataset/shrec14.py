import os

import numpy as np
import torch
import torch_sparse as tsparse
import torch_geometric
import torch_geometric.data
import torch_geometric.io as gio
import torch_geometric.transforms as transforms
import tqdm

from utils import generate_transform_matrices
import dataset.downscale as dscale
from utils.transforms import Move, Rotate

class Shrec14Dataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, 
        root:str, 
        device:torch.device=torch.device("cpu"),
        train:bool=True, test:bool=True,
        transform_data:bool=True):
        self.url = 'http://www.cs.cf.ac.uk/shaperetrieval/shrec14/'
        
        def to_device(mesh:torch_geometric.data.Data):
            mesh.pos = mesh.pos.to(device)
            mesh.edge_index = mesh.edge_index.to(device)
            mesh.face = mesh.face.to(device)
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

        (self.downscale_matrices, 
        self.downscaled_edges, 
        self.downscaled_faces) = torch.load(self.processed_paths[1])


        testset_slice, trainset_slice = (0,80), (80, 400)
        if train and not test:
            self.data, self.slices = self.collate([self.get(i) for i in range(*trainset_slice)])
            self.downscale_matrices = self.downscale_matrices[trainset_slice[0]:trainset_slice[1]]
            self.downscaled_edges = self.downscaled_edges[trainset_slice[0]:trainset_slice[1]]
            self.downscaled_faces = self.downscaled_faces[trainset_slice[0]:trainset_slice[1]]

        elif not train and test:
            self.data, self.slices = self.collate([self.get(i) for i in range(*testset_slice)])

            self.downscale_matrices = self.downscale_matrices[testset_slice[0]:testset_slice[1]]
            self.downscaled_edges = self.downscaled_edges[testset_slice[0]:testset_slice[1]]
            self.downscaled_faces = self.downscaled_faces[testset_slice[0]:testset_slice[1]]


    @property
    def raw_file_names(self):
        tofilename =  lambda x : "Data/{}.obj".format(x)
        return [tofilename(fi) for fi in range(400)]

    @property
    def processed_file_names(self):
        return ['data.pt', 'downscale_data.pt']

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download it from {} and move it to {}'.format(self.url, self.raw_dir))
    
    def process(self):
        face2edge = transforms.FaceToEdge(remove_faces=False)
        datalist_file = os.path.join(self.processed_dir,"tmp.pt")

        # Read data into huge `Data` list.
        if os.path.exists(datalist_file):
            data_list = torch.load(datalist_file)
        else:
            data_list = []

        for i, path in enumerate(tqdm.tqdm(self.raw_paths)):
            if len(data_list) > i: 
                continue

            mesh = torch_geometric.io.read_obj(path)
            mesh.y = i%10 # set the mesh class (note that SHREC14 models are ordered by class)
            mesh.subject = int(i/10)
            face2edge(mesh)
            
            # add decimation matrices
            pos, faces = mesh.pos.numpy(), mesh.face.t().numpy()
            _,F,E,D = generate_transform_matrices(pos, faces, [4,4,4])
            mesh.downscale_matrices = [_scipy_to_torch_sparse(d) for d in D]
            mesh.downscaled_edges = [_scipy_to_torch_sparse(e) for e in E]
            mesh.downscaled_faces = [torch.tensor(f).t() for f in F]

            # add data to the save list
            data_list.append(mesh)
            
            # save the data if last element in 50
            if i%50 == 49:
                torch.save(data_list, datalist_file)

        downscale_matrices = []
        downscaled_edges = []
        downscaled_faces = []
        for mesh in data_list:
            downscale_matrices.append(mesh.downscale_matrices)
            downscaled_edges.append([i for (i,v, s) in mesh.downscaled_edges])
            downscaled_faces.append(mesh.downscaled_faces)

            delattr(mesh, "downscale_matrices")
            delattr(mesh, "downscaled_edges")
            delattr(mesh, "downscaled_faces")


        data, slices = self.collate(data_list)
        torch.save( (data, slices), self.processed_paths[0])
        torch.save((downscale_matrices,downscaled_edges,downscaled_faces), self.processed_paths[1])
        #os.remove(datalist_file)



def _scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape
    return (i,v, torch.Size(shape))

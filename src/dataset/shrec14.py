import os
from multiprocessing import Pool


import numpy as np
import torch
import torch_sparse as tsparse
import torch_geometric
import torch_geometric.data
import torch_geometric.io as gio
import torch_geometric.transforms as transforms
import tqdm

import utils
from utils import generate_transform_matrices
import dataset.downscale as dscale
from utils.transforms import Move, Rotate

from torch_geometric.data.dataloader import DataLoader

def remesh(Mvert,Mtriv,nvert):
    mesh = om.TriMesh(Mvert,Mtriv)
    mesh.request_vertex_normals()

    modQ = om.TriMeshModQuadricHandle()
    decimator = om.TriMeshDecimater(mesh)
    
    decimator.add(modQ)
    decimator.initialize()
    decimator.decimate_to(nvert)
    mesh.garbage_collection()
    
    vt = list(mesh.vertices())[0]
    
    return np.asarray(mesh.points()), np.asarray(mesh.face_vertex_indices())

class Shrec14Dataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, 
        root:str, 
        device:torch.device=torch.device("cpu"),
        train:bool=True, test:bool=True,
        transform_data:bool=True,
        transform=None,
        synth=False,
        spectral=False,
        nvert=0):
        self.url = 'http://www.cs.cf.ac.uk/shaperetrieval/shrec14/'
        
        self.spectral=spectral
        self.nvert = nvert
        def to_device(mesh:torch_geometric.data.Data):
            mesh.pos = mesh.pos.to(device)
            mesh.edge_index = mesh.edge_index.to(device)
            mesh.face = mesh.face.to(device)
            mesh.y = mesh.y.to(device)
            return mesh

        if transform_data and transform is None:
            # rotate and move
            transform = transforms.Compose([  
                transforms.Center(),
#                 transforms.RandomScale((0.8,1.2)),
                Rotate(dims=[1]), 
                Move(mean=[0,0,0], std=[0.05,0.05,0.05]), 
                transforms.RandomTranslate(0.01),
                to_device])

        # center each mesh into its centroid
        pre_transform = transforms.Center()
        super().__init__(root=root, transform=transform, pre_transform=pre_transform)
#         else:
#             super().__init__(root=root, transform=to_device)

        self.data, self.slices = torch.load(self.processed_paths[0])

        (self.downscale_matrices, 
        self.downscaled_edges, 
        self.downscaled_faces) = torch.load(self.processed_paths[1])

        datas = []
        for i in range(400):
            d = super().get(i)
            d.id=i//10
            datas.append(d)

        testset_slice, trainset_slice = list(range(0,40))+list(range(200,240)) , list(range(40,200))+list(range(240,400))
        if train and not test:
            self.data, self.slices = self.collate([datas[i] for i in trainset_slice])

        elif not train and test:
            self.data, self.slices = self.collate([datas[i] for i in testset_slice])


    @property
    def raw_file_names(self):
        tofilename =  lambda x : "Data/{}.obj".format(x)
        return [tofilename(fi) for fi in range(300)]

    @property
    def processed_file_names(self):
        return ['data.pt', 'downscale_data.pt']

    def download(self):
        print('Downloaded?')
#         raise RuntimeError(
#             'Dataset not found. Please download it from {} and move it to {}'.format(self.url, self.raw_dir))
    
    def proc(i_path):
        face2edge = transforms.FaceToEdge(remove_faces=False)
        
        i,path = i_path 
        print('Processing shape %d' % i)
        mesh = torch_geometric.io.read_obj(path)
        mesh.y = i%20 # set the mesh class (note that SHREC14 models are ordered by class)
        mesh.subject = int(i/20)
        face2edge(mesh)
        mesh.idx = i

        # add decimation matrices
        pos, faces = mesh.pos.numpy(), mesh.face.t().numpy()
        _,F,E,D = generate_transform_matrices(pos, faces, [4,4,4])
        mesh.downscale_matrices = [_scipy_to_torch_sparse(d) for d in D]
        mesh.downscaled_edges = [_scipy_to_torch_sparse(e) for e in E]
        mesh.downscaled_faces = [torch.tensor(f).t() for f in F]

        # add data to the save list
        return mesh

    def process(self):
        datalist_file = os.path.join(self.processed_dir,"tmp.pt")

        # Read data into huge `Data` list.
        if os.path.exists(datalist_file):
            data_list = torch.load(datalist_file)
        else:
            data_list = []
        
        p = Pool(40)
        for s in range(len(data_list),300,40):
            print('Shapes from %d to %d' % (s, s+(len(self.raw_paths[s:s+40]))-1) )
            data_list = data_list + p.map(Shrec14Dataset.proc,enumerate(self.raw_paths[s:s+40],start=s))
            torch.save(data_list, datalist_file)

        downscale_matrices = []
        downscaled_edges = []
        downscaled_faces = []
        for mesh in data_list:
            downscale_matrices.append(mesh.downscale_matrices)
            downscaled_edges.append([i for (i,v,s) in mesh.downscaled_edges])
            downscaled_faces.append(mesh.downscaled_faces)

            delattr(mesh, "downscale_matrices")
            delattr(mesh, "downscaled_edges")
            delattr(mesh, "downscaled_faces")

        data, slices = self.collate(data_list)
        torch.save( (data, slices), self.processed_paths[0])
        torch.save((downscale_matrices,downscaled_edges,downscaled_faces), self.processed_paths[1])

    def get(self,idx): 
        data = super().get(idx)
                
        if self.nvert>0:
            outv,outt = remesh(data.pos,data.face.t(),self.nvert)
            data.pos,data.face = torch.from_numpy(outv).float(),torch.from_numpy(outt).t().long()
            face2edge = transforms.FaceToEdge(remove_faces=False)
            face2edge(data)
        
        if self.spectral:
            data.eigvals, data.eigvecs = utils.eigenpairs(data.pos, data.face.t(), K=50, double_precision=False)
            data.stiff, data.area = utils.laplacebeltrami_FEM_v2(data.pos, data.face.t())
        data.downscale_matrices = self.downscale_matrices[idx:idx+1]
        data.downscaled_edges = self.downscaled_edges[idx:idx+1]
        data.downscaled_faces = self.downscaled_faces[idx:idx+1]
            
        data.oripos = data.pos.clone()     
        return data
    


def _scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape
    return (i,v, torch.Size(shape))


class Shrec14Dataset_retrivial(Shrec14Dataset):
    def __init__(self, 
        root:str, 
        device:torch.device=torch.device("cpu"),
        train:bool=True, test:bool=True,
        transform_data:bool=True,
        synth=False,
        spectral=False):
        self.url = 'http://www.cs.cf.ac.uk/shaperetrieval/shrec14/'
        
        self.spectral=spectral
        
        def to_device(mesh:torch_geometric.data.Data):
            mesh.pos = mesh.pos.to(device)
            mesh.edge_index = mesh.edge_index.to(device)
            mesh.face = mesh.face.to(device)
            mesh.y = mesh.y.to(device)
            mesh.subject = mesh.subject.to(device)
            return mesh

        
        transform = None
        if transform_data:            
            # rotate and move
            if synth:
                print('synth')
                transform = transforms.Compose([
                    Rotate(dims=[1]), 
                    Move(mean=[0,0,0], std=[0.02,0.06,0.02]),
                    to_device])
            else:
                transform = transforms.Compose([
                    Rotate(dims=[0,1,2]), 
                    Move(mean=[0,0,0], std=[0.05,0.05,0.05]),
                    to_device])

            # center each mesh into its centroid
            pre_transform = Move(mean=[0,0,0], std=[0.0,0.0,0.0])
            super().__init__(root=root,device=device, train=True, test=True, transform_data=transform_data, transform=transform)
        else:
            super().__init__(root=root,device=device, train=True, test=True, transform_data=transform_data, transform=transform)

#         self.data.y = self.data.subject
#         print(type(self.data.subject))
#         print(self.get(0))
        n_poses = 10
        n_subj = 40
        if synth:
            n_poses = 20
            n_subj = 15
        t_size = n_poses//5    
        nshapes = n_poses*n_subj;
        
        all_data = []
        for i in range(nshapes):
            data = super().get(i)
            data.y =  data.subject
            data.downscale_matrices = self.downscale_matrices[i]
            data.downscaled_edges =   self.downscaled_edges[i]
            data.downscaled_faces =   self.downscaled_faces[i]
            all_data.append(data)
            
 
           
        np.random.seed(0)
        idxs = torch.from_numpy(np.asarray([np.random.permutation(n_poses) for i in range(n_subj)]))
        idxs = np.asarray(idxs + (n_poses*np.arange(n_subj)[:,None]),'int32')
 
        if train and not test:
            self.data, self.slices = self.collate([all_data[i] for i in idxs[:,t_size:].flatten()])
            
        elif not train and test:
            self.data, self.slices = self.collate([all_data[i] for i in idxs[:,:t_size].flatten()])
            
        
    def get(self,idx): 
        print('new')
        data = super().get(idx)
        #data.pos,data.face = remesh(data.pos,data.face.t(),2000)
        
        #if self.spectral:
        #    data.eigvals, data.eigvecs = utils.eigenpairs(data.pos, data.face.t(), K=50, double_precision=False)
        #    data.stiff, data.area = utils.laplacebeltrami_FEM_v2(data.pos, data.face.t())
            #face2edge = transforms.FaceToEdge(remove_faces=False)
            #face2edge(data)
        data.oripos = data.pos.clone()     
        return data
            

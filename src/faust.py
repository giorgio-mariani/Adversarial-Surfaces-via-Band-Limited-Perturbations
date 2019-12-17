import os

import numpy as np
import torch
import torch_geometric.data
import torch_geometric.io as gio
import torch_geometric.transforms as transforms
import tqdm

import mesh.decimation

class FaustDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, root:str):
        super(FaustDataset, self).__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.downscaled_cache = os.path.join(self.processed_dir,"ds.npy")
        self._E, self._F, self._D = None, None, None
        self.url = 'http://faust.is.tue.mpg.de/'


    @property
    def raw_file_names(self):
        tofilename =  lambda x : "tr_reg_"+str(x).zfill(3)+".ply"
        return [tofilename(fi) for fi in range(100)]

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def _ds_cached(self):
        return os.path.exists(self.downscaled_cache)

    @property
    def _ds_loaded(self):
        return self._D is not None and not self._E is None  and not self._F is None

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
    
    def _load_transfrom_data(self):
        abs_meshfile = os.path.join(self.raw_dir, "tr_reg_000.ply")
        # if not cached, then compute and store
        if self._ds_cached and self._ds_loaded:
            return
        else:
            if self._ds_cached: #data is cached, but not loaded (for example after a restart)
                E,F,D = np.load(self.downscaled_cache,allow_pickle=True) #load data
            else: # data is neither cached nor loaded
                data = gio.read_ply(abs_meshfile)
                v, f = data.pos.numpy(), data.face.t().numpy()
                _,F,E,D = misc.mesh_decimation.generate_transform_matrices(v, f, [4,4,4])
                np.save(self.downscaled_cache, (E,F,D))
                
            # assign data to respective fields
            F_t = [torch.tensor(f).t() for f in F]
            D_t = [scipy_to_torch_sparse(d) for d in D]
            E_t = [scipy_to_torch_sparse(e) for e in E]
            self._E, self._F, self._D = E_t, F_t, D_t
    
    @property
    def downscale_matrices(self):
        self._load_transfrom_data()
        return self._D

    @property
    def downscaled_edges(self):
        self._load_transfrom_data()
        return self._E

    @property
    def downscaled_faces(self):
        self._load_transfrom_data()
        return self._F


def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape
    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor


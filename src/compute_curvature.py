import torch
import tqdm
import numpy as np

def tri_areas(vertices, faces):
    v1 = vertices[faces[:, 0], :]
    v2 = vertices[faces[:, 1], :]
    v3 = vertices[faces[:, 2], :]
    
    v1 = v1 - v3
    v2 = v2 - v3
    return torch.norm(torch.cross(v1, v2, dim=1), dim=1) * .5



def laplacebeltrami_FEM(vertices, faces):
    n = vertices.shape[0]
    m = faces.shape[0]
    device = vertices.device

    angles = {}
    for i in (1.0, 2.0, 3.0):
        a = torch.fmod(torch.as_tensor(i - 1), torch.as_tensor(3.)).long()
        b = torch.fmod(torch.as_tensor(i), torch.as_tensor(3.)).long()
        c = torch.fmod(torch.as_tensor(i + 1), torch.as_tensor(3.)).long()

        ab = vertices[faces[:,b],:] - vertices[faces[:,a],:];
        ac = vertices[faces[:,c],:] - vertices[faces[:,a],:];

        ab = torch.nn.functional.normalize(ab, p=2, dim=1)
        ac = torch.nn.functional.normalize(ac, p=2, dim=1)
        
        o = torch.mul(ab, ac)
        o = torch.sum(o, dim=1)
        o = torch.acos(o)
        o = torch.div(torch.cos(o), torch.sin(o))
        
        angles[i] = o
    
    indicesI = torch.cat((faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 2], faces[:, 1], faces[:, 0]))
    indicesJ = torch.cat((faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 1], faces[:, 0], faces[:, 2]))
    indices = torch.stack((indicesI, indicesJ))
    
    one_to_n = torch.arange(0, n, dtype=torch.long, device=device)
    eye_indices = torch.stack((one_to_n, one_to_n))

    values = torch.cat((angles[3], angles[1], angles[2], angles[1], angles[3], angles[2])) * 0.5

    stiff = torch.sparse_coo_tensor(indices=indices, dtype=values.dtype,
                                 values=-values,
                                 device=device,
                                 size=(n, n)).coalesce()
    stiff = stiff + torch.sparse_coo_tensor(indices=eye_indices, dtype=values.dtype,
                                 values=-torch.sparse.sum(stiff, dim=0).to_dense(),
                                 device=device,
                                 size=(n, n)).coalesce()
    
    areas = tri_areas(vertices, faces)
    areas = areas.repeat(6) / 12
    
    mass = torch.sparse_coo_tensor(indices=indices, dtype=values.dtype,
                             values=areas,
                             device=device,
                             size=(n, n)).coalesce()
    mass = mass + torch.sparse_coo_tensor(indices=eye_indices, dtype=values.dtype,
                                 values=torch.sparse.sum(mass, dim=0).to_dense(),
                                 device=device,
                                 size=(n, n)).coalesce()

    lumped_mass = torch.sparse.sum(mass, dim=1).to_dense()
    return stiff, mass, lumped_mass

def meancurvature(pos, faces):
  cpu = torch.device("cpu")
  if type(pos) != np.ndarray:
    pos = pos.to(cpu).clone().detach().numpy()
  if pos.shape[-1] != 3:
    raise ValueError("Vertices positions must have shape [n,3]")
  if type(faces) != np.ndarray:
    faces = faces.to(cpu).clone().detach().numpy()
  if faces.shape[-1] != 3:
    raise ValueError("Face indices must have shape [m,3]") 
  if intensity is None:
    intensity = np.ones([pos.shape[0]])
  elif type(intensity) != np.ndarray:
    intensity = intensity.to(cpu).clone().detach().numpy()
  n = pos.shape[0]
  stiff, mass, lumped = laplacebeltrami_FEM(pos, faces)
  ai, av = mass
  mcf = tsparse.spmm(ai, torch.reciprocal(av), n, n, tsparse.spmm(stiff, n, n, pos))
  return mcf.norm(dim=-1, p=2), stiff, mass

def compute_meancurvdiff(ppos, pos, faces):
  mcp, ,   = meancurvature(ppos, faces)
  mc, , (_, a) = meancurvature(pos, faces)
  diff_curvature = mc-mcp
  curvature_dist = (adiff_curvature**2).sum().sqrt().item()
  return curvature_dist
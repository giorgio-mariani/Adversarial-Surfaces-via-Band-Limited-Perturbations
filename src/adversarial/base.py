import torch
import tqdm


import utils

class AdversarialExample(object):
  def __init__(self,
      pos:torch.Tensor,
      edges:torch.LongTensor,
      faces:torch.LongTensor,
      classifier:torch.nn.Module,
      target:int = None):
    super().__init__()
    utils.check_data(pos, edges, faces, float_type=torch.float)
    float_type = pos.dtype

    self.pos = pos
    self.faces = faces
    self.edges = edges
    self.classifier = classifier
    
    if target is not None:
        self.target = torch.tensor([target], device=pos.device, dtype=torch.long)
    else:
        self.target = None

    #constants
    self.vertex_count = pos.shape[0]
    self.edge_count = edges.shape[0]
    self.face_count = faces.shape[0]

    # compute useful data
    self.stiff, self.area = utils.laplacebeltrami_FEM_v2(self.pos, self.faces)
    
  @property
  def device(self):  return self.pos.device
  
  @property
  def dtype_int(self): return self.edges.dtype

  @property
  def dtype_float(self): return self.pos.dtype

  @property
  def perturbed_pos(self):
    raise NotImplementedError()

  @property
  def is_successful(self):
    adversarial_prediction = utils.prediction(self.classifier, self.perturbed_pos).item()
    prediction = utils.prediction(self.classifier, self.pos).item()
    if self.is_targeted:
      return adversarial_prediction == self.target
    else:
      return  prediction != adversarial_prediction
    
  @property
  def is_targeted(self):
      return self.target is not None

class Builder(object):
  def __init__(self):
    super().__init__()
    self.adex_data = dict()

  def set_mesh(self, pos, edges, faces):
    self.adex_data["pos"] = pos
    self.adex_data["edges"] = edges
    self.adex_data["faces"] = faces
    return self
    
  def set_target(self, t:int):
    self.adex_data["target"] = t
    return self

  def set_classifier(self, classifier:torch.nn.Module):
    self.adex_data["classifier"] = classifier
    return self

  def set_parameters(self, **args):
      for k,v in args.items(): self.adex_data[k] = v
      return self

  def build(self, usetqdm:str=None)->AdversarialExample:
    raise NotImplementedError()
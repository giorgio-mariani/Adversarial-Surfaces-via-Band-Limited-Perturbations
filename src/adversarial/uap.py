
import torch

def _pred(Z):
    _, index = torch.sort(Z, dim=0)
    return index[-1]

def _target(Z):
    _, index = torch.sort(Z, dim=0)
    return index[-2]

def error(data):
    return 1


def projection(v, r, eps):
    return 1
    

def UAP_computation(
    data,
    adv_builder,
    delta:float, 
    eps:float, model):

    #filter data:
    v = torch.zeros([6890, 3])
    while error(data) <= (1-delta):
        for meshi in data:
            xi = meshi.pos
            ei = meshi.edge_index
            fi = meshi.face.t()
            yi = meshi.y

            
            # if the classifier is incorrect skip the next steps
            if  _pred(model(xi + v)) != yi: continue
            
            if _pred(model(xi + v)) == yi:
                #Compute the minimal perturbation that sends xi + v 
                # to the decision boundary:
                target = _target(model(xi + v))
                adv_builder.set_mesh(xi+v, ei, fi)
                adv_builder.set_target(target)
                adex = adv_builder.build_and_tune(
                    starting_coefficient=1,
                    search_iterations=10,
                    minimization_iterations=100)
                r = xi - adex.perturbed_pos
                
                #Update the perturbation:
                v = projection(v, r, eps)
    return v
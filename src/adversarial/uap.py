
import torch
import tqdm

def _pred(Z):
    _, index = torch.sort(Z, dim=0)
    return index[-1]

def _target(Z):
    _, index = torch.sort(Z, dim=0)
    return index[-2]

def error(data, classifier, v):
    #percentage of misclassified data
    s = 0
    for m in data:
        x, y = m.pos, m.y
        if _pred(classifier(x+v)) != y: s+=1
    error = s/len(data)
    print("mean error: {}".format(error))
    return error


def projection(v, r, eps):
    p = v + r
    return p
    

def UAP_computation(
    data,
    adv_builder,
    classifier,
    delta:float,
    eps:float,
    starting_coeff:float,
    learning_rate:float):

    device = data[0].pos.device
    typefloat = data[0].pos.dtype
    shape = data[0].pos.shape

    #filter data:
    filter_index = torch.tensor(
        [i for i in range(len(data)) if _pred(classifier(data[i].pos)) == data[i].y],
        dtype=torch.long, device=device)
    data = data[filter_index]

    # start universal adversarial perturbation computation
    v = torch.zeros(shape, dtype=typefloat, device=device)
    while error(data, classifier, v) <= (1-delta):
        for mi in tqdm.tqdm(data):
            xi = mi.pos
            ei = mi.edge_index.t().to(device)
            fi = mi.face.t().to(device)
            yi = mi.y
            
            if _pred(classifier(xi + v)) == yi:
                #Compute the minimal perturbation that sends xi + v 
                # to the decision boundary:
                adv_builder.set_mesh(xi+v, ei, fi)
                adv_builder.set_target(_target(classifier(xi + v)))
                adex = adv_builder.build_and_tune(
                    starting_coefficient=starting_coeff,
                    search_iterations=2,
                    minimization_iterations=2,
                    learning_rate=learning_rate)
                r = (xi - adex.perturbed_pos).detach()
                
                #Update the perturbation:
                v = projection(v, r, eps)
    return v
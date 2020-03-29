import torch


def get_gradient(classifier, x, y):
    x:torch.Tensor = x.clone().detach().requires_grad_(True)
    criterion = torch.nn.CrossEntropyLoss()

    if x.grad is not None:
        x.grad.detach_()
        x.grad.zero_()

    z = classifier(x).view(1,-1)
    loss = criterion(z, y)
    loss.backward()
    return  x.grad

def fast_gradient(classifier, x, y, eigvecs, areavec, max_eps=1, N=20, targeted:bool=False):
    # binary search
    gradient = get_gradient(classifier, x, y)
    gradient = -gradient if targeted else gradient

    signed_grad = torch.sign(gradient)
    signed_grad_spectral = eigvecs.t().mm(torch.diag(areavec).mm(signed_grad))
    signed_grad_filtered = eigvecs.mm(signed_grad_spectral)
    eps_min, eps_max, x_adv = 0, 0.1, None
    for i in range(N):
        eps = (eps_min+eps_max)/2
        tmp = x + eps*signed_grad_filtered
        yp = classifier(tmp).argmax()
        if yp != y:
            eps_max = eps
            x_adv = tmp
        else:
            eps_min = eps
    return x_adv
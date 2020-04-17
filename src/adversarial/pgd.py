import torch
import scipy

from .base import AdversarialExample, Builder
import utils

class FGSMBuilder(Builder):
    def __init__(self):
        super().__init__()

    def set_alpha(self, alpha=0.1):
        self.adex_data["alpha"]=alpha
        return self

    def build(self, usetqdm=None) -> AdversarialExample:
        adex = FGSMAdversarialExample(**self.adex_data)
        adex.compute()
        return adex

class FGSMAdversarialExample(AdversarialExample):
    def __init__(self, 
        pos, edges, faces, 
        classifier, 
        target:int=None,
        eigs_num:int=100, 
        alpha:float=1):
        super().__init__(
            pos=pos, edges=edges, faces=faces,
            classifier=classifier, target=target)
        self.eigvals, self.eigvecs = utils.eigenpairs(pos, faces, K=eigs_num, double_precision=True)
        self.eigs_num = eigs_num
        self.alpha = alpha

    def get_gradient(self, y):
        x = self.pos.clone().detach().requires_grad_(True)
        criterion = torch.nn.CrossEntropyLoss()

        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()

        z = self.classifier(x).view(1,-1)
        loss = criterion(z, y)
        loss.backward()
        return  x.grad

    def project(self, r):
        return r

    def compute(self):
        if self.is_targeted:
            gradient = -self.get_gradient(self.target)
        else:
            y = utils.misc.prediction(self.classifier, self.pos).view(1)
            gradient = self.get_gradient(y)
        self._perturbed_pos = self.pos + self.alpha*torch.sign(gradient)

    @property
    def perturbed_pos(self):
        return self._perturbed_pos

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class PGDBuilder(FGSMBuilder):
    def __init__(self):
        super().__init__()
    
    def set_iterations(self, iterations:int=1):
        self.adex_data["iterations"] = iterations
        return self

    def set_eigs_number(self,k=100):
        self.adex_data["eigs_num"] = k
        return self

    def set_epsilon(self, eps=1):
        self.adex_data["eps"] = eps
        return self 
    
    def set_projection(self, proj_func):
        self.adex_data["projection"] = proj_func
        return self

    def build(self, usetqdm=None) -> AdversarialExample:
        adex = PGDAdversarialExample(**self.adex_data)
        adex.compute()
        return adex

class PGDAdversarialExample(FGSMAdversarialExample):
    def __init__(self, 
        pos, edges, faces, 
        classifier, 
        target:int=None,
        eigs_num:int=100, 
        iterations:int=10,
        projection=None,
        alpha:float=1,
        eps:float=1):
        super().__init__(
            pos=pos, edges=edges, faces=faces,
            classifier=classifier, target=target,
            eigs_num=eigs_num, alpha=alpha)
        self.iterations = iterations
        self.project = projection
        self.eps = torch.tensor(eps, device=pos.device, dtype=pos.dtype)

    def compute(self):
        x_adv = self.pos
        adv_r = (self.pos - self.pos) #simple way to get zero
        for i in range(self.iterations):
            # get gradient
            if self.is_targeted:
                gradient = -self.get_gradient(self.target)
            else:
                y = utils.misc.prediction(self.classifier, self.pos + adv_r).view(1) #assuming classifier is correct
                gradient = self.get_gradient(y)
            
            # compute step
            adv_r = self.project(self, adv_r + self.alpha*torch.sign(gradient))

        self._perturbed_pos = self.pos + adv_r

    @property
    def perturbed_pos(self):
        return self._perturbed_pos


def clip(adex, x):
    eps = adex.eps
    return torch.max(torch.min(x, eps), -eps)

def lowband_filter(adex, x):
    x_spectral = adex.eigvecs.t().mm(torch.diag(adex.area[1]).mm(x))
    x_filtered = adex.eigvecs.mm(x_spectral)
    x_filtered = clip(adex, x_filtered)
    return x_filtered
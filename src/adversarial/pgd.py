import torch
import scipy

from .base import AdversarialExample, Builder
import utils

class FGSMBuilder(Builder):
    def __init__(self):
        super().__init__()

    def set_epsilon(self, eps=0.1):
        self.adex_data["eps"]=eps
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
        eps:float=1):
        super().__init__(
            pos=pos, edges=edges, faces=faces,
            classifier=classifier, target=target)
        self.eigvals, self.eigvecs = utils.eigenpairs(pos, faces, K=eigs_num, double_precision=True)
        self.eigs_num = eigs_num
        self.eps = eps

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
        self._perturbed_pos = self.pos + self.eps*torch.sign(gradient)

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
        eps:float=1):
        super().__init__(
            pos=pos, edges=edges, faces=faces,
            classifier=classifier, target=target,
            eigs_num=eigs_num, eps=eps)
        self.iterations = iterations
    
    def project(self, x):
        x_spectral = self.eigvecs.t().mm(torch.diag(self.area[1]).mm(x))
        x_filtered = self.eigvecs.mm(x_spectral)
        return x_filtered

    def compute(self):
        x_adv = self.pos
        for i in range(self.iterations):
            # get gradient
            if self.is_targeted:
                gradient = -self.get_gradient(self.target)
            else:
                y = utils.misc.prediction(self.classifier, x_adv).view(1) #assuming classifier is correct
                gradient = self.get_gradient(y)
            
            # compute step
            x_adv = x_adv + self.project(self.eps*torch.sign(gradient))

        self._perturbed_pos = x_adv

    @property
    def perturbed_pos(self):
        return self._perturbed_pos
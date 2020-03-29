import torch
import scipy

from .base import AdversarialExample, Builder
import utils

class FGSMBuilder(Builder):
    def __init__(self):
        super().__init__()
    
    def set_search_iterations(self, iterations:int=1):
        self.adex_data["search_iterations"] = iterations
    
    def set_eigvectors_number(self,k=100):
        self.adex_data["eigs_num"] = k

    def set_epsilon(self, eps=0.1):
        self.adex_data["eps"]=eps

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
        search_iterations:int=1,
        eps:float=1):
        super().__init__(
            pos=pos, edges=edges, faces=faces,
            classifier=classifier, target=target)
        self.eigvals, self.eigvecs = utils.eigenpairs(pos, faces, K=eigs_num)
        self.eigs_num = eigs_num
        self.search_iterations = search_iterations
        self.eps = eps

    def get_gradient(self, y):
        classifier= self.classifier
        x = self.pos.clone().detach().requires_grad_(True)
        criterion = torch.nn.CrossEntropyLoss()

        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()

        z = classifier(x).view(1,-1)
        loss = criterion(z, y)
        loss.backward()
        return  x.grad

    def compute(self):
        classifier = self.classifier
        x = self.pos
        eigvecs = self.eigvecs
        areaindex, areavec = self.area

        if self.is_targeted:
            gradient = -self.get_gradient(self.target)
        else:
            y = utils.prediction(classifier, self.pos).view(1)
            gradient = self.get_gradient(y)

        signed_grad = torch.sign(gradient)
        signed_grad_spectral = eigvecs.t().mm(torch.diag(areavec).mm(signed_grad))
        signed_grad_filtered = eigvecs.mm(signed_grad_spectral)
        eps_min, eps_max = 0, self.eps*2
        x_adv = None
        
        for i in range(self.search_iterations):
            eps = (eps_min+eps_max)/2
            tmp = x + eps*signed_grad_filtered
            yp = utils.prediction(classifier, tmp)
            if yp != y:
                eps_max = eps
                x_adv = tmp
            else:
                eps_min = eps
        self._perturbed_pos = x_adv if x_adv is not None else tmp

    @property
    def perturbed_pos(self):
        return self._perturbed_pos


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.multivariate_normal import MultivariateNormal


def Hessian(phi4, x):
    batch_size = x.shape[0]
    dim = x.shape[-1]
    H = torch.eye(dim, device=x.device).unsqueeze(0).expand((batch_size, -1, -1)) * \
        (3 * phi4.coef + 1 / phi4.coef * (3 * x.unsqueeze(-1) ** 2 - 1))
    triu_matrix = torch.triu(torch.triu(torch.ones((dim, dim), device=x.device), diagonal=-1).T,
                             diagonal=-1).unsqueeze(0).expand((batch_size, -1, -1))
    H -= phi4.coef * triu_matrix
    return H


def U_Laplace(x, phi4):
    x_ = F.pad(input=x, pad=(1,) * (2 * 1), mode='constant', value=0)
    grad_term = ((x_[:, 1:] - x_[:, :-1]) ** 2 / 2).sum(-1)
    V = ((1 - x ** 2) ** 2 / 4 + phi4.b * x).sum(-1)
    coef = phi4.a * phi4.dim_grid
    return grad_term * coef + V / coef


def log_Laplace(x, phi4):
    log_Laplace = - phi4.beta * U_Laplace(x, phi4)
    log_Laplace_corr = phi4.dim_phys / 2 * math.log(2 * math.pi / phi4.beta)
    log_Laplace_corr -= torch.logdet(Hessian(phi4, x))
    return log_Laplace, log_Laplace + log_Laplace_corr


class PhiFour(nn.Module):
    def __init__(self, a, b, dim_grid, dim_phys=1,
                 beta=1,
                 bc=('dirichlet', 0),
                 tilt=None,
                 device='cpu'):
        """
        Class to handle operations around PhiFour model
        Args:
            a: coupling term coef
            b: local field coef
            dim_grid: grid size in one dimension
            dim_phys: number of dimensions of the physical grid
            beta: inverse temperature
            tilt: None or {"val":0.7, "lambda":0.1} - for biasing distribution
        """
        self.device = device

        self.a = a
        self.b = b
        self.beta = beta
        self.dim_grid = dim_grid
        self.dim_phys = dim_phys
        self.sum_dims = tuple(i + 1 for i in range(dim_phys))

        self.bc = bc
        self.tilt = tilt

    def init_field(self, n_or_values):
        if isinstance(n_or_values, int):
            x = torch.rand((n_or_values,) + (self.dim_grid,) * self.dim_phys)
            x = x * 2 - 1
        else:
            x = n_or_values
        return x

    def reshape_to_dimphys(self, x):
        if self.dim_phys == 2:
            x_ = x.reshape(-1, self.dim_grid, self.dim_grid)
        else:
            x_ = x
        return x_

    def V(self, x):
        x = self.reshape_to_dimphys(x)
        coef = self.a * self.dim_grid
        V = ((1 - x ** 2) ** 2 / 4 + self.b * x).sum(self.sum_dims) / coef
        if self.tilt is not None:
            tilt = (self.tilt['val'] - x.mean(self.sum_dims)) ** 2
            tilt = self.tilt["lambda"] * tilt / (4 * self.dim_grid)
            V += tilt
        return V

    def U(self, x):
        # Does not include the temperature! need to be explicitely added in Gibbs factor
        assert self.dim_phys < 3
        x = self.reshape_to_dimphys(x)

        if self.bc[0] == 'dirichlet':
            x_ = F.pad(input=x, pad=(1,) * (2 * self.dim_phys), mode='constant',
                       value=self.bc[1])
        elif self.bc[0] == 'pbc':
            # adding "channel dimension" for circular torch padding
            x_ = x.unsqueeze(0)
            # only pad one side, not to double count gradients at the edges
            x_ = F.pad(input=x_, pad=(1, 0,) * (self.dim_phys), mode='circular')
            x_.squeeze_(0)
        else:
            raise NotImplementedError("Only dirichlet and periodic BC"
                                      "implemeted for now")

        if self.dim_phys == 2:
            grad_x = ((x_[:, 1:, :-1] - x_[:, :-1, :-1]) ** 2 / 2)
            grad_y = ((x_[:, :-1, 1:] - x_[:, :-1, :-1]) ** 2 / 2)
            grad_term = (grad_x + grad_y).sum(self.sum_dims)
        else:
            grad_term = ((x_[:, 1:] - x_[:, :-1]) ** 2 / 2).sum(self.sum_dims)

        coef = self.a * self.dim_grid
        return grad_term * coef + self.V(x)

    def grad_U(self, x_init):
        x = x_init.detach()
        x = x.requires_grad_()
        optimizer = torch.optim.SGD([x], lr=0)
        optimizer.zero_grad()
        loss = self.U(x).sum()
        loss.backward()
        return x.grad.data

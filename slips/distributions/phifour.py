import math
import torch
import torch.nn.functional as F


def run_gdflow(U, grad_U, x_, n_steps, dt, verbose_freq=100):
    x = x_.clone()
    r = range(n_steps)
    for t in r:
        x -= dt * grad_U(x)
    return x.detach()


class PhiFour(torch.nn.Module):
    def __init__(self, a, b, dim_grid, dim_phys=1,
                 beta=1,
                 bc=('dirichlet', 0),
                 tilt=None):
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
        self.a = a
        self.b = b
        self.beta = beta
        self.dim_grid = dim_grid
        self.dim_phys = dim_phys
        self.sum_dims = tuple(i + 1 for i in range(dim_phys))
        self.bc = bc
        self.tilt = tilt
        self.coef = self.a * self.dim_grid
        super().__init__()

    def reshape_to_dimphys(self, x):
        if self.dim_phys == 2:
            x_ = x.reshape(-1, self.dim_grid, self.dim_grid)
        else:
            x_ = x
        return x_

    def V(self, x):
        x = self.reshape_to_dimphys(x)
        V = ((1 - x ** 2) ** 2 / 4 + self.b * x).sum(self.sum_dims) / self.coef
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

        return grad_term * self.coef + self.V(x)

    def log_prob(self, x):
        return -self.beta * self.U(x)

    def grad_U(self, x):
        assert self.bc == ('dirichlet', 0)
        assert self.dim_phys != 2
        assert self.tilt is None
        x = self.reshape_to_dimphys(x)
        ret = (self.b - x * (1. - torch.square(x))) / self.coef
        ret[:, 1:-1] += self.coef * (2. * x[:, 1:-1] - x[:, 2:] - x[:, :-2])
        ret[:, 0] += self.coef * (2. * x[:, 0] - x[:, 1])
        ret[:, -1] += self.coef * (2. * x[:, -1] - x[:, -2])
        return ret

    def Hessian(self, x):
        dim = x.shape[-1]
        H = torch.eye(dim, device=x.device) * (3 * self.coef + 1 / self.coef * (3 * x ** 2 - 1))
        H -= self.coef * torch.triu(torch.triu(torch.ones_like(H),
                                               diagonal=-1).T, diagonal=-1)
        return H

    def log_Laplace(self, x):
        log_Laplace = -self.beta * self.U(x.unsqueeze(0)).squeeze(0)
        log_Laplace_corr = (self.dim_grid / 2) * math.log(2 * math.pi / self.beta)
        log_Laplace_corr -= 0.5 * torch.logdet(self.Hessian(x))
        return log_Laplace, log_Laplace + log_Laplace_corr

    def compute_stats_integration(self):
        # Compute the minimum
        x_init = torch.ones((2, self.dim_grid))
        x_init[1, :] *= -1
        self.x_min = run_gdflow(self.U, self.grad_U, x_init, n_steps=10000, dt=5e-3)
        # Compute the energy difference
        log_Laplace_pos, log_Laplace_cor_pos = self.log_Laplace(self.x_min[0])
        log_Laplace_neg, log_Laplace_cor_neg = self.log_Laplace(self.x_min[1])
        en_diff, en_diff_cor = log_Laplace_neg - log_Laplace_pos, log_Laplace_cor_neg - log_Laplace_cor_pos
        # Compute the weights
        return torch.exp(en_diff).item(), torch.exp(en_diff_cor).item()

    def compute_phi_four_weight(self, samples):
        mask = (samples[:, int(self.dim_grid / 2)] > 0)
        return ((1. - mask.float().mean()) / mask.float().mean())

    def _apply(self, fn):
        new_self = super(PhiFour, self)._apply(fn)
        if hasattr(new_self, 'x_min'):
            new_self.x_min = fn(new_self.x_min)
        return new_self

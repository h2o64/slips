# Collection of alphas for generalized stochastic localization

# Libraries
import math

def binary_search(f, low, high, target_value, n_attemps):
    """Binary search function"""
    for _ in range(n_attemps):
        # Get the middle point
        mid = (low + high) / 2.
        ret = f(mid)
        # Check the different conditions
        if ret < target_value:
            low = mid
        else:
            high = mid
    return (low + high) / 2.

class Alpha:
    """Object defining the alpha(t) factor in generalized stochastic localization algorithm"""
    can_use_ei = False
    is_classic = False
    is_finite_time = False
    def g(self, t):
        raise NotImplementedError('g is not implemented yet.')
    def g_inv(self, t):
        raise NotImplementedError('g_inv is not implemented yet.')
    def alpha(self, t):
        return self.g(t) * math.sqrt(t)
    def alpha_dot(self, t):
        raise NotImplementedError('alpha_dot is not implemented yet.')
    def log_alpha_dot(self, t):
        raise NotImplementedError('log_alpha_dot is not implemented yet.')

class AlphaMnM(Alpha):
    """Implement alpha(t) = 1"""
    def g(self, t):
        return 1.0
    def alpha(self, t):
        return 1.0

class AlphaClassic(Alpha):
    """Implement alpha(t) = sqrt(t) * sqrt(t)"""
    can_use_ei = True
    is_classic = True
    def g(self, t):
        return math.sqrt(t)
    def g_inv(self, t):
        return t**2
    def alpha(self, t):
        return t
    def alpha_dot(self, t):
        return 1.0
    def log_alpha_dot(self, t):
        return 1. / t

class AlphaLogLinear(Alpha):
    """Implement alpha(t) = t^{a/2} * sqrt(t)"""
    can_use_ei = True
    def __init__(self, a=1.0):
        if a < 1.:
            raise ValueError('a has to be greater or equal than 1 (currently {:.2e}).'.format(a))
        self.a = a
    def g(self, t):
        return pow(t, self.a / 2.)
    def g_inv(self, t):
        return pow(t, 2. / self.a)
    def alpha(self, t):
        return pow(t, (self.a + 1.) / 2.)
    def alpha_dot(self, t):
        return (self.a + 1.) * pow(t, (self.a - 1.) / 2.) / 2.
    def log_alpha_dot(self, t):
        return (self.a + 1.) / (2. * t)

class AlphaGeometric(Alpha):
    """Implement alpha(t) = (t / T)^{a/2} * (1 - (t / T))^{-b/2} * sqrt(t)"""
    is_finite_time = True
    def __init__(self, T=1.0, a=1.0, b=1.0):
        if T <= 0.:
            raise ValueError('T has to be greater than 0 (currently {:.2e}).'.format(T))
        if a < 1.:
            raise ValueError('a has to be greater or equal than 1 (currently {:.2e}).'.format(a))
        if b <= 0.:
            raise ValueError('b has to be greater than 0 (currently {:.2e}).'.format(b))
        self.T = T
        self.a = a
        self.b = b
        self.a_equal_b = a == b
        self.is_case_11 = (a == 1.0) and (b == 1.0)
        self.is_case_21 = (a == 2.0) and (b == 1.0)
        self.is_case_105 = (a == 1.0) and (b == 0.5)
        self.is_case_12 = (a == 1.0) and (b == 2.0)
        self.can_use_ei = self.is_case_11 or self.is_case_21 or self.is_case_105 or self.is_case_12
    def g(self, t):
        t_over_T = t / self.T
        if self.is_case_11:
            return math.sqrt(t_over_T / (1. - t_over_T))
        else:
            return pow(t_over_T, self.a / 2.) * pow(1. - t_over_T, -self.b / 2.)
    def g_inv(self, t, n_attemps=512, precision=1e-10):
        if self.a_equal_b:
            t_pow_2_over_alpha = pow(t, 2. / self.a)
            return self.T * t_pow_2_over_alpha / (1. + t_pow_2_over_alpha)
        else:
            return binary_search(self.g, precision, self.T - precision, t, n_attemps)
    def alpha_dot(self, t):
        if self.is_case_11:
            return (1. / math.sqrt(self.T - t)) + ((t / pow(self.T - t, 3. / 2.)) / 2.)
        else:
            ret = (self.a + 1.) * pow(t, (self.a - 1.) / 2.) * pow(1. - (t / self.T), -self.b / 2.)
            ret += self.b * pow(t, (self.a + 1.) / 2.) * pow(self.T, self.b / 2.) / pow(self.T - t, (self.b / 2.) + 1.)
            return 0.5 * ret / pow(self.T, self.a / 2.)
    def log_alpha_dot(self, t):
        return ((self.a + 1.) / (2. * t)) + (self.b / (2. * (self.T - t)))

class AlphaLogTangent(Alpha):
    """Implement alpha(t) = tan(0.5 * pi * t / T)^{alpha / 2} * sqrt(t)"""
    is_finite_time = True
    def __init__(self, T=1.0, a=1.0):
        if T <= 0.:
            raise ValueError('T has to be greater than 0 (currently {:.2e}).'.format(T))
        if a < 1.:
            raise ValueError('a has to be greater or equal than 1 (currently {:.2e}).'.format(a))
        self.T = T
        self.a = a
    def g(self, t):
        return pow(math.tan(0.5 * math.pi * t / self.T), self.a / 2.)
    def g_inv(self, t):
        return 2. * self.T * math.atan(pow(t, 2. / self.a)) / math.pi
    def alpha_dot(self, t):
        pi_t_over_2_T = 0.5 * math.pi * t / self.T
        ret = pow(math.tan(pi_t_over_2_T), self.a / 2.) / (2. * math.sqrt(t))
        ret += math.sqrt(t) * self.a * math.pi * (pow(math.sin(pi_t_over_2_T), (self.a / 2.) - 1.) / pow(math.cos(pi_t_over_2_T), (self.a / 2.) + 1.)) / (4. * self.T)
        return ret
    def log_alpha_dot(self, t):
        pi_t_over_2_T = 0.5 * math.pi * t / self.T
        ret = 1. / (2. * t)
        ret += (self.a * math.pi * (1. + (math.tan(pi_t_over_2_T)**2)) / math.tan(pi_t_over_2_T)) / (4. * self.T)
        return ret

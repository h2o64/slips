# Collection of integrators for different SDEs of generalized stochastic localization

# Libraries
import math
import torch

def make_time_discretization(T, K, eps_start=1e-6, eps_end=1e-6, split_in_middle=False, log_start=False, log_end=False):
    """Make a time discretization scheme

    Args:
        T (float): Maximum time
        K (int): Number of time discretization
        eps_start (float): Starting time (default is 1e-6)
        eps_end (float): Ending time offset (compared to T) (default is 1e-6)
        split_in_middle (bool): Whether to split the discretization at T/2 (default is False)
        log_start (bool): Whether to use logarithmic time near eps_start (default is False)
        log_end (bool): Whether to use logarithmic time near eps_end (default is False)

    Returns:
        ts (torch.Tensor of shape (K,)): Time discretization
    """

    # Check the values
    if eps_start > T - eps_end:
    	raise ValueError('eps_start (={:.2e}) is greating that T - eps_end (= {:.2e} - {:.2e} = {:.2e}).'.format(
    		eps_start, T, eps_end, T - eps_end
    	))

    # Auxiliary function
    def make_a_and_b(p):
        if log_start:
            a = torch.logspace(math.log10(eps_start), math.log10(T/2.), p)
        else:
            a = torch.linspace(eps_start, T/2., p)
        if log_end:
            b = (T - torch.logspace(math.log10(eps_end), math.log10(T/2.), p)).flip(dims=(0,))
        else:
            b = (T - torch.linspace(eps_end, T/2., p)).flip(dims=(0,))
        return a, b

    is_K_even = K % 2 == 0
    if not split_in_middle and log_start:
    	return torch.logspace(math.log10(eps_start), math.log10(T - eps_end), K)
    elif log_start or log_end:
        if is_K_even:
            a, b = make_a_and_b(K // 2)
            return torch.concat([a[:-1], torch.linspace(a[-2], b[1], 2), b[1:]])
        else:
            a, b = make_a_and_b((K // 2) + 1)
            return torch.concat([a[:-1], torch.FloatTensor([T / 2.]), b[1:]])
    else:
        return torch.linspace(eps_start, T-eps_end, K)

def make_time_discretization_from_snr(log_g_sq, log_g_sq_inv, T, K, eps_start=1e-6, eps_end=1e-6):
    """Make a time discretization scheme from the SNR profile

    Args:
        g_sq (function): Function returning log g^2(t)
        g_sq_inv (function): Function returning the inverse of log g^2(t)
        T (float): Maximum time
        K (int): Number of time discretization
        eps_start (float): Starting time (default is 1e-6)
        eps_end (float): Ending time offset (compared to T) (default is 1e-6)

    Returns:
        ts (torch.Tensor of shape (K,)): Time discretization
    """

    # Check the values
    if eps_start > T - eps_end:
    	raise ValueError('eps_start (={:.2e}) is greating that T - eps_end (= {:.2e} - {:.2e} = {:.2e}).'.format(
    		eps_start, T, eps_end, T - eps_end
    	))

    # Evaluate the SNR
    snr_min, snr_max = log_g_sq(eps_start), log_g_sq(T - eps_end)
    # Segement the SNR
    snr_linspace = torch.linspace(snr_min, snr_max, K)
    # Inverse the SNR
    return torch.FloatTensor(list(map(log_g_sq_inv, snr_linspace)))

def integration_step_nabla_log_p_t(y_k, nabla_log_p_t, t_k_p_1, t_k, sigma, alpha, use_exponential_integrator):
	"""Do a single step of integration (either EM or EI) on the following SDE

	    dY_t = (alpha'(t) / alpha(t)) * [Y_t + sigma^2  * t * \nabla \log p_t(Y_t)] dt + \sigma dB_t 

	where \nabla \log p_t(Y_t; \sigma)) is given.

	Args:
	    y_k (torch.Tensor of shape (batch_size, *data_shape)): Sample from step t_k
	    nabla_log_p_t (function): Function corresponding to \nabla \log p_t(Y_t; \sigma))
	    t_k_p_1 (torch.Tensor of shape ()): Time t_{k+1}
	    t_k (torch.Tensor of shape ()): Time t_k
	    sigma (torch.Tensor of shape ()): Value of sigma
	    alpha (Alpha object): Specific type of generalized stochastic localization factor
	    use_exponential_integrator (bool): Whether to use the exponential integrator

	Returns:
	    y_k_p_1 (torch.Tensor of shape (batch_size, *data_shape)): Sample from step t_[k+1}
	"""
	delta = t_k_p_1 - t_k
	if use_exponential_integrator and alpha.can_use_ei:
		ratio = t_k_p_1 / t_k
		if alpha.is_classic:
			y_k_p_1 = ratio * y_k + (ratio - 1.) * torch.square(sigma) * t_k * nabla_log_p_t(y_k, t_k, sigma, alpha)
			y_k_p_1 += sigma * math.sqrt(ratio * delta) * torch.randn_like(y_k_p_1)
		elif alpha.is_finite_time:
			t_k_over_T = t_k / alpha.T
			t_k_p_1_over_T = t_k_p_1 / alpha.T
			factor = pow((1. - t_k_over_T) / (1. - t_k_p_1_over_T), alpha.b / 2.)
			ratio_pow_ap1_over_2 = pow(ratio, (alpha.a + 1.) / 2.)
			y_k_p_1 = ratio_pow_ap1_over_2 * factor * y_k
			y_k_p_1 += (ratio_pow_ap1_over_2 * factor - 1.) * torch.square(sigma) * t_k * nabla_log_p_t(y_k, t_k, sigma, alpha)
			if alpha.is_case_11:
				y_k_p_1 += sigma * math.sqrt((t_k_p_1_over_T / (1. - t_k_p_1_over_T)) * (-t_k_p_1 * math.log(ratio) + alpha.T * (ratio - 1.))) * torch.randn_like(y_k_p_1)
			elif alpha.is_case_21:
				y_k_p_1 += sigma * t_k_p_1 * math.sqrt((ratio - 1.) / (1. - t_k_p_1_over_T)) * math.sqrt(0.5 * ((t_k_p_1 + t_k) / (t_k * t_k_p_1)) - (1. / alpha.T)) * torch.randn_like(y_k_p_1)
			elif alpha.is_case_105:
				factor_new = (math.atanh(math.sqrt(1. - t_k_p_1_over_T)) - math.atanh(math.sqrt(1. - t_k_over_T))) / math.sqrt(alpha.T * (alpha.T - t_k_p_1))
				factor_new -= 1. / t_k_p_1
				factor_new += math.sqrt((alpha.T - t_k) / (alpha.T - t_k_p_1)) / t_k
				y_k_p_1 += sigma * t_k_p_1 * math.sqrt(factor_new) * torch.randn_like(y_k_p_1)
			elif alpha.is_case_12:
				y_k_p_1 += sigma * math.sqrt(delta * (t_k_p_1_over_T**2 + ratio) - 2. * t_k_p_1 * t_k_p_1_over_T * math.log(ratio)) * torch.randn_like(y_k_p_1) / (1. - t_k_p_1_over_T)
			else:
				raise ValueError('This alpha is not EI ready.')
		else:
			ratio_pow_ap1_over_2 = pow(ratio, (alpha.a + 1.) / 2.)
			y_k_p_1 = ratio_pow_ap1_over_2 * y_k
			y_k_p_1 += (ratio_pow_ap1_over_2 - 1.) * torch.square(sigma) * t_k * nabla_log_p_t(y_k, t_k, sigma, alpha)
			y_k_p_1 += sigma * math.sqrt(t_k_p_1 * (pow(ratio, alpha.a) - 1.0) / alpha.a) * torch.randn_like(y_k_p_1)
	else:
		log_alpha_dot_k = alpha.log_alpha_dot(t_k)
		y_k_p_1 = (1. +  delta * log_alpha_dot_k) * y_k
		y_k_p_1 += delta * log_alpha_dot_k * torch.square(sigma) * t_k * nabla_log_p_t(y_k, t_k, sigma, alpha)
		y_k_p_1 += sigma * math.sqrt(delta) * torch.randn_like(y_k_p_1)
	return y_k_p_1

def integration_step_mc_est(y_k, mc_est, t_k_p_1, t_k, sigma, alpha, use_exponential_integrator=None):
	"""Do a single step of integration (either EM or EI) on the following SDE

	    dY_t = alpha'(t) * \mathbb{E}_{X ~ q_t(x|Y_t)}[X] dt + sigma * dB_t

	where \mathbb{E}_{X ~ q_t(x|Y_t)}[X] is given.

	Args:
	    y_k (torch.Tensor of shape (batch_size, *data_shape)): Sample from step t_k
	    mc_est (function): Function corresponding to \mathbb{E}_{X ~ q_t(x|Y_t)}[X]
	    t_k_p_1 (torch.Tensor of shape ()): Time t_{k+1}
	    t_k (torch.Tensor of shape ()): Time t_k
	    sigma (torch.Tensor of shape ()): Value of sigma
	    alpha (Alpha object): Specific type of generalized stochastic localization factor
	    use_exponential_integrator (bool): Whether to use the exponential integrator (EM and EI are equivalent here)

	Returns:
	    y_k_p_1 (torch.Tensor of shape (batch_size, *data_shape)): Sample from step t_[k+1}
	"""

	delta = t_k_p_1 - t_k
	if use_exponential_integrator:
		y_k_p_1 = y_k + (alpha.alpha(t_k_p_1) - alpha.alpha(t_k)) * mc_est(y_k, t_k, sigma, alpha)
		y_k_p_1 += sigma * math.sqrt(delta) * torch.randn_like(y_k_p_1)
	else:
		y_k_p_1 = y_k + delta * alpha.alpha_dot(t_k) * mc_est(y_k, t_k, sigma, alpha)
		y_k_p_1 += sigma * math.sqrt(delta) * torch.randn_like(y_k_p_1)
	return y_k_p_1

def integration_step_mc_est_grad(y_k, mc_est, t_k_p_1, t_k, sigma, alpha, use_exponential_integrator):
	"""Do a single step of integration (either EM or EI) on the following SDE

	    dY_t = [alpha'(t)/alpha(t)] * [Y_t + sigma^2 * t * \mathbb{E}_{X ~ q_t(x|Y_t)}[\nabla \log \pi(X)]] / alpha(t)] dt + sigma dB_t

	where \mathbb{E}_{X ~ q_t(x|Y_t)}[\nabla \log \pi(X)] is given.

	Args:
	    y_k (torch.Tensor of shape (batch_size, *data_shape)): Sample from step t_k
	    mc_est (function): Function corresponding to \mathbb{E}_{X ~ q_t(x|Y_t)}[\nabla \log \pi(X)]
	    t_k_p_1 (torch.Tensor of shape ()): Time t_{k+1}
	    t_k (torch.Tensor of shape ()): Time t_k
	    sigma (torch.Tensor of shape ()): Value of sigma
	    alpha (Alpha object): Specific type of generalized stochastic localization factor
	    use_exponential_integrator (bool): Whether to use the exponential integrator

	Returns:
	    y_k_p_1 (torch.Tensor of shape (batch_size, *data_shape)): Sample from step t_[k+1}
	"""
	delta = t_k_p_1 - t_k
	if use_exponential_integrator and alpha.can_use_ei:
	    ratio = t_k_p_1 / t_k
	    if alpha.is_classic:
	    	y_k_p_1 = ratio * y_k + (ratio - 1.) * torch.square(sigma) * mc_est(y_k, t_k, sigma, alpha)
	    	y_k_p_1 += sigma * math.sqrt(ratio * delta) * torch.randn_like(y_k_p_1)
	    elif alpha.is_finite_time:
	    	t_k_over_T = t_k / alpha.T
	    	t_k_p_1_over_T = t_k_p_1 / alpha.T
	    	factor = math.sqrt((1. - t_k_over_T) / (1. - t_k_p_1_over_T))
	    	y_k_p_1 = ratio * factor * y_k + (ratio * factor - 1.) * torch.square(sigma) * t_k * mc_est(y_k, t_k, sigma, alpha) / alpha.alpha(t_k)
	    	y_k_p_1 += sigma * math.sqrt((t_k_p_1_over_T / (1. - t_k_p_1_over_T)) * (-t_k_p_1 * math.log(ratio) + alpha.T * (ratio - 1.))) * torch.randn_like(y_k_p_1)
	    else:
	    	ratio_pow_ap1_over_2 = pow(ratio, (alpha.a + 1.) / 2.)
	    	y_k_p_1 = ratio_pow_ap1_over_2 * y_k
	    	y_k_p_1 += (ratio_pow_ap1_over_2 - 1.) * torch.square(sigma) * mc_est(y_k, t_k, sigma, alpha) / pow(t_k, (alpha.a - 1.) / 2.)
	    	y_k_p_1 += sigma * math.sqrt(t_k_p_1 * (pow(ratio, alpha.a) - 1.0) / alpha.a) * torch.randn_like(y_k_p_1)
	else:
		log_alpha_dot_k = alpha.log_alpha_dot(t_k)
		y_k_p_1 = (1. + log_alpha_dot_k * delta) * y_k
		y_k_p_1 += delta * log_alpha_dot_k * torch.square(sigma) * t_k * mc_est(y_k, t_k, sigma, alpha) / alpha.alpha(t_k)
		y_k_p_1 += sigma * math.sqrt(delta) * torch.randn_like(y_k_p_1)
	return y_k_p_1

def integration_step_mc_est_reparam(y_k, mc_est, t_k_p_1, t_k, sigma, alpha, use_exponential_integrator):
	"""Do a single step of integration (either EM or EI) on the following SDE

	    dY_t = [alpha'(t)/alpha(t)] * [Y_t + sigma * sqrt(t) * \mathbb{E}_{Z ~ q(z | Y_t)}[Z]] dt + sigma dB_t

	where \mathbb{E}_{Z ~ q(z | Y_t)}[Z] is given.

	Args:
	    y_k (torch.Tensor of shape (batch_size, *data_shape)): Sample from step t_k
	    mc_est (function): Function corresponding to \mathbb{E}_{Z ~ q(z | Y_t)}[Z]
	    t_k_p_1 (torch.Tensor of shape ()): Time t_{k+1}
	    t_k (torch.Tensor of shape ()): Time t_k
	    sigma (torch.Tensor of shape ()): Value of sigma
	    alpha (Alpha object): Specific type of generalized stochastic localization factor
	    use_exponential_integrator (bool): Whether to use the exponential integrator

	Returns:
	    y_k_p_1 (torch.Tensor of shape (batch_size, *data_shape)): Sample from step t_[k+1}
	"""
	delta = t_k_p_1 - t_k
	if use_exponential_integrator and alpha.can_use_ei:
	    ratio = t_k_p_1 / t_k
	    if alpha.is_classic:
	    	y_k_p_1 = ratio * y_k + (ratio - 1.) * sigma * math.sqrt(t_k) * mc_est(y_k, t_k, sigma, alpha)
	    	y_k_p_1 += sigma * math.sqrt(ratio * delta) * torch.randn_like(y_k_p_1)
	    elif alpha.is_finite_time:
	    	t_k_over_T = t_k / alpha.T
	    	t_k_p_1_over_T = t_k_p_1 / alpha.T
	    	factor = math.sqrt((1. - t_k_over_T) / (1. - t_k_p_1_over_T))
	    	y_k_p_1 = ratio * factor * y_k + (ratio * factor - 1.) * sigma * math.sqrt(t_k) * mc_est(y_k, t_k, sigma, alpha)
	    	y_k_p_1 += sigma * math.sqrt((t_k_p_1_over_T / (1. - t_k_p_1_over_T)) * (-t_k_p_1 * math.log(ratio) + alpha.T * (ratio - 1.))) * torch.randn_like(y_k_p_1)
	    else:
	    	ratio_pow_ap1_over_2 = pow(ratio, (alpha.a + 1.) / 2.)
	    	y_k_p_1 = ratio_pow_ap1_over_2 * y_k
	    	y_k_p_1 += (ratio_pow_ap1_over_2 - 1.) * sigma * math.sqrt(t_k) * mc_est(y_k, t_k, sigma, alpha)
	    	y_k_p_1 += sigma * math.sqrt(t_k_p_1 * (pow(ratio, alpha.a) - 1.0) / alpha.a) * torch.randn_like(y_k_p_1)
	else:
		log_alpha_dot_k = alpha.log_alpha_dot(t_k)
		y_k_p_1 = (1. + log_alpha_dot_k * delta) * y_k
		y_k_p_1 += delta * log_alpha_dot_k * sigma * math.sqrt(t_k) * mc_est(y_k, t_k, sigma, alpha)
		y_k_p_1 += sigma * math.sqrt(delta) * torch.randn_like(y_k_p_1)
	return y_k_p_1

def integration_step_mc_est_grad_reparam(y_k, mc_est, t_k_p_1, t_k, sigma, alpha, use_exponential_integrator):
	"""Do a single step of integration (either EM or EI) on the following SDE

	    dY_t = [alpha'(t)/alpha(t)] * [Y_t + sigma^2 * t * \mathbb{E}_{Z ~ q(z | Y_t)}[\nabla \log \pi((Y_t / alpha(t)) + (sigma/g(t)) * Z)] / alpha(t)] dt
	    		+ sigma dB_t

	where \mathbb{E}_{Z ~ q(z | Y_t)}[\nabla \log \pi((Y_t / alpha(t)) + (sigma/g(t)) * Z)] is given.

	Args:
	    y_k (torch.Tensor of shape (batch_size, *data_shape)): Sample from step t_k
	    mc_est (function): Function corresponding to \mathbb{E}_{Z ~ q(z | Y_t)}[\nabla \log \pi((Y_t / alpha(t)) + (sigma/g(t)) * Z)]
	    t_k_p_1 (torch.Tensor of shape ()): Time t_{k+1}
	    t_k (torch.Tensor of shape ()): Time t_k
	    sigma (torch.Tensor of shape ()): Value of sigma
	    alpha (Alpha object): Specific type of generalized stochastic localization factor
	    use_exponential_integrator (bool): Whether to use the exponential integrator

	Returns:
	    y_k_p_1 (torch.Tensor of shape (batch_size, *data_shape)): Sample from step t_[k+1}
	"""
	delta = t_k_p_1 - t_k
	if use_exponential_integrator and alpha.can_use_ei:
	    ratio = t_k_p_1 / t_k
	    if alpha.is_classic:
	    	y_k_p_1 = ratio * y_k + (ratio - 1.) * torch.square(sigma) * mc_est(y_k, t_k, sigma, alpha)
	    	y_k_p_1 += sigma * math.sqrt(ratio * delta) * torch.randn_like(y_k_p_1)
	    elif alpha.is_finite_time:
	    	t_k_over_T = t_k / alpha.T
	    	t_k_p_1_over_T = t_k_p_1 / alpha.T
	    	factor = math.sqrt((1. - t_k_over_T) / (1. - t_k_p_1_over_T))
	    	y_k_p_1 = ratio * factor * y_k + (ratio * factor - 1.) * torch.square(sigma) * t_k * mc_est(y_k, t_k, sigma, alpha) / alpha.alpha(t_k)
	    	y_k_p_1 += sigma * math.sqrt((t_k_p_1_over_T / (1. - t_k_p_1_over_T)) * (-t_k_p_1 * math.log(ratio) + alpha.T * (ratio - 1.))) * torch.randn_like(y_k_p_1)
	    else:
	    	ratio_pow_ap1_over_2 = pow(ratio, (alpha.a + 1.) / 2.)
	    	y_k_p_1 = ratio_pow_ap1_over_2 * y_k
	    	y_k_p_1 += (ratio_pow_ap1_over_2 - 1.) * torch.square(sigma) * mc_est(y_k, t_k, sigma, alpha) / pow(t_k, (alpha.a - 1.) / 2.)
	    	y_k_p_1 += sigma * math.sqrt(t_k_p_1 * (pow(ratio, alpha.a) - 1.0) / alpha.a) * torch.randn_like(y_k_p_1)
	else:
		log_alpha_dot_k = alpha.log_alpha_dot(t_k)
		y_k_p_1 = (1. + log_alpha_dot_k * delta) * y_k
		y_k_p_1 += delta * log_alpha_dot_k * torch.square(sigma) * t_k * mc_est(y_k, t_k, sigma, alpha) / alpha.alpha(t_k)
		y_k_p_1 += sigma * math.sqrt(delta) * torch.randn_like(y_k_p_1)
	return y_k_p_1
import numpy as np


master_step_size = 1e-2
coefficients = np.randn((1,10))
fudge_factor = 1e-6
historical_gradient = 0
grad  = 0
historical_gradient += np.pow(grad,2)
adjusted_grad = grad / (fudge_factor + np.sqrt(historical_gradient))
w = w - master_step_size * adjusted_grad
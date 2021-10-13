with open('version', mode='r') as fp:
    __version__ = fp.readline().rstrip()

# We import some of the functions at the top of the module for ease of use.

# Tensorflow loss functions for count data
from mirabolic.neural_glm.actuarial_loss_functions import (
    Poisson_link,
    Poisson_link_with_exposure,
    Negative_binomial_link_1,
    Negative_binomial_link_1_with_exposure,
)

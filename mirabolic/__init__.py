import os

this_dir = os.path.dirname(__file__)
with open(os.path.join(this_dir, 'version'), mode='r') as fp:
    __version__ = fp.readline().rstrip()

import mirabolic.neural_glm as neural_glm

# We import some of the functions for ease of reference.

# Tensorflow loss functions for count data
from mirabolic.neural_glm.actuarial_loss_functions import (
    Poisson_link,
    Poisson_link_with_exposure,
    Negative_binomial_link_1,
    Negative_binomial_link_1_with_exposure,
)

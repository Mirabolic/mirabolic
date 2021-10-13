# Loss functions corresponding to some popular actuarial statistical models.

# These functions all correspond to the negative log-likelihood of the
# corresponding distribution (with data-constant terms suppressed).  In
# typical fashion, we ignore terms that are constant functions of the
# observed data, since they do not affect the gradients.

import tensorflow as tf
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python.ops.math_ops import cast as tf_cast
from tensorflow.math import log, lgamma, exp, sigmoid


def one_dim(t):
    # Convert tensors of shape (N,1) to (N).
    # Tensors of shape (N) remain unchanged.
    return(tf.reshape(t, (-1,)))


def Poisson_link(y_true, y_pred):
    # "y_pred" predicts the Poisson log(lambda), since "log" is the standard
    # Poisson Regression link function
    y_log_lambda = one_dim(convert_to_tensor(y_pred))
    num_events = one_dim(tf_cast(y_true, y_pred.dtype))

    # We ignore terms in the neg log likelihood that are constant given
    # the observations, since they're irrelevant for minimization
    neg_log_likelihood = -(num_events*y_log_lambda - exp(y_log_lambda))
    return neg_log_likelihood


def Poisson_link_with_exposure(y_true, y_pred):
    # "y_pred" predicts log(lambda).  "y_true" is a pair of values,
    # consisting of <N, T>, where "N" is the number of events and "T" is the
    # length of exposure.  (So, for the same lambda, if you have twice the
    # exposure, you'd expect to see about twice the events.)  In the case that
    # T=1 for all observations, this loss function reduces to the Poisson()
    # above.
    y_log_lambda = one_dim(convert_to_tensor(y_pred))
    y_observations = tf_cast(y_true, y_pred.dtype)
    num_events = one_dim(y_observations[:, 0])
    exposure = one_dim(y_observations[:, 1])

    # Rescale Lambda to account for exposure
    y_log_lambda = y_log_lambda + log(exposure)

    neg_log_likelihood = -(num_events*y_log_lambda - exp(y_log_lambda))
    return neg_log_likelihood


def Negative_binomial_link_1(y_true, y_pred):
    # There are multiple possible link functions for negative binomial
    # regression; we present one here.
    num_events = one_dim(tf_cast(y_true, y_pred.dtype))

    y_pred = convert_to_tensor(y_pred)

    r = one_dim(y_pred[:, 0])
    # (Link function:) Convert from R to R^+
    r = exp(r)

    p = one_dim(y_pred[:, 1])
    # (Link function:) Convert from R to [0,1]
    p = sigmoid(p)

    neg_log_likelihood = -(
        lgamma(num_events+r)
        + num_events*log(1-p)
        - lgamma(r)
        + r*log(p))
    return neg_log_likelihood


def Negative_binomial_link_1_with_exposure(y_true, y_pred):
    # To handle exposure, we need to interpret our original
    # distribution as the count function of some underlying
    # stochastic process.  For a Poisson distribution, this
    # is easy; we use the Poisson process, with i.i.d.
    # exponentially distributed interarrival times.

    # The situation is not so clear when we have a negative
    # binomial distribution for the count, because there
    # are multiple stochastic processes we can choose from.

    # In practical terms, given an NB(r, p) distribution,
    # if we wish to scale the rate by alpha, we can either
    # treat the rate as r or as (1-p)/p.  If we wish
    # to rescale exposure by alpha, then, we can either do
    #    r  =>  alpha*r
    # or
    #    p  =>  p / [p + alpha*(1-p)]
    # The former situation corresponds to a Negative Binomial
    # Levy Process, which has a few nice theoretical
    # properties, so we choose to do that.
    y_pred = convert_to_tensor(y_pred)

    r = one_dim(y_pred[:, 0])
    # (Link function:) Convert from R to R^+
    r = exp(r)

    p = one_dim(y_pred[:, 1])
    # (Link function:) Convert from R to [0,1]
    p = sigmoid(p)

    y_observations = tf_cast(y_true, y_pred.dtype)
    num_events = one_dim(y_observations[:, 0])
    exposure = one_dim(y_observations[:, 1])

    r = exposure * r
    neg_log_likelihood = -(
        lgamma(num_events+r)
        + num_events*log(1-p)
        - lgamma(r)
        + r*log(p)
    )
    return neg_log_likelihood

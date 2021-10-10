# Loss functions corresponding to some popular actuarial statistical models.

# These functions all correspond to the negative log-likelihood of the
# corresponding distribution.  In typical fashion, we ignore terms that are
# constant functions of the observed data, since they do not affect the
# gradients.

import tensorflow as tf
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python.ops.math_ops import cast as tf_cast
from tensorflow.math import log, lgamma, exp, sigmoid
import keras.backend as K


def Poisson(y_true, y_pred):
    # "y_pred" predicts the Poisson "lambda".  Note that the standard
    # tf.keras.losses.Poisson() corresponds to -likelihood, whereas this
    # computes the -log(likelihood), which should be more numerically
    # stable, hopefully.
    y_lambda = convert_to_tensor(y_pred)
    num_events = tf_cast(y_true, y_pred.dtype)
    return -K.mean(
        num_events*y_lambda
        - exp(y_lambda),
        axis=-1,
    )


def Poisson_with_exposure(y_true, y_pred):
    # "y_pred" predicts the Poisson "lambda".  "y_true" is a pair of values,
    # consisting of <N, T>, where "N" is the number of events and "T" is the
    # length of exposure.  (So, for the same lambda, if you have twice the
    # exposure, you'd expect to see about twice the events.)  In the case that
    # T=1 for all observations, this loss function reduces to the Poisson()
    # above.
    y_lambda = convert_to_tensor(y_pred)
    y_observations = tf_cast(y_true, y_pred.dtype)
    num_events = y_observations[:, 0]
    num_events = tf.reshape(num_events, (-1,))
    exposure = y_observations[:, 1]
    exposure = tf.reshape(exposure, (-1,))

    # Rescale Lambda to account for exposure
    y_lambda = exposure * y_lambda
    return -K.mean(
        num_events*y_lambda
        - exp(y_lambda),
        axis=-1
    )


def Negative_Binomial(y_true, y_pred):
    num_events = tf_cast(y_true, y_pred.dtype)

    y_pred = convert_to_tensor(y_pred)
    # Extract distribution parameters
    r = y_pred[:, 0]
    r = tf.reshape(r, (-1,))
    # (Link function:) Convert from R to R^+
    r = exp(r)

    p = y_pred[:, 1]
    p = tf.reshape(p, (-1,))
    # (Link function:) Convert from R to [0,1]
    p = sigmoid(p)

    return -K.mean(
        lgamma(num_events+r)
        + num_events*log(1-p)
        - lgamma(r)
        + r*log(p),
        axis=-1
    )


def Negative_Binomial_with_exposure(y_true, y_pred):
    # Natural rate for neg binomial is (1-p)/p
    # So, changing exposure by alpha means we reset to
    # p_rescaled = p / [p + alpha(1-p)]
    y_pred = convert_to_tensor(y_pred)

    # Extract distribution parameters
    r = y_pred[:, 0]
    r = tf.reshape(r, (-1,))
    # (Link function:) Convert from R to R^+
    r = exp(r)

    p = y_pred[:, 1]
    p = tf.reshape(p, (-1,))
    # (Link function:) Convert from R to [0,1]
    p = sigmoid(p)

    y_observations = tf_cast(y_true, y_pred.dtype)
    num_events = y_observations[:, 0]
    num_events = tf.reshape(num_events, (-1,))
    exposure = y_observations[:, 1]
    exposure = tf.reshape(exposure, (-1,))

    p = p / (p + exposure * (1 - p))
    return -K.mean(
        lgamma(num_events+r)
        + y_true*log(1-p)
        - lgamma(r)
        + r*log(p),
        axis=-1
    )

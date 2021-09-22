# Loss functions corresponding to some popular actuarial statistical models.

# These functions all correspond to the negative log-likelihood of the
# corresponding distribution.  In typical fashion, we ignore terms that are
# constant functions of the observed data, since they do not affect the
# gradients.

from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python.ops.math_ops import cast as tf_cast
from tensorflow.math import log, lgamma, exp
import keras.backend as K


def Poisson(y_true, y_pred):
    # "y_pred" predicts the Poisson "lambda".  Note that the standard
    # tf.keras.losses.Poisson() corresponds to predicting exp(lambda).
    y_pred = convert_to_tensor(y_pred)
    y_true = tf_cast(y_true, y_pred.dtype)
    return -K.mean(
        y_true*y_pred
        - exp(y_pred),
        axis=-1
    )


def NegativeBinomial(y_true, y_pred):
    y_pred = convert_to_tensor(y_pred)
    # Extract distribution parameters
    r, p = y_pred
    y_true = tf_cast(y_true, y_pred.dtype)
    return -K.mean(
        lgamma(y_true+r)
        + y_true*log(1-p)
        - lgamma(r)
        + r*log(p),
        axis=-1
    )

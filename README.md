# Mirabolic
Tools for statistical modeling and analysis, written by [Mirabolic](https://www.mirabolic.net/).  These modules can be installed by running
```
pip install --upgrade mirabolic
```
and the source code can be found at https://github.com/Mirabolic/mirabolic

## CDF Confidence Intervals

When exploring data, it can be very helpful to plot observations as a [CDF](https://en.wikipedia.org/wiki/Cumulative_distribution_function).  Producing a CDF essentially amounts to sorting the observed data from smallest to largest.  We can treat[^iid] the value in the middle of the sorted list as approximately the median, the value 90% of the way up the list is near the 90th percentile, and so forth.

[^iid]: We assume the data consists of i.i.d. draws from some unknown probability distribution.

When interpreting a CDF, or comparing two of them, one often wishes for something akin to a confidence interval.  How close is the middle value to the median?  Somewhat surprisingly, it is possible to compute the corresponding confidence intervals exactly.[^Beta]

[^Beta]: More precisely, suppose we draw a sample of n observations and consider the i-th smallest; if we are sampling from *any* continuous probability distribution, then the distribution of the corresponding quantile has a [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution), B(i, n-i+1).

For a single data point, the uncertainty around its quantile can be thought of as a confidence interval.  If we consider all the data points, then we refer to a *confidence band*.[^Credible]

[^Credible]: Because we have access to a prior distribution on quantiles, these are arguably *[credible intervals](https://en.wikipedia.org/wiki/Credible_interval)* and *credible bands*, rather than confidence intervals and bands.  We do not concern ourselves with this detail.

We provide a simple function for plotting CDFs with confidence bands; one invokes it by calling something like:
```
import mirabolic
import matplotlib.pyplot as plt

mirabolic.cdf_plot(data=[17.2, 5.1, 13, ...])
plt.show()
```

More examples can be found in (`mirabolic/cdf/sample_usage.py`)[https://github.com/Mirabolic/mirabolic/blob/main/mirabolic/cdf/sample_usage.py].

## Neural Nets for GLM regression

GLMs ([Generalized Linear Models](https://en.wikipedia.org/wiki/Generalized_linear_model)) are a relatively broad class of statistical model first popularlized in the 1970s.  These have grown popular in the actuarial literature as a method of predicting insurance claims costs and frequency.

With the appropriate loss function, GLMs can be expressed as neural nets.  These two techniques have traditionally been treated as distinct, but bridging the divide provides two advantages.

First, a vast amount of effort has been spent on optimizing and accelerating neural nets over the past several years (GPUs and TPUs, parallelization).  By expressing a GLM as a neural net, we can leverage this work.[^NN]

[^NN]: In terms of focus, [this chart](https://trends.google.com/trends/explore?geo=US&q=deep%20learning,actuarial%20science) suggests something of the explosion of interest in neural nets and deep learning relative to more traditional actuarial models.

Second, expressing a GLM as a neural net opens the possibility of extending the neural net before or after the GLM component.  For instance, suppose we build three subnets that each computed a single feature, and then feed the three outputs as inputs into the Poisson regression net.  This single larger network would allow the three subnets to engineer their individual features such that the loss function of the joint network was optimized.  This approach provides a straightforward way of performing non-linear feature engineering but retaining the explainability of a GLM.  This two-step approach may provide regulatory advantages, since US Departments of Insurance (DOIs) have been reluctant to approve end-to-end deep learning models.

We provide loss functions for several of the most commonly used GLMs.  Minimal code might look something like this:
```
import mirabolic.neural_glm as neural_glm
from keras.models import Sequential
import tf

model = Sequential()
# Actually design your neural net...
# model.add(...)
loss=neural_glm.Poisson_link_with_exposure
optimizer = tf.keras.optimizers.Adam()
model.compile(loss=neural_glm, optimizer=optimizer)
```

To illustrate this process in more detail, we provide code to perform [Poisson regression](https://en.wikipedia.org/wiki/Poisson_regression) and Negative Binomial regression using a neural net.  

To see the code in action, grab [the source code](https://github.com/Mirabolic/mirabolic) from GitHub, then [change to this directory](https://github.com/Mirabolic/mirabolic/tree/main/mirabolic/neural_glm), and run
```
python run_examples.py
```
This will generate Poisson-distributed data and corresponding features and then try to recover the "betas" (i.e., the linear coefficients of the GLM) using various models, outputting both the true and recovered values.


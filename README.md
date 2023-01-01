# Mirabolic
Tools for statistical modeling and analysis, written by [Mirabolic](https://www.mirabolic.net/).  These modules can be installed by running
```
pip install --upgrade mirabolic
```
and the source code can be found at https://github.com/Mirabolic/mirabolic

## CDF Confidence Intervals

When exploring data, it can be very helpful to plot observations as a [CDF](https://en.wikipedia.org/wiki/Cumulative_distribution_function).  Producing a CDF essentially amounts to sorting the observed data from smallest to largest.  We hope that the value in the middle of the sorted list is near the median, the value 90% of the way up the list is near the 90th percentile, and so forth.

When interpreting a CDF, or comparing two of them, one often wishes for something akin to a confidence interval.  Somewhat surprisingly, it is possible to compute these intervals exactly.[^Beta]

[^Beta]: More precisely, suppose we draw a sample of n observations and consider the i-th smallest; if we are sampling from *any* continuous probability distribution, then the distribution of the corresponding quantile has a [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution), B(i, n-i+1).

For a single data point, the uncertainty around its quantile can be thought of as a confidence interval.  If we consider all the data points, then we refer to a *confidence band*.[^Credible]

[^Credible]: Because we have access to a prior distribution on quantiles, these are arguably *(credible intervals)[https://en.wikipedia.org/wiki/Credible_interval]* and *credible bands*, rather than confidence intervals and bands.  We do not concern ourselves with this detail.

We provide a simple function for plotting CDFs with confidence bands; one invokes it by calling something like:
```
mirabolic.cdf_plot(data=[17.2, 5.1, 13, ...])
```

More examples can be found in (`mirabolic/cdf/sample_usage.py`)[https://github.com/Mirabolic/mirabolic/blob/main/mirabolic/cdf/sample_usage.py].

## Neural Nets for GLM regression

GLMs ([Generalized Linear Models](https://en.wikipedia.org/wiki/Generalized_linear_model)) are a relatively broad class of statistical model first popularlized in the 1970s.  These have grown popular in the actuarial literature as a method of predicting insurance claims costs and frequency.

With the appropriate loss function, GLMs can be formulated as types of neural nets.  To illustrate this, we perform [Poisson regression](https://en.wikipedia.org/wiki/Poisson_regression) in Keras using a nearly trivial network and a custom loss function.  Expressing a GLM as a neural net opens the possibility of extending the neural net before or after the GLM component.  For instance, suppose we build three subnets that each computed a single feature, and then feed the three outputs as inputs into the Poisson regression net.  This single larger network would allow the three subnets to engineer their individual features such that the loss function of the joint network was optimized.  This approach provides a straightforward way of performing non-linear feature engineering but retaining the explainability of a GLM.

To see the code in action, run
```
python sample_poisson.py
```
This will generate some Poisson-distributed data and corresponding features and then try to recover the "betas" (i.e., the linear coefficients of the GLM), outputting both the true and recovered values.

We also include the loss function required for negative binomial regression, which can be useful when modeling count data with higher variance.

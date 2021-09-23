# Mirabolic
Tools for statistical modeling and analysis.

## Neural Nets for GLM regression

GLMs ([Generalized Linear Models](https://en.wikipedia.org/wiki/Generalized_linear_model)) are a relatively broad class of statistical model first popularlized in the 1970s.  These have grown popular in the actuarial literature as a method of predicting insurance claims costs and frequency.

With the appropriate loss function, GLMs can be formulated as types of neural nets.  To illustrate this, we perform [Poisson regression](https://en.wikipedia.org/wiki/Poisson_regression) in Keras using a nearly trivial network and a custom loss function.  Expressing a GLM as a neural net opens the possibility of extending the neural net before or after the GLM component.  For instance, suppose we build a three subnets that each computed a single feature, and then feed the three outputs as inputs into the Poisson regression net.  This single larger network would allow the three subnets to engineer their individual features such that the loss function of the joint network was optimized.  This approach provides a straightforward way of performing non-linear feature engineering but retaining the explainability of a GLM.

To see the code in action, run
```
python sample_poisson.py
```
This will generate some Poisson-distributed data and corresponding features and then try to recover the "betas" (i.e., the linear coefficients of the GLM), outputting both the true and recovered values.

We also include the loss function required for negative binomial regression, which can be useful when modeling count data with higher variance.

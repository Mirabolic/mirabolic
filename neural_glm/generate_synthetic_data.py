# Make some synthetic data from, e.g., Poisson and Negative Binomial
# distributions.

import numpy as np


class synthetic_data():
    # Note: "features" = "exogenous  variables"
    #       "labels"   = "endogenous variables"

    def __init__(self,
                 N=100000,
                 num_features=8,
                 true_betas=None,
                 distribution='Poisson',
                 train_fraction=.6,
                 test_fraction=.3,
                 valid_fraction=.1,
                 exposure=False,
                 ):
        self.N = N
        self.num_features = num_features
        assert distribution in ['Poisson', 'Negative Binomial']
        self.distribution = distribution
        self.fractions = {'train': train_fraction,
                          'test': test_fraction,
                          'valid': valid_fraction}

        self.has_exposure = exposure
        if true_betas is None:
            self.make_betas()
        else:
            self.true_betas = true_betas

        self.make_features()
        self.make_params()
        self.make_exposure()
        self.make_data()
        self.make_train_test_valid_split()

    def make_betas(self):
        self.true_betas = {}
        if self.distribution == 'Poisson':
            b = np.power(np.random.randn(self.num_features), 3)
            b /= np.max(np.abs(b))
            self.true_betas['lambda'] = b
        elif self.distribution == 'Negative Binomial':
            self.true_betas['n'] = np.random.exponential(
                size=self.num_features)
            self.true_betas['p'] = np.random.uniform(
                size=self.num_features)

    def make_features(self):
        self.features = np.random.randn(self.N, self.num_features)

    def make_params(self):
        self.params = {}
        if self.distribution == 'Poisson':
            self.params['lambda'] = np.exp(
                self.features @ self.true_betas['lambda'])
        elif self.distribution == 'Negative Binomial':
            # Note: these are non-standard link functions;
            # canonical is "g(mu)=log[mu/ [n*(1+mu/n)]"
            # where mu=np/(1-p)
            self.params['n'] = np.exp(self.features @ self.true_betas['n'])
            self.params['p'] = 1 / (
                1 + np.exp(-(self.features @ self.true_betas['p'])))

    def make_exposure(self):
        if self.has_exposure:
            self.exposure = (
                9 * np.random.randint(2, size=self.N).astype(float) + 1)
            self.exposure *= 0.1
            # Useful for debugging...
            # self.exposure = np.ones(self.N)

    def make_data(self):
        if self.distribution == 'Poisson':
            lam = self.params['lambda']
            if self.has_exposure:
                lam *= self.exposure
            self.labels = np.random.poisson(
                lam=lam, size=self.N)
        elif self.distribution == 'Negative Binomial':
            n = self.params['n']
            if self.has_exposure:
                n *= self.exposure
            self.labels = np.random.negative_binomial(
                n=n,
                p=self.params['p'],
                size=self.N)
        if self.has_exposure:
            self.labels = np.vstack([self.labels, self.exposure]).T

    def make_train_test_valid_split(self):
        self.features_split = {}
        self.labels_split = {}
        index = 0
        sum_fractions = 0
        for f in self.fractions:
            count = round(self.fractions[f] * self.N)
            self.features_split[f] = self.features[index: index+count]
            self.labels_split[f] = self.labels[index: index+count]
            index += count
            sum_fractions += self.fractions[f]
        assert index <= self.N
        assert sum_fractions <= 1

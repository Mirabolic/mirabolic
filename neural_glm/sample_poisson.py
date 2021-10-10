# Sample code for performing a Poisson regression using Keras

import generate_synthetic_data
import basic_glm_nn
import numpy as np

##########################
# What to test?
##########################
distribution = 'Poisson'
# distribution = 'Negative Binomial'
exposure = False  # Does exposure differ between observations?

##########################
# Make some synthetic data
##########################
synthesized = generate_synthetic_data.synthetic_data(
    N=100000,  # How many data points?
    distribution=distribution,
    exposure=exposure,
)
features_split = synthesized.features_split
labels_split = synthesized.labels_split

##########################
# Make a neural net
##########################
results = basic_glm_nn.build_and_train_basic_glm(
    loss=distribution,
    x_train=features_split['train'], y_train=labels_split['train'],
    x_test=features_split['test'], y_test=labels_split['test'],
    x_valid=features_split['valid'], y_valid=labels_split['valid'],
    exposure=exposure,
)

if distribution == 'Poisson':
    print('Betas:  True       Recovered')
    for i in range(len(synthesized.true_betas['lambda'])):
        true_beta = synthesized.true_betas['lambda'][i]
        recovered_beta = results['betas'][i, 0]
        print(7*' ', f'{true_beta:8.5f}',
              ' ', f'{recovered_beta:8.5f}')
    print('     Constant: %.5f' % results['beta_constant'])

elif distribution == 'Negative Binomial':
    print('Betas:')
    for i, key in enumerate(['n', 'p']):
        print('====== %s ======' % key)
        true_betas = synthesized.true_betas[key]
        recovered_betas = results['betas'][:, i]
        for i in range(len(true_betas)):
            true_beta = true_betas[i]
            recovered_beta = recovered_betas[i]
            # Convert link function
            if key == 'n':
                recovered_beta = np.exp(recovered_beta)
            elif key == 'p':
                recovered_beta = 1/(1+np.exp(recovered_beta))
            print(7*' ', f'{true_beta:8.5f}',
                  ' ', f'{recovered_beta:8.5f}')

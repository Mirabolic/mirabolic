# Sample code for performing a Poisson regression using Keras

import generate_synthetic_data
import basic_glm_nn


##########################
# Make some synthetic data
##########################
synthesized = generate_synthetic_data.synthetic_data()
features_split = synthesized.features_split
labels_split = synthesized.labels_split

##########################
# Make a neural net
##########################
results = basic_glm_nn.build_and_train_basic_glm(
    loss='Poisson',
    x_train=features_split['train'], y_train=labels_split['train'],
    x_test=features_split['test'], y_test=labels_split['test'],
    x_valid=features_split['valid'], y_valid=labels_split['valid'],
)

print('Betas:  True       Recovered')
for i in range(len(synthesized.true_betas['lambda'])):
    true_beta = synthesized.true_betas['lambda'][i]
    recovered_beta = results['betas'][i, 0]
    print(7*' ', f'{true_beta:8.5f}',
          ' ', f'{recovered_beta:8.5f}')

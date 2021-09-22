# Simple neural net implementing a GLM

import actuarial_loss_functions
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense


def basic_glm_model(num_features=None,
                    name=None,
                    kernel_regularizer=None,
                    optimizer=None,
                    loss=None,
                    ):
    if optimizer is None:
        optimizer = 'adam'

    if (loss == 'Poisson') or loss is None:
        loss = actuarial_loss_functions.Poisson
    elif loss == 'Negative Binomial':
        loss = actuarial_loss_functions.NegativeBinomial

    output_dim = 1  # Only 1 output, because regression
    model = Sequential()
    model.add(Dense(
        output_dim,
        input_dim=num_features,
        activation='linear',
        kernel_regularizer=kernel_regularizer,
        name='betas',
    ))
    model.compile(loss=loss, optimizer=optimizer)

    return(model)


def build_and_train_basic_glm(
    loss=None,
    model=None,
    min_stopping_delta=.000001,
    x_train=None, y_train=None,
    x_valid=None, y_valid=None,
    x_test=None, y_test=None,
    batch_size=256,
    epochs=40,
    verbose=0,
):
    num_features = np.shape(x_train)[1]

    if model is None:
        model = basic_glm_model(loss=loss, num_features=num_features)
    if (x_valid is not None) and (y_valid is not None):
        validation_data = (x_valid, y_valid)
        callbacks = [keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=min_stopping_delta)]
    else:
        validation_data = None
        callbacks = None
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        validation_data=validation_data,
                        callbacks=callbacks,
                        )
    if (x_test is not None) and (y_test is not None):
        score = model.evaluate(x_test, y_test, verbose=0)
    else:
        score = None

    betas = model.get_layer(name='betas').get_weights()[0]

    results = {}
    results['model'] = model
    results['score'] = score
    results['history'] = history
    results['betas'] = betas

    return(results)

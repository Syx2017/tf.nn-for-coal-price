from __future__ import print_function
from preprocess import dataprepare
import math
import os
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.3f}'.format

coal_price_dataframe = pd.read_csv('C:\\Users\\Syx\\Desktop\\ML\\SVM3.csv')
print(coal_price_dataframe.describe())
featureList = coal_price_dataframe.columns
print(featureList)
# preprocess data
month_coal_price_dataframe = pd.DataFrame()
dataprepare('C:\\Users\\Syx\\Desktop\\ML\\SVM3.csv', 'day', 'CCI5000', dataframe=month_coal_price_dataframe)
dataprepare('C:\\Users\\Syx\\Desktop\\ML\\SVM3.csv', 'day', 'CCI3800', dataframe=month_coal_price_dataframe)
dataprepare('C:\\Users\\Syx\\Desktop\\ML\\SVM3.csv', 'day', 'travelcost', dataframe=month_coal_price_dataframe)
dataprepare('C:\\Users\\Syx\\Desktop\\ML\\SVM3.csv', 'day', 'Qhstorage', dataframe=month_coal_price_dataframe)
dataprepare('C:\\Users\\Syx\\Desktop\\ML\\SVM3.csv', 'day', 'seastorage', dataframe=month_coal_price_dataframe)
dataprepare('C:\\Users\\Syx\\Desktop\\ML\\SVM3.csv', 'week', 'CCTDQH5500', dataframe=month_coal_price_dataframe)
dataprepare('C:\\Users\\Syx\\Desktop\\ML\\SVM3.csv', 'month', 'SHyear5500', dataframe=month_coal_price_dataframe)
dataprepare('C:\\Users\\Syx\\Desktop\\ML\\SVM3.csv', 'month', 'SHmonth5500', dataframe=month_coal_price_dataframe)
dataprepare('C:\\Users\\Syx\\Desktop\\ML\\SVM3.csv', 'month', 'overallstorage', dataframe=month_coal_price_dataframe)
dataprepare('C:\\Users\\Syx\\Desktop\\ML\\SVM3.csv', 'month', 'eleconsume', dataframe=month_coal_price_dataframe)
#month_coal_price_dataframe['eleconsume'] = month_coal_price_dataframe['eleconsume']/10 #standarlize
print(month_coal_price_dataframe)

'''
coal_price = np.array(month_coal_price_dataframe['CCTDQH5500']).reshape(-1, 1)
x_scaler = StandardScaler()
x_scaler.fit(coal_price)
coal_price = x_scaler.transform(coal_price).reshape(-1,)
month_coal_price_dataframe['CCTDQH5500'] = coal_price
'''
month_coal_price_dataframe = month_coal_price_dataframe.reindex(
    np.random.permutation(month_coal_price_dataframe.index))
print(month_coal_price_dataframe.describe())

selected_features = month_coal_price_dataframe[['CCI5000', 'CCI3800', 'travelcost',
                                                'Qhstorage', 'SHyear5500', 'SHmonth5500', 'eleconsume']]
my_targets = month_coal_price_dataframe['CCTDQH5500']

training_examples = selected_features.head(28)
training_targets = my_targets.head(28)
validation_examples = selected_features.tail(8)
validation_targets = my_targets.tail(8)

print(training_examples)
print(training_targets)

def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_nn_regression_model(
        my_optimizer,
        steps,
        batch_size,
        hidden_units,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets,):
    """Trains a neural network regression model.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Args:
      my_optimizer: An instance of `tf.train.Optimizer`, the optimizer to use.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      hidden_units: A `list` of int values, specifying the number of neurons in each layer.
      training_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for training.
      training_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for training.
      validation_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for validation.
      validation_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for validation.

    Returns:
      A tuple `(estimator, training_losses, validation_losses)`:
        estimator: the trained `DNNRegressor` object.
        training_losses: a `list` containing the training loss values taken during training.
        validation_losses: a `list` containing the validation loss values taken during training.
    """

    periods = 10
    steps_per_period = steps / periods

    # Create a DNNRegressor object.
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(
        feature_columns=construct_feature_columns(training_examples),
        hidden_units=hidden_units,
        optimizer=my_optimizer,
        activation_fn=tf.nn.relu,
    )

    # Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets,
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets,
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                      validation_targets,
                                                      num_epochs=1,
                                                      shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    training_error_rate = []
    validation_error_rate =[]
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        dnn_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))
        training_mean_error_rate = np.mean(abs(training_predictions-training_targets)/training_targets)
        validation_mean_error_rate = np.mean(abs(validation_predictions-validation_targets)/validation_targets)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
        training_error_rate.append(training_mean_error_rate)
        validation_error_rate.append(validation_mean_error_rate)
    print('training error rate: ', training_error_rate[-1])
    print('validation error rate: ', validation_error_rate[-1])
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()

    ground_truth = np.append(np.array(training_targets), np.array(validation_targets))
    prediction = np.append(training_predictions, validation_predictions)

    plt.subplot(1, 2, 2)
    plt.ylabel("coal price")
    plt.xlabel("data point")
    plt.title("ground truth vs. prediction")
    plt.tight_layout()
    plt.plot(ground_truth, label="ground truth")
    plt.plot(prediction, label="prediction")
    plt.legend(loc='upper right')
    plt.show()

    print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
    print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

    return dnn_regressor, training_rmse, validation_rmse

_ = train_nn_regression_model(
    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001),
    steps=10000,
    batch_size=10,
    hidden_units=[100, 100],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets,
    )
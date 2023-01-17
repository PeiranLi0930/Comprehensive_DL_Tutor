import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split

def test_loop(test_cases):
    
    success = 0
    fails = 0
    
    for test_case in test_cases:
        try:
            assert test_case["result"] == test_case["expected"]
            success += 1
    
        except:
            fails += 1
            print(f'{test_case["name"]}: {test_case["error_message"]}\nExpected: {test_case["expected"]}\nResult: {test_case["result"]}\nPlease open utils.py if you want to see the unit test here.\n')

    if fails == 0:
        print("\033[92m All public tests passed")

    else:
        print('\033[92m', success," Tests passed")
        print('\033[91m', fails, " Tests failed")
        raise Exception("Please check the error messages above.")
        
def test_white_df(white_df):
    
    test_cases = [
        {
            "name": "type_check",
            "result": type(white_df.is_red[0]),
            "expected": np.int64,
            "error_message": f'white_df.is_red has an incorrect type.'
        },
        {
            "name": "output_check",
            "result": white_df.is_red[0],
            "expected": 0,
            "error_message": "white_df.is_red is not set correctly"
        },
        {
            "name": "len_check",
            "result": len(white_df),
            "expected": 3961,
            "error_message": "Number of rows is incorrect. Please drop duplicates."
        }
    ]
    
    test_loop(test_cases)
    
def test_red_df(red_df):
    
    test_cases = [
        {
            "name": "type_check",
            "result": type(red_df.is_red[0]),
            "expected": np.int64,
            "error_message": f'red_df.is_red has an incorrect type.'
        },
        {
            "name": "output_check",
            "result": red_df.is_red[0],
            "expected": 1,
            "error_message": "red_df.is_red is not set correctly"
        },
        {
            "name": "len_check",
            "result": len(red_df),
            "expected": 1359,
            "error_message": "Number of rows is incorrect. Please drop duplicates."
        }
    ]
    
    test_loop(test_cases)
    
def test_df_drop(df):
    
    test_cases = [
        {
            "name": "df.alcohol[0]_check",
            "result": df.alcohol[0],
            "expected": 9.4,
            "error_message": f'Value is not as expected. Please check quality interval.'
        },
        {
            "name": "df.alcohol[100]_check",
            "result": df.alcohol[100],
            "expected": 10.9,
            "error_message": f'Value is not as expected. Please check quality interval.'
        }
    ]
    
    test_loop(test_cases)

def test_data_sizes(train_size, test_size, val_size):
    
    test_cases = [
        {
            "name": "train_test_size_check",
            "result": train_size > test_size,
            "expected": True,
            "error_message": f'train.size is too small. Please check your code.'
        },
        {
            "name": "train_val_size_check",
            "result": train_size > val_size,
            "expected": True,
            "error_message": f'train.size is too small. Please check your code.'
        },
        {
            "name": "test_val_size_check",
            "result": test_size > val_size,
            "expected": True,
            "error_message": f'test.size is too small. Please check your code.'
        }
    ]
    
    test_loop(test_cases)

def test_format_output(df, train_Y, val_Y, test_Y):
    
    train, test = train_test_split(df, test_size=0.2, random_state=1)
    train, val = train_test_split(train, test_size=0.2, random_state=1)
    
    test_cases = [
        {
            "name": "train_Y[0]_check",
            "result": np.all(train_Y[0] == np.array(train.quality)),
            "expected": True,
            "error_message": f'train_Y[0] is not equal to train.quality. Please check your code.'
        },
        {
            "name": "train_Y[1]_check",
            "result": np.all(train_Y[1] == np.array(train.is_red)),
            "expected": True,
            "error_message": f'train_Y[1] is not equal to train.is_red. Please check your code.'
        },
        {
            "name": "val_Y[0]_check",
            "result": np.all(val_Y[0] == np.array(val.quality)),
            "expected": True,
            "error_message": f'train_Y[0] is not equal to val.quality. Please check your code.'
        },
        {
            "name": "val_Y[1]_check",
            "result": np.all(val_Y[1] == np.array(val.is_red)),
            "expected": True,
            "error_message": f'train_Y[1] is not equal to val.is_red. Please check your code.'
        },
        {
            "name": "test_Y[0]_check",
            "result": np.all(test_Y[0] == np.array(test.quality)),
            "expected": True,
            "error_message": f'test_Y[0] is not equal to test.quality. Please check your code.'
        },
        {
            "name": "test_Y[1]_check",
            "result": np.all(test_Y[1] == np.array(test.is_red)),
            "expected": True,
            "error_message": f'test_Y[1] is not equal to test.is_red. Please check your code.'
        }
    ]
    
    test_loop(test_cases)

def test_norm(norm_train_X, norm_val_X, norm_test_X, train, val, test):
    
    from pandas.core.frame import DataFrame
    
    test_cases = [
        {
            "name": "norm_train_X_type_check",
            "result": type(norm_train_X),
            "expected": DataFrame,
            "error_message": f'norm_train_X has an incorrect type.'
        },
        {
            "name": "norm_val_X_type_check",
            "result": type(norm_val_X),
            "expected": DataFrame,
            "error_message": f'norm_val_X has an incorrect type.'
        },
        {
            "name": "norm_test_X_type_check",
            "result": type(norm_test_X),
            "expected": DataFrame,
            "error_message": f'norm_test_X has an incorrect type.'
        },
        {
            "name": "norm_train_X_length_check",
            "result": len(norm_train_X),
            "expected": len(train),
            "error_message": f'norm_train_X has an incorrect length.'
        },
        {
            "name": "norm_val_X_length_check",
            "result": len(norm_val_X),
            "expected": len(val),
            "error_message": f'norm_val_X has an incorrect length.'
        },
        {
            "name": "norm_test_X_length_check",
            "result": len(norm_test_X),
            "expected": len(test),
            "error_message": f'norm_test_X has an incorrect length.'
        },
    ]
    
    test_loop(test_cases)

def test_base_model(base_model):
    
    test_inputs = tf.keras.layers.Input(shape=(11,))
    test_output = base_model(test_inputs)
    test_model = Model(inputs=test_inputs, outputs=test_output)

    test_cases = [
        {
            "name": "return_type_check",
            "result": type(test_output),
            "expected": tf.Tensor,
            "error_message": 'Return type is incorrect. Please check your code.'
        },
        {
            "name": "return_shape_check",
            "result": str(test_output.shape),
            "expected": '(None, 128)',
            "error_message": 'Return shape is incorrect. Please check your code.'
        },
        {
            "name": "tensor_dtype_check",
            "result": str(test_output.dtype),
            "expected": "<dtype: 'float32'>",
            "error_message": 'model dtype is incorrect. Please check your code.'
        },
        {
            "name": "base_model_num_layers_check",
            "result": len(test_model.layers),
            "expected": 3,
            "error_message": 'There are too many layers. Please check your code.'
        },
        {
            "name": "base_model_layer1_check",
            "result": type(test_model.layers[-2]),
            "expected": Dense,
            "error_message": 'First layer type is incorrect. Please check your code.'
        },
        {
            "name": "base_model_layer2_check",
            "result": type(test_model.layers[-1]),
            "expected": Dense,
            "error_message": 'Output layer type is incorrect. Please check your code.'
        },
    ]
    
    test_loop(test_cases)

def test_final_model(final_model):
    
    test_inputs = tf.keras.layers.Input(shape=(11,))
    test_output = final_model(test_inputs)

    test_cases = [
        {
            "name": "return_type_check",
            "result": type(test_output),
            "expected": tf.keras.Model,
            "error_message": 'Return type is incorrect. Please check your code.'
        },
        {
            "name": "layer_3_activation_check",
            "result": test_output.layers[4].activation,
            "expected": tf.keras.activations.sigmoid,
            "error_message": 'wine_quality layer has an incorrect activation. Please check your code.'
        },
    ]
    
    test_loop(test_cases)
    
def test_model_compile(model):

    from tensorflow.python.keras.metrics import MeanMetricWrapper

    test_cases = [
        {
            "name": "metrics_0_check",
            "result": type(model.metrics[0]),
            "expected": tf.keras.metrics.RootMeanSquaredError,
            "error_message": 'wine quality metrics is incorrect. Please check your code.'
        },
        {
            "name": "metrics_1_check",
            "result": (model.metrics[1].name == 'wine_type_accuracy') or 
                      (model.metrics[1].name == 'wine_type_binary_accuracy'),
            "expected": True,
            "error_message": f'wine type metrics: {model.metrics[1].name} is incorrect. Please check your code.'
        },
        {
            "name": "wine_type_loss_check",
            "result": (model.loss['wine_type'] == 'binary_crossentropy') or 
                      (model.loss['wine_type'].name == 'binary_crossentropy') or 
                      (str(model.loss['wine_type']).split()[1] == 'binary_crossentropy'),
            "expected": True,
            "error_message": f'wine type loss: {model.loss["wine_type"]} is incorrect. Please check your code.'
        },
        {
            "name": "wine_quality_loss_check",
            "result": (model.loss['wine_quality'] in ['mse', 'mean_squared_error']) or 
                      (str(model.loss['wine_quality']).split()[1] == 'mean_squared_error') or 
                      (model.loss['wine_quality'].name == 'mean_squared_error'),
            "expected": True,
            "error_message": f'wine quality loss: {model.loss["wine_type"]} is incorrect. Please check your code.'
        },
    ]
    
    test_loop(test_cases)
    
def test_history(history):

    vars_history = vars(history)
    
    test_cases = [
        {
            "name": "type_check",
            "result": type(history),
            "expected": tf.keras.callbacks.History,
            "error_message": 'history type is incorrect. Please check model.fit().'
        },
        {
            "name": "params_samples_check",
            "result": vars_history['params']['samples'],
            "expected": 3155,
            "error_message": 'Training samples is incorrect. Please check arguments to model.fit().'
        },
        {
            "name": "params_do_validation_check",
            "result": vars_history['params']['do_validation'],
            "expected": True,
            "error_message": 'No validation data is present. Please check arguments to model.fit().'
        },
    ]
    
    test_loop(test_cases)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


















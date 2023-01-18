import tensorflow as tf
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import numpy as np

def test_loop(test_cases):
    
    success = 0
    fails = 0
    
    for test_case in test_cases:
        try:
            assert test_case["result"] == test_case["expected"]
            success += 1
    
        except:
            fails += 1
            print(f'{test_case["name"]}: {test_case["error_message"]}\nExpected: {test_case["expected"]}\nResult: {test_case["result"]}\n')

    if fails == 0:
        print("\033[92m All public tests passed")

    else:
        print('\033[92m', success," Tests passed")
        print('\033[91m', fails, " Tests failed")
        raise Exception("Please check the error messages above.")

        
def test_simple_quadratic(SimpleQuadratic):

    expected_units = 128
    expected_activation_function = tf.keras.activations.relu
    expected_activation_string = 'relu'
    shape_0 = 8
    shape_1 = 2
    
    test_layer = SimpleQuadratic(units=expected_units, activation=expected_activation_string)
    
    test_layer.build((shape_0, shape_1))

    test_inputs = tf.random.uniform((shape_0, shape_1))
    
    test_call_value = test_layer.call(test_inputs)
    
    a_type = type(test_layer.a)
    b_type = type(test_layer.b)
    c_type = type(test_layer.c)
    
    test_layer_forced_weights = SimpleQuadratic(units=1, activation=None)
    test_layer_forced_weights.a = tf.constant([2.0], dtype='float32', shape=(1,1))
    test_layer_forced_weights.b = tf.constant([2.0], dtype='float32', shape=(1,1))
    test_layer_forced_weights.c = tf.constant([2.0], dtype='float32', shape=(1,1))
    test_layer_forced_weights_inputs = tf.constant([4.0], dtype='float32', shape=(1,1))
    test_layer_forced_weights_expected_output = 42.0
    
    test_cases = [
        {
            "name": "units_check",
            "result": test_layer.units,
            "expected": expected_units,
            "error_message": f'Incorrect number of units.'
        },
        {
            "name": "activations_check",
            "result": test_layer.activation,
            "expected": tf.keras.activations.relu,
            "error_message": "Got different activation function."
        },
        {
            "name": "a_type_check",
            "result": a_type,
            "expected": ResourceVariable,
            "error_message": f'State variable a is of different type. Expected ResourceVariable but got {a_type}'
        },
        {
            "name": "b_type_check",
            "result": b_type,
            "expected": ResourceVariable,
            "error_message": f'State variable b is of different type. Expected ResourceVariable but got {b_type}'
        },
        {
            "name": "c_type_check",
            "result": c_type,
            "expected": ResourceVariable,
            "error_message": f'State variable c is of different type. Expected ResourceVariable but got {c_type}'
        },
        {
            "name": "a_initializer_check",
            "result": test_layer.a.numpy().sum() != 0,
            "expected": True,
            "error_message": f'State variable a is not initialized randomly. Please check initializer used.'
        },
        {
            "name": "b_initializer_check",
            "result": test_layer.b.numpy().sum() != 0,
            "expected": True,
            "error_message": f'State variable b is not initialized randomly. Please check initializer used.'
        },
        {
            "name": "c_initializer_check",
            "result": test_layer.c.numpy().sum() == 0,
            "expected": True,
            "error_message": f'State variable c is not initialized to zeroes. Please check initializer used.'
        },
        {
            "name": "output_check",
            "result": test_layer_forced_weights.call(test_layer_forced_weights_inputs).numpy()[0][0],
            "expected": test_layer_forced_weights_expected_output,
            "error_message": f'Expected output is incorrect. Please check operations in the call() method.'
        }
    ]
    
    test_loop(test_cases)

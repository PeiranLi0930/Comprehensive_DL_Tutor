import numpy as np
from dlai_tools.testing_utils import single_test, multiple_test


def initialize_parameters_zeros_test(target):
    layer_dims = [3,2,1]
    expected_output = {'W1': np.array([[0., 0., 0.],
        [0., 0., 0.]]),
 'b1': np.array([[0.],
        [0.]]),
 'W2': np.array([[0., 0.]]),
 'b2': np.array([[0.]])}
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]
    
    multiple_test(test_cases, target)
    
def initialize_parameters_random_test(target):
    layer_dims = [3,2,1]
    expected_output = {'W1': np.array([[ 17.88628473,   4.36509851,   0.96497468],
        [-18.63492703,  -2.77388203,  -3.54758979]]),
 'b1': np.array([[0.],
        [0.]]),
 'W2': np.array([[-0.82741481, -6.27000677]]),
 'b2': np.array([[0.]])}
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]
    
    multiple_test(test_cases, target)
    
def initialize_parameters_he_test(target):
    
    layer_dims = [3, 1, 2]
    expected_output = {'W1': np.array([[1.46040903, 0.3564088 , 0.07878985]]), 
                       'b1': np.array([[0.]]), 
                       'W2': np.array([[-2.63537665], [-0.39228616]]), 
                       'b2': np.array([[0.],[0.]])}
    
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [layer_dims],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]
    
    multiple_test(test_cases, target)
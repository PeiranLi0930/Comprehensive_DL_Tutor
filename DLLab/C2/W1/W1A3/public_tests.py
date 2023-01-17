import numpy as np
from dlai_tools.testing_utils import single_test, multiple_test

def forward_propagation_test(target):
    x, theta = 2, 4
    expected_output = 8
    test_cases = [
        {
            "name": "equation_output_check",
            "input": [x, theta],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    single_test(test_cases, target) 
        
def backward_propagation_test(target):
    x, theta = 3, 4
    expected_output = 3
    test_cases = [
        {
            "name": "equation_output_check",
            "input": [x, theta],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    single_test(test_cases, target)

def gradient_check_test(target):
    x, theta = 3, 4
    expected_output = 7.814075313343006e-11
    test_cases = [
        {
            "name": "equation_output_check",
            "input": [x, theta],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    single_test(test_cases, target)
    
    
def predict_test(target):
    np.random.seed(1)
    X = np.random.randn(2, 3)
    parameters = {'W1': np.array([[-0.00615039,  0.0169021 ],
        [-0.02311792,  0.03137121],
        [-0.0169217 , -0.01752545],
        [ 0.00935436, -0.05018221]]),
     'W2': np.array([[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]),
     'b1': np.array([[ -8.97523455e-07],
        [  8.15562092e-06],
        [  6.04810633e-07],
        [ -2.54560700e-06]]),
     'b2': np.array([[  9.14954378e-05]])}
    expected_output = np.array([[True, False, True]])

    test_cases = [
        {
            "name":"datatype_check",
            "input": [parameters, X],
            "expected": expected_output,
            "error":"Data type mismatch"
        },
        {
            "name": "shape_check",
            "input": [parameters, X],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters, X],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    single_test(test_cases, target)

    

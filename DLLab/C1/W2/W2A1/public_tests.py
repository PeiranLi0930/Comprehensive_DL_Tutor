import numpy as np
from test_utils import test

        
def basic_sigmoid_test(target):
    x = 1
    expected_output = 0.7310585786300049
    test_cases = [
        {
            "name": "datatype_check",
            "input": [x],
            "expected": float,
            "error": "Datatype mismatch."
        },
        {
            "name": "equation_output_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    test(test_cases, target)
         
def sigmoid_test(target):
    x = np.array([1, 2, 3])
    expected_output = np.array([0.73105858,
                                0.88079708,
                                0.95257413])
    test_cases = [
        {
            "name":"datatype_check",
            "input": [x],
            "expected": np.ndarray,
            "error":"Datatype mismatch."
        },
        {
            "name": "shape_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    test(test_cases, target)
    
            
        
def sigmoid_derivative_test(target):
    x = np.array([1, 2, 3])
    expected_output = np.array([0.19661193,
                                0.10499359,
                                0.04517666])
    test_cases = [
        {
            "name":"datatype_check",
            "input": [x],
            "expected": np.ndarray,
            "error":"The function should return a numpy array."
        },
        {
            "name": "shape_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong shape."
        },
        {
            "name": "equation_output_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong output."
        }
    ]
    
    test(test_cases, target)

def image2vector_test(target):
    image = np.array([[[ 0.67826139,  0.29380381],
                      [ 0.90714982,  0.52835647],
                      [ 0.4215251 ,  0.45017551]],

                     [[ 0.92814219,  0.96677647],
                      [ 0.85304703,  0.52351845],
                      [ 0.19981397,  0.27417313]],

                     [[ 0.60659855,  0.00533165],
                      [ 0.10820313,  0.49978937],
                      [ 0.34144279,  0.94630077]]])
    
    expected_output = np.array([[ 0.67826139],
                                [ 0.29380381],
                                [ 0.90714982],
                                [ 0.52835647],
                                [ 0.4215251 ],
                                [ 0.45017551],
                                [ 0.92814219],
                                [ 0.96677647],
                                [ 0.85304703],
                                [ 0.52351845],
                                [ 0.19981397],
                                [ 0.27417313],
                                [ 0.60659855],
                                [ 0.00533165],
                                [ 0.10820313],
                                [ 0.49978937],
                                [ 0.34144279],
                                [ 0.94630077]])
    test_cases = [
        {
            "name":"datatype_check",
            "input": [image],
            "expected": np.ndarray,
            "error":"The function should return a numpy array."
        },
        {
            "name": "shape_check",
            "input": [image],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [image],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    test(test_cases, target)

def normalizeRows_test(target):
    x = np.array([[0, 3, 4],
                  [1, 6, 4]])
    expected_output = np.array([[ 0., 0.6, 0.8 ],
                                [ 0.13736056, 0.82416338, 0.54944226]])
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [x],
            "expected": np.ndarray,
            "error":"The function should return a numpy array."
        },
        {
            "name": "shape_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    test(test_cases, target)       
        
def softmax_test(target):
    x = np.array([[9, 2, 5, 0, 0],
                  [7, 5, 0, 0 ,0]])
    expected_output = np.array([[ 9.80897665e-01, 8.94462891e-04,
                                 1.79657674e-02, 1.21052389e-04,
                                 1.21052389e-04],
                                
                                [ 8.78679856e-01, 1.18916387e-01,
                                 8.01252314e-04, 8.01252314e-04,
                                 8.01252314e-04]])
    test_cases = [
        {
            "name":"datatype_check",
            "input": [x],
            "expected": np.ndarray,
            "error":"The function should return a numpy array."
        },
        {
            "name": "shape_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [x],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    test(test_cases, target)

def L1_test(target):
    yhat = np.array([.9, 0.2, 0.1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    expected_output = 1.1
    test_cases = [
        {
            "name":"datatype_check",
            "input": [yhat, y],
            "expected": float,
            "error":"The function should return a float."
        },
        {
            "name": "equation_output_check",
            "input": [yhat, y],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    test(test_cases, target)
    
def L2_test(target):
    yhat = np.array([.9, 0.2, 0.1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    expected_output = 0.43
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [yhat, y],
            "expected": float,
            "error":"The function should return a float."
        },
        {
            "name": "equation_output_check",
            "input": [yhat, y],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    test(test_cases, target)


import numpy as np
from numpy import array
from testCases import *
from dlai_tools.testing_utils import single_test, multiple_test

         
def update_parameters_with_gd_test(target):
    parameters, grads, learning_rate = update_parameters_with_gd_test_case()
    learning_rate = 0.01

    expected_output = {'W1': np.array([[ 1.63535156, -0.62320365, -0.53718766],
                                [-1.07799357,  0.85639907, -2.29470142]]),
                       'b1': np.array([[ 1.74604067],
                                [-0.75184921]]),
                       'W2': np.array([[ 0.32171798, -0.25467393,  1.46902454],
                                [-2.05617317, -0.31554548, -0.3756023 ],
                                [ 1.1404819 , -1.09976462, -0.1612551 ]]),
                       'b2': np.array([[-0.88020257],
                                [ 0.02561572],
                                [ 0.57539477]])}

    params_up = target(parameters, grads, learning_rate)

    for key in params_up.keys():
        assert type(params_up[key]) == np.ndarray, f"Wrong type for {key}. We expected np.ndarray, but got {type(params_up[key])}"
        assert params_up[key].shape == parameters[key].shape, f"Wrong shape for {key}. {params_up[key].shape} != {parameters[key].shape}"
        assert np.allclose(params_up[key], expected_output[key]), f"Wrong values for {key}. Check the formulas. Expected: \n {expected_output[key]}"
    
    print("\033[92mAll test passed")
            
        
def random_mini_batches_test(target):
    np.random.seed(1)
    mini_batch_size = 2
    X = np.random.randn(5, 7)
    Y = np.random.randn(1, 7) < 0.5

    expected_output = [(np.array([[ 1.74481176, -0.52817175],
                                  [-0.38405435, -0.24937038],
                                  [-1.10061918, -0.17242821],
                                  [-0.93576943,  0.50249434],
                                  [-0.67124613, -0.69166075]]), 
                        np.array([[ True,  True]])), 
                       (np.array([[-0.61175641, -1.07296862],
                                  [ 0.3190391 ,  1.46210794],
                                  [-1.09989127, -0.87785842],
                                  [ 0.90159072,  0.90085595],
                                  [ 0.53035547, -0.39675353]]), 
                        np.array([[ True, False]])), 
                       (np.array([[ 1.62434536, -2.3015387 ],
                                  [-0.7612069 , -0.3224172 ],
                                  [ 1.13376944,  0.58281521],
                                  [ 1.14472371, -0.12289023],
                                  [-0.26788808, -0.84520564]]), 
                        np.array([[ True,  True]])), 
                       (np.array([[ 0.86540763],
                                  [-2.06014071],
                                  [ 0.04221375],
                                  [-0.68372786],
                                  [-0.6871727 ]]), 
                        np.array([[False]]))]
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [X, Y, mini_batch_size],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [X, Y, mini_batch_size],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [X, Y, mini_batch_size],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]
    
    multiple_test(test_cases, target)

def initialize_velocity_test(target):
    parameters = initialize_velocity_test_case()
    
    expected_output = {'dW1': np.array([[0., 0.],
                                        [0., 0.],
                                        [0., 0.]]), 'db1': np.array([[0.],
                                        [0.]]), 'dW2': np.array([[0., 0., 0.],
                                                                 [0., 0., 0.],
                                                                 [0., 0., 0.]]), 'db2': array([[0.],
                                                                                               [0.],
                                                                                               [0.]])}
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [parameters],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [parameters],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    multiple_test(test_cases, target)

def update_parameters_with_momentum_test(target):
    parameters, grads, v = update_parameters_with_momentum_test_case()
    beta = 0.9
    learning_rate = 0.01
    expected_parameters = {'W1': np.array([[ 1.62544598, -0.61290114, -0.52907334],
                                            [-1.07347112,  0.86450677, -2.30085497]]),
                 'b1': np.array([[ 1.74493465],
                        [-0.76027113]]),
                 'W2': np.array([[ 0.31930698, -0.24990073,  1.4627996 ],
                        [-2.05974396, -0.32173003, -0.38320915],
                        [ 1.13444069, -1.0998786 , -0.1713109 ]]),
                 'b2': np.array([[-0.87809283],
                        [ 0.04055394],
                        [ 0.58207317]])}
    expected_v = {'dW1': np.array([[-0.11006192,  0.11447237,  0.09015907],
                        [ 0.05024943,  0.09008559, -0.06837279]]),
                 'dW2': np.array([[-0.02678881,  0.05303555, -0.06916608],
                        [-0.03967535, -0.06871727, -0.08452056],
                        [-0.06712461, -0.00126646, -0.11173103]]),
                 'db1': np.array([[-0.01228902],
                        [-0.09357694]]),
                 'db2': np.array([[0.02344157],
                        [0.16598022],
                        [0.07420442]])}
    expected_output = (expected_parameters, expected_v)
    test_cases = [
        {
            "name":"datatype_check",
            "input": [parameters, grads, v, beta, learning_rate],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [parameters, grads, v, beta, learning_rate],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters, grads, v, beta, learning_rate],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    multiple_test(test_cases, target)    
        
def initialize_adam_test(target):
    parameters = initialize_adam_test_case()
    expected_v = {'dW1': np.array([[0., 0., 0.],
        [0., 0., 0.]]),
 'db1': np.array([[0.],
        [0.]]),
 'dW2': np.array([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]]),
 'db2': np.array([[0.],
        [0.],
        [0.]])}
    expected_s = {'dW1': np.array([[0., 0., 0.],
        [0., 0., 0.]]),
 'db1': np.array([[0.],
        [0.]]),
 'dW2': np.array([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]]),
 'db2': np.array([[0.],
        [0.],
        [0.]])}
    expected_output = (expected_v, expected_s)
    test_cases = [
        {
            "name":"datatype_check",
            "input": [parameters],
            "expected": expected_output,
            "error":"The function should return a numpy array."
        },
        {
            "name": "shape_check",
            "input": [parameters],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    multiple_test(test_cases, target)

def update_parameters_with_adam_test(target):
    parametersi, grads, vi, si, t, learning_rate, beta1, beta2, epsilon = update_parameters_with_adam_test_case()

    c1 = 1.0 / (1 - beta1**t)
    c2 = 1.0 / (1 - beta2**t)

    expected_v = {'dW1': np.array([-0.22012384,  0.22894474,  0.18031814]), 
                  'dW2': np.array([-0.05357762,  0.10607109, -0.13833215]), 
                  'db1': np.array([-0.02457805]), 
                  'db2': np.array([0.04688314])}
    expected_s = {'dW1': np.array([0.13567261, 0.14676395, 0.09104097]),
                  'dW2':np.array([8.03757060e-03, 3.15030152e-02, 5.35801947e-02]),
                  'db1':np.array([0.00169142]),
                  'db2':np.array([0.00615448])}
    expected_parameters = {'W1': np.array([ 1.63942428, -0.6268425,  -0.54320974]),
                  'W2':np.array([ 0.33356139, -0.26425199, 1.47707772]),
                  'b1':np.array([1.75854357]),
                  'b2':np.array([-0.89228024])}

    parameters, v, s, vc, sc  = target(parametersi, grads, vi, si, t, learning_rate, beta1, beta2, epsilon)

    for key in v.keys():
        assert type(v[key]) == np.ndarray, f"Wrong type for v['{key}']. Expected np.ndarray"
        assert v[key].shape == vi[key].shape, f"Wrong shape for  v['{key}']. The update must keep the dimensions of v inputs"
        assert np.allclose(v[key][0], expected_v[key]), f"Wrong values. Check you formulas for v['{key}']"
        #print(f"v[\"{key}\"]: \n {str(v[key][0])}")

    for key in vc.keys():
        assert type(vc[key]) == np.ndarray, f"Wrong type for v_corrected['{key}']. Expected np.ndarray"
        assert vc[key].shape == vi[key].shape, f"Wrong shape for  v_corrected['{key}']. The update must keep the dimensions of v inputs"
        assert np.allclose(vc[key][0], expected_v[key] * c1), f"Wrong values. Check you formulas for v_corrected['{key}']"
        #print(f"vc[\"{key}\"]: \n {str(vc[key])}")

    for key in s.keys():
        assert type(s[key]) == np.ndarray, f"Wrong type for s['{key}']. Expected np.ndarray"
        assert s[key].shape == si[key].shape, f"Wrong shape for  s['{key}']. The update must keep the dimensions of s inputs"
        assert np.allclose(s[key][0], expected_s[key]), f"Wrong values. Check you formulas for s['{key}']"
        #print(f"s[\"{key}\"]: \n {str(s[key])}")

    for key in sc.keys():
        assert type(sc[key]) == np.ndarray, f"Wrong type for s_corrected['{key}']. Expected np.ndarray"
        assert sc[key].shape == si[key].shape, f"Wrong shape for  s_corrected['{key}']. The update must keep the dimensions of s inputs"
        assert np.allclose(sc[key][0], expected_s[key] * c2), f"Wrong values. Check you formulas for s_corrected['{key}']"   
        # print(f"sc[\"{key}\"]: \n {str(sc[key])}")

    for key in parameters.keys():
        assert type(parameters[key]) == np.ndarray, f"Wrong type for parameters['{key}']. Expected np.ndarray"
        assert parameters[key].shape == parametersi[key].shape, f"Wrong shape for  parameters['{key}']. The update must keep the dimensions of parameters inputs"
        assert np.allclose(parameters[key][0], expected_parameters[key]), f"Wrong values. Check you formulas for parameters['{key}']"   
        #print(f"{key}: \n {str(parameters[key])}")

    print("\033[92mAll test passed")
    
def update_lr_test(target):
    learning_rate = 0.5
    epoch_num = 2
    decay_rate = 1
    expected_output = 0.16666666666666666
    
    output = target(learning_rate, epoch_num, decay_rate)
    
    assert np.isclose(output, expected_output), f"output: {output} expected: {expected_output}"
    print("\033[92mAll test passed")

def schedule_lr_decay_test(target):
    learning_rate = 0.5
    epoch_num_1 = 100
    epoch_num_2 = 10
    decay_rate = 1
    time_interval = 100
    expected_output_1 = 0.25
    expected_output_2 = 0.5
    
    output_1 = target(learning_rate, epoch_num_1, decay_rate, time_interval)
    output_2 = target(learning_rate, epoch_num_2, decay_rate, time_interval)


    assert np.isclose(output_1, expected_output_1),f"output: {output_1} expected: {expected_output_1}"
    assert np.isclose(output_2, expected_output_2),f"output: {output_2} expected: {expected_output_2}"
    
    learning_rate = 0.3
    epoch_num_1 = 1000
    epoch_num_2 = 100
    decay_rate = 0.25
    time_interval = 100
    expected_output_1 = 0.085714285
    expected_output_2 = 0.24

    output_1 = target(learning_rate, epoch_num_1, decay_rate, time_interval)
    output_2 = target(learning_rate, epoch_num_2, decay_rate, time_interval)


    assert np.isclose(output_1, expected_output_1),f"output: {output_1} expected: {expected_output_1}"
    assert np.isclose(output_2, expected_output_2),f"output: {output_2} expected: {expected_output_2}"

    print("\033[92mAll test passed")
    
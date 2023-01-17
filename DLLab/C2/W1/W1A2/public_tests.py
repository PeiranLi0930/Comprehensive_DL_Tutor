import numpy as np
from dlai_tools.testing_utils import single_test, multiple_test
        
def compute_cost_with_regularization_test(target):
    np.random.seed(1)
    Y = np.array([[1, 1, 0, 1, 0]])
    W1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    W2 = np.random.randn(3, 2)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    A3 = np.array([[ 0.40682402,  0.01629284,  0.16722898,  0.10118111,  0.40682402]])
    lambd = 0.1
    expected_output = np.float64(1.7864859451590758)
    test_cases = [
        {
            "name": "shape_check",
            "input": [A3, Y, parameters, lambd],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [A3, Y, parameters, lambd],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]
    
    single_test(test_cases, target)
    
def backward_propagation_with_regularization_test(target):
    np.random.seed(1)
    X = np.random.randn(3, 5)
    Y = np.array([[1, 1, 0, 1, 0]])
    cache = (np.array([[-1.52855314,  3.32524635,  2.13994541,  2.60700654, -0.75942115],
         [-1.98043538,  4.1600994 ,  0.79051021,  1.46493512, -0.45506242]]),
  np.array([[ 0.        ,  3.32524635,  2.13994541,  2.60700654,  0.        ],
         [ 0.        ,  4.1600994 ,  0.79051021,  1.46493512,  0.        ]]),
  np.array([[-1.09989127, -0.17242821, -0.87785842],
         [ 0.04221375,  0.58281521, -1.10061918]]),
  np.array([[ 1.14472371],
         [ 0.90159072]]),
  np.array([[ 0.53035547,  5.94892323,  2.31780174,  3.16005701,  0.53035547],
         [-0.69166075, -3.47645987, -2.25194702, -2.65416996, -0.69166075],
         [-0.39675353, -4.62285846, -2.61101729, -3.22874921, -0.39675353]]),
  np.array([[ 0.53035547,  5.94892323,  2.31780174,  3.16005701,  0.53035547],
         [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]]),
  np.array([[ 0.50249434,  0.90085595],
         [-0.68372786, -0.12289023],
         [-0.93576943, -0.26788808]]),
  np.array([[ 0.53035547],
         [-0.69166075],
         [-0.39675353]]),
  np.array([[-0.3771104 , -4.10060224, -1.60539468, -2.18416951, -0.3771104 ]]),
  np.array([[ 0.40682402,  0.01629284,  0.16722898,  0.10118111,  0.40682402]]),
  np.array([[-0.6871727 , -0.84520564, -0.67124613]]),
  np.array([[-0.0126646]]))
    lambd = 0.7
    
    expected_output = {'dZ3': np.array([[-0.59317598, -0.98370716,  0.16722898, -0.89881889,  0.40682402]]),
 'dW3': np.array([[-1.77691347, -0.11832879, -0.09397446]]),
 'db3': np.array([[-0.38032981]]),
 'dA2': np.array([[ 0.40761434,  0.67597671, -0.11491519,  0.6176438 , -0.27955836],
        [ 0.50135568,  0.83143484, -0.14134288,  0.7596868 , -0.34384996],
        [ 0.39816708,  0.66030962, -0.11225181,  0.6033287 , -0.27307905]]),
 'dZ2': np.array([[ 0.40761434,  0.67597671, -0.11491519,  0.6176438 , -0.27955836],
        [ 0.        ,  0.        , -0.        ,  0.        , -0.        ],
        [ 0.        ,  0.        , -0.        ,  0.        , -0.        ]]),
 'dW2': np.array([[ 0.79276486,  0.85133918],
        [-0.0957219 , -0.01720463],
        [-0.13100772, -0.03750433]]),
 'db2': np.array([[0.26135226],
        [0.        ],
        [0.        ]]),
 'dA1': np.array([[ 0.2048239 ,  0.33967447, -0.05774423,  0.31036252, -0.14047649],
        [ 0.3672018 ,  0.60895764, -0.10352203,  0.5564081 , -0.25184181]]),
 'dZ1': np.array([[ 0.        ,  0.33967447, -0.05774423,  0.31036252, -0.        ],
        [ 0.        ,  0.60895764, -0.10352203,  0.5564081 , -0.        ]]),
 'dW1': np.array([[-0.25604646,  0.12298827, -0.28297129],
        [-0.17706303,  0.34536094, -0.4410571 ]]),
 'db1': np.array([[0.11845855],
        [0.21236874]])}
    test_cases = [
        {
            "name": "shape_check",
            "input": [X, Y, cache, lambd],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [X, Y, cache, lambd],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]
    
    multiple_test(test_cases, target)
    
def forward_propagation_with_dropout_test(target):
    np.random.seed(1)
    X = np.random.randn(3, 5)
    W1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    W2 = np.random.randn(3, 2)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    
    A3 = np.array([[0.36974721, 0.00305176, 0.04565099, 0.49683389, 0.36974721]])
    cache = (np.array([[-1.52855314,  3.32524635,  2.13994541,  2.60700654, -0.75942115],
        [-1.98043538,  4.1600994 ,  0.79051021,  1.46493512, -0.45506242]]),
 np.array([[ True, False,  True,  True,  True],
        [ True,  True,  True,  True,  True]]),
 np.array([[0.        , 0.        , 3.05706487, 3.72429505, 0.        ],
        [0.        , 5.94299915, 1.1293003 , 2.09276446, 0.        ]]),
 np.array([[-1.09989127, -0.17242821, -0.87785842],
        [ 0.04221375,  0.58281521, -1.10061918]]),
 np.array([[1.14472371],
        [0.90159072]]),
 np.array([[ 0.53035547,  5.88414161,  3.08385015,  4.28707196,  0.53035547],
        [-0.69166075, -1.42199726, -2.92064114, -3.49524533, -0.69166075],
        [-0.39675353, -1.98881216, -3.55998747, -4.44246165, -0.39675353]]),
 np.array([[ True,  True,  True, False,  True],
        [ True,  True,  True,  True,  True],
        [False, False,  True,  True, False]]),
 np.array([[0.75765067, 8.40591658, 4.40550021, 0.        , 0.75765067],
        [0.        , 0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        ]]),
 np.array([[ 0.50249434,  0.90085595],
        [-0.68372786, -0.12289023],
        [-0.93576943, -0.26788808]]),
 np.array([[ 0.53035547],
        [-0.69166075],
        [-0.39675353]]),
 np.array([[-0.53330145, -5.78898099, -3.04000407, -0.0126646 , -0.53330145]]),
 np.array([[0.36974721, 0.00305176, 0.04565099, 0.49683389, 0.36974721]]),
 np.array([[-0.6871727 , -0.84520564, -0.67124613]]),
 np.array([[-0.0126646]]))
    keep_prob = 0.7
    expected_output = (A3, cache)
    test_cases = [
        #{
        #    "name":"datatype_check",
        #    "input": [X, parameters, keep_prob],
        #    "expected": expected_output,
        #    "error":"Datatype mismatch"
        #},
        {
            "name": "shape_check",
            "input": [X, parameters, keep_prob],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [X, parameters, keep_prob],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]
    
    multiple_test(test_cases, target)
    
def backward_propagation_with_dropout_test(target):
    np.random.seed(1)
    X = np.random.randn(3, 5)
    Y = np.array([[1, 1, 0, 1, 0]])
    cache = (np.array([[-1.52855314,  3.32524635,  2.13994541,  2.60700654, -0.75942115],
           [-1.98043538,  4.1600994 ,  0.79051021,  1.46493512, -0.45506242]]), np.array([[ True, False,  True,  True,  True],
           [ True,  True,  True,  True, False]], dtype=bool), np.array([[ 0.        ,  0.        ,  4.27989081,  5.21401307,  0.        ],
           [ 0.        ,  8.32019881,  1.58102041,  2.92987024,  0.        ]]), np.array([[-1.09989127, -0.17242821, -0.87785842],
           [ 0.04221375,  0.58281521, -1.10061918]]), np.array([[ 1.14472371],
           [ 0.90159072]]), np.array([[ 0.53035547,  8.02565606,  4.10524802,  5.78975856,  0.53035547],
           [-0.69166075, -1.71413186, -3.81223329, -4.61667916, -0.69166075],
           [-0.39675353, -2.62563561, -4.82528105, -6.0607449 , -0.39675353]]), np.array([[ True, False,  True, False,  True],
           [False,  True, False,  True,  True],
           [False, False,  True, False, False]], dtype=bool), np.array([[ 1.06071093,  0.        ,  8.21049603,  0.        ,  1.06071093],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]]), np.array([[ 0.50249434,  0.90085595],
           [-0.68372786, -0.12289023],
           [-0.93576943, -0.26788808]]), np.array([[ 0.53035547],
           [-0.69166075],
           [-0.39675353]]), np.array([[-0.7415562 , -0.0126646 , -5.65469333, -0.0126646 , -0.7415562 ]]), np.array([[ 0.32266394,  0.49683389,  0.00348883,  0.49683389,  0.32266394]]), np.array([[-0.6871727 , -0.84520564, -0.67124613]]), np.array([[-0.0126646]]))
    keep_prob = 0.8
    
    expected_output = {'dZ3': np.array([[-0.67733606, -0.50316611,  0.00348883, -0.50316611,  0.32266394]]),
 'dW3': np.array([[-0.06951191,  0.        ,  0.        ]]),
 'db3': np.array([[-0.2715031]]),
 'dA2': np.array([[ 0.58180856,  0.        , -0.00299679,  0.        , -0.27715731],
        [ 0.        ,  0.53159854, -0.        ,  0.53159854, -0.34089673],
        [ 0.        ,  0.        , -0.00292733,  0.        , -0.        ]]),
 'dZ2': np.array([[ 0.58180856,  0.        , -0.00299679,  0.        , -0.27715731],
        [ 0.        ,  0.        , -0.        ,  0.        , -0.        ],
        [ 0.        ,  0.        , -0.        ,  0.        , -0.        ]]),
 'dW2': np.array([[-0.00256518, -0.0009476 ],
        [ 0.        ,  0.        ],
        [ 0.        ,  0.        ]]),
 'db2': np.array([[0.06033089],
        [0.        ],
        [0.        ]]),
 'dA1': np.array([[ 0.36544439,  0.        , -0.00188233,  0.        , -0.17408748],
        [ 0.65515713,  0.        , -0.00337459,  0.        , -0.        ]]),
 'dZ1': np.array([[ 0.        ,  0.        , -0.00188233,  0.        , -0.        ],
        [ 0.        ,  0.        , -0.00337459,  0.        , -0.        ]]),
 'dW1': np.array([[0.00019884, 0.00028657, 0.00012138],
        [0.00035647, 0.00051375, 0.00021761]]),
 'db1': np.array([[-0.00037647],
        [-0.00067492]])}
    
    test_cases = [
        {
            "name": "shape_check",
            "input": [X, Y, cache, keep_prob],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [X, Y, cache, keep_prob],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]
    
    multiple_test(test_cases, target)



    

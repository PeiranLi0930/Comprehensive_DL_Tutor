import tensorflow as tf
from tensorflow.keras import layers

def test_loop(test_cases):
    
    success = 0
    fails = 0
    
    for test_case in test_cases:
        try:
            if type(test_case["expected"]) == list:
                assert test_case["result"] in test_case["expected"]
                success += 1
            
            else:
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

        
def test_block_class(Block):
  
    filters = 64
    kernel_size = 3
    padding = 'same'
    pool_size = 3
    repetitions = 2
    test_block = Block(filters, kernel_size, repetitions, pool_size)
    test_block(tf.random.uniform(shape=[2, 3, 4, 5]))

    vars_test_block = vars(test_block)
    
    test_cases = [
        {
            "name": "max_pool_type_check",
            "result": type(test_block.max_pool),
            "expected": layers.MaxPooling2D,
            "error_message": f'Incorrect layer type for self.maxpool'
        },
        {
            "name": "max_pool_size_check",
            "result": vars_test_block['max_pool'].pool_size,
            "expected": (pool_size, pool_size),
            "error_message": f'max pool size incorrect. check parameters.'
        },
        {
            "name": "max_pool_size_check",
            "result": vars_test_block['max_pool'].strides,
            "expected": (2,2),
            "error_message": f'max pool strides incorrect. check parameters.'
        },
        {
            "name": "conv2D_0_type_check",
            "result": type(vars_test_block['conv2D_0']),
            "expected": layers.Conv2D,
            "error_message": f'Incorrect layer type for block_0'
        },
        {
            "name": "conv2D_1_type_check",
            "result": type(vars_test_block['conv2D_1']),
            "expected": layers.Conv2D,
            "error_message": f'Incorrect layer type for block_0'
        },
        {
            "name": "conv2D_0_filters_check",
            "result": vars_test_block['conv2D_0'].filters,
            "expected": filters,
            "error_message": f'Incorrect filters for Conv2D layer. Please check parameters.'
        },
        {
            "name": "conv2D_0_kernel_size_check",
            "result": vars_test_block['conv2D_0'].kernel_size,
            "expected": (kernel_size, kernel_size),
            "error_message": f'Incorrect kernel_size for Conv2D layer. Please check parameters.'
        },
        {
            "name": "conv2D_0_activation_check",
            "result": vars_test_block['conv2D_0'].activation,
            "expected": [tf.keras.activations.relu, tf.nn.relu],
            "error_message": f'Incorrect activation for Conv2D layer. Please check parameters.'
        },
                {
            "name": "conv2D_0_padding_check",
            "result": vars_test_block['conv2D_0'].padding,
            "expected": padding,
            "error_message": f'Incorrect padding for Conv2D layer. Please check parameters.'
        },
        
    ]
    
    test_loop(test_cases)
    
def test_myvgg_class(MyVGG, Block):
    test_vgg = MyVGG(num_classes=2)
    test_vgg_layers = test_vgg.layers

    def get_block_params(block):
        return (block.filters, block.kernel_size, block.repetitions)
    
    test_cases = [
        {
            "name": "block_a_type_check",
            "result": type(test_vgg.block_a),
            "expected": Block,
            "error_message": "self.block_a has an incorrect type. Please check declaration."
        },
        {
            "name": "block_b_type_check",
            "result": type(test_vgg.block_b),
            "expected": Block,
            "error_message": "self.block_b has an incorrect type. Please check declaration."
        },
        {
            "name": "block_c_type_check",
            "result": type(test_vgg.block_c),
            "expected": Block,
            "error_message": "self.block_c has an incorrect type. Please check declaration."
        },
        {
            "name": "block_d_type_check",
            "result": type(test_vgg.block_d),
            "expected": Block,
            "error_message": "self.block_d has an incorrect type. Please check declaration."
        },
        {
            "name": "block_e_type_check",
            "result": type(test_vgg.block_e),
            "expected": Block,
            "error_message": "self.block_e has an incorrect type. Please check declaration."
        },
        {
            "name": "block_a_param_check",
            "result": get_block_params(test_vgg.block_a),
            "expected": (64, 3, 2),
            "error_message": "self.block_a has incorrect parameters. Please check hints in the code comments."
        },
        {
            "name": "block_b_param_check",
            "result": get_block_params(test_vgg.block_b),
            "expected": (128, 3, 2),
            "error_message": "self.block_b has incorrect parameters. Please check hints in the code comments."
        },
        {
            "name": "block_c_param_check",
            "result": get_block_params(test_vgg.block_c),
            "expected": (256, 3, 3),
            "error_message": "self.block_c has incorrect parameters. Please check hints in the code comments."
        },
        {
            "name": "block_d_param_check",
            "result": get_block_params(test_vgg.block_d),
            "expected": (512, 3, 3),
            "error_message": "self.block_d has incorrect parameters. Please check hints in the code comments."
        },
        {
            "name": "block_e_param_check",
            "result": get_block_params(test_vgg.block_e),
            "expected": (512, 3, 3),
            "error_message": "self.block_e has incorrect parameters. Please check hints in the code comments."
        },
        {
            "name": "flatten_type_check",
            "result": type(test_vgg.flatten),
            "expected": layers.Flatten,
            "error_message": "self.flatten has an incorrect type. Please check declaration."
        },
        {
            "name": "fc_type_check",
            "result": type(test_vgg.fc),
            "expected": layers.Dense,
            "error_message": "self.fc has an incorrect type. Please check declaration."
        },
        {
            "name": "fc_units_check",
            "result": test_vgg.fc.units,
            "expected": 256,
            "error_message": "self.fc has an incorrect number of units. Please check declaration."
        },
        {
            "name": "fc_activation_check",
            "result": test_vgg.fc.activation,
            "expected": [tf.keras.activations.relu, tf.nn.relu],
            "error_message": "self.fc has an incorrect activation. Please check declaration."
        },
        {
            "name": "classifier_type_check",
            "result": type(test_vgg.classifier),
            "expected": layers.Dense,
            "error_message": "self.classifier has an incorrect type. Please check declaration."
        },
        {
            "name": "fc_units_check",
            "result": test_vgg.classifier.units,
            "expected": 2,
            "error_message": "self.classifier has an incorrect number of units. Please check declaration."
        },
        {
            "name": "fc_activation_check",
            "result": test_vgg.classifier.activation,
            "expected": tf.keras.activations.softmax,
            "error_message": "self.classifier has an incorrect activation. Please check declaration."
        },
        {
            "name": "layer_0_check",
            "result": type(test_vgg_layers[0]),
            "expected": Block,
            "error_message": "Layer 0 of myVGG is incorrect. Please check its call() method."
        },
        {
            "name": "layer_1_check",
            "result": type(test_vgg_layers[1]),
            "expected": Block,
            "error_message": "Layer 1 of myVGG is incorrect. Please check its call() method."
        },
        {
            "name": "layer_2_check",
            "result": type(test_vgg_layers[2]),
            "expected": Block,
            "error_message": "Layer 2 of myVGG is incorrect. Please check its call() method."
        },
        {
            "name": "layer_3_check",
            "result": type(test_vgg_layers[3]),
            "expected": Block,
            "error_message": "Layer 3 of myVGG is incorrect. Please check its call() method."
        },
        {
            "name": "layer_4_check",
            "result": type(test_vgg_layers[4]),
            "expected": Block,
            "error_message": "Layer 4 of myVGG is incorrect. Please check its call() method."
        },
        {
            "name": "layer_5_check",
            "result": type(test_vgg_layers[5]),
            "expected": layers.Flatten,
            "error_message": "Layer 5 of myVGG is incorrect. Please check its call() method."
        },
        {
            "name": "layer_6_check",
            "result": type(test_vgg_layers[6]),
            "expected": layers.Dense,
            "error_message": "Layer 6 of myVGG is incorrect. Please check its call() method."
        },
        {
            "name": "layer_7_check",
            "result": type(test_vgg_layers[7]),
            "expected": layers.Dense,
            "error_message": "Layer 7 of myVGG is incorrect. Please check its call() method."
        },
        
    ]
    
    test_loop(test_cases)
    
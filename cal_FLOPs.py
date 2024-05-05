import math

def calculate_flops_resnet(input_size, resnet_name):
    total_flops = 0
    resnet = resnet_config[resnet_name]
    input_channels = 3
    for layer in resnet:
        if layer['type'] == 'conv1':
            total_flops += calculate_conv_flops(input_size // 2, layer['kernel_size'], input_channels, layer['output_channels'])
        elif layer['type'] == 'fc':
            total_flops += calculate_fc_flops(layer['input_size'], layer['output_size'])
        else:
            if resnet_name == 'resnet18' or resnet_name == 'resnet34':
                total_flops += calculate_basic_flops(layer, input_size // 2)
            else:
                total_flops += calculate_bottle_flops(layer, input_size // 2)
        input_size = input_size // 2
    return total_flops


def calculate_basic_flops(layer, output_size):
    layer_flops = 0
    if layer['type'] == 'conv2_x':
        layer_flops += layer['num'] * 2 * calculate_conv_flops(output_size, layer['kernel_size'], layer['output_channels'], layer['output_channels'])
    else:
        layer_flops += calculate_conv_flops(output_size, layer['kernel_size'], layer['output_channels'] // 2, layer['output_channels'])
        layer_flops += calculate_conv_flops(output_size, layer['kernel_size'], layer['output_channels'], layer['output_channels'])
        layer_flops += calculate_conv_flops(output_size, 1, layer['output_channels'] // 2, layer['output_channels'])
        other_flops = 0
        other_flops += 2 * calculate_conv_flops(output_size, layer['kernel_size'], layer['output_channels'], layer['output_channels'])
        layer_flops += (layer['num'] - 1) * other_flops
    return layer_flops


def calculate_bottle_flops(layer, output_size):
    layer_flops = 0
    if layer['type'] == 'conv2_x':
        layer_flops += calculate_conv_flops(output_size, 1, layer['output_channels'] // 4, layer['output_channels'] // 4)
        layer_flops += calculate_conv_flops(output_size, 1, layer['output_channels'] // 4, layer['output_channels'])    # resnet shortcut
    else:
        layer_flops += calculate_conv_flops(output_size, 1, layer['output_channels'] // 2, layer['output_channels'] // 4)
        layer_flops += calculate_conv_flops(output_size, 1, layer['output_channels'] // 2, layer['output_channels'])    # resnet shortcut
    layer_flops += calculate_conv_flops(output_size, layer['kernel_size'], layer['output_channels'] // 4, layer['output_channels'] // 4)
    layer_flops += calculate_conv_flops(output_size, 1, layer['output_channels'] // 4, layer['output_channels'])

    other_flops = 0
    other_flops += calculate_conv_flops(output_size, 1, layer['output_channels'], layer['output_channels'] // 4)
    other_flops += calculate_conv_flops(output_size, layer['kernel_size'], layer['output_channels'] // 4, layer['output_channels'] // 4)
    other_flops += calculate_conv_flops(output_size, 1, layer['output_channels'] // 4, layer['output_channels'])
    layer_flops += (layer['num'] - 1) * other_flops
    return layer_flops


resnet_config = {
    'resnet18': [
        {'type': 'conv1', 'kernel_size': 7, 'output_channels': 64, 'num': 1},
        {'type': 'conv2_x', 'kernel_size': 3, 'output_channels': 64, 'num': 2, 'basicBlock': True},
        {'type': 'conv3_x', 'kernel_size': 3, 'output_channels': 128, 'num': 2, 'basicBlock': True},
        {'type': 'conv4_x', 'kernel_size': 3, 'output_channels': 256, 'num': 2, 'basicBlock': True},
        {'type': 'conv5_x', 'kernel_size': 3, 'output_channels': 512, 'num': 2, 'basicBlock': True},
        {'type': 'fc', 'input_size': 512, 'output_size': 1000}
    ],
    'resnet34': [
        {'type': 'conv1', 'kernel_size': 7, 'output_channels': 64, 'num': 1},
        {'type': 'conv2_x', 'kernel_size': 3, 'output_channels': 64, 'num': 3, 'basicBlock': True},
        {'type': 'conv3_x', 'kernel_size': 3, 'output_channels': 128, 'num': 4, 'basicBlock': True},
        {'type': 'conv4_x', 'kernel_size': 3, 'output_channels': 256, 'num': 6, 'basicBlock': True},
        {'type': 'conv5_x', 'kernel_size': 3, 'output_channels': 512, 'num': 3, 'basicBlock': True},
        {'type': 'fc', 'input_size': 512, 'output_size': 1000}
    ],
    'resnet50': [
        {'type': 'conv1', 'kernel_size': 7, 'output_channels': 64, 'num': 1},  # Initial convolution
        {'type': 'conv2_x', 'kernel_size': 3, 'output_channels': 256, 'num': 3, 'bottleneck': True},  # First set of bottlenecks
        {'type': 'conv3_x', 'kernel_size': 3, 'output_channels': 512, 'num': 4, 'bottleneck': True},  # Second set
        {'type': 'conv4_x', 'kernel_size': 3, 'output_channels': 1024, 'num': 6, 'bottleneck': True},  # Third set
        {'type': 'conv5_x', 'kernel_size': 3, 'output_channels': 2048, 'num': 3, 'bottleneck': True},  # Fourth set
        {'type': 'fc', 'input_size': 2048, 'output_size': 1000},  # Fully connected layer
    ],
    'resnet101': [
        {'type': 'conv1', 'kernel_size': 7, 'output_channels': 64, 'num': 1},  # Initial convolution
        {'type': 'conv2_x', 'kernel_size': 3, 'output_channels': 256, 'num': 3, 'bottleneck': True},  # First set of bottlenecks
        {'type': 'conv3_x', 'kernel_size': 3, 'output_channels': 512, 'num': 4, 'bottleneck': True},  # Second set
        {'type': 'conv4_x', 'kernel_size': 3, 'output_channels': 1024, 'num': 23, 'bottleneck': True},  # Third set
        {'type': 'conv5_x', 'kernel_size': 3, 'output_channels': 2048, 'num': 3, 'bottleneck': True},  # Fourth set
        {'type': 'fc', 'input_size': 2048, 'output_size': 1000},  # Fully connected layer
    ],
    'resnet152': [
        {'type': 'conv1', 'kernel_size': 7, 'output_channels': 64, 'num': 1},  # Initial convolution
        {'type': 'conv2_x', 'kernel_size': 3, 'output_channels': 256, 'num': 3, 'bottleneck': True},  # First set of bottlenecks
        {'type': 'conv3_x', 'kernel_size': 3, 'output_channels': 512, 'num': 8, 'bottleneck': True},  # Second set
        {'type': 'conv4_x', 'kernel_size': 3, 'output_channels': 1024, 'num': 36, 'bottleneck': True},  # Third set
        {'type': 'conv5_x', 'kernel_size': 3, 'output_channels': 2048, 'num': 3, 'bottleneck': True},  # Fourth set
        {'type': 'fc', 'input_size': 2048, 'output_size': 1000},  # Fully connected layer
    ]
}


def calculate_conv_flops(output_size, kernel_size, input_channels, output_channels):
    return kernel_size * kernel_size * input_channels * output_channels * output_size * output_size


def calculate_fc_flops(input_size, output_size):
    return input_size * output_size


if __name__ == '__main__':
    resnet_name = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    for _ in resnet_name:
        print("Calculated FLOPs for {}: ".format(_), format(calculate_flops_resnet(224, _), ".2e"))




dataset_params = {
  'data_path' : './cifar10_data',
  'model_path': './model/train'
}

common_params = {
  'net_name'     : 'vgg11',
  'batch_size'   : 128,
  'image_size'   : (32, 32),
  'learning_rate': 0.01,
  'moment'       : 0.9,
  'display_step' : 100,
  'num_epochs'   : 500,
  'predict_step' : 500
}

graph_node = {
  'input'      : 'input:0',
  'is_training': 'is_training:0',
  'keep_prob'  : 'keep_prob:0',
  'output'     : 'output:0'
}

net_style = ['vgg11', 'vgg13', 'vgg16', 'vgg19',
             'resnet20', 'resnet32', 'resnet44', 'resnet56',
             'XceptionNet',
             'MobileNet',
             'DensetNet40_12', 'DenseNet100_12', 'DenseNet100_24',
             'DenseNetBC100_12', 'DenseNetBC250_24', 'DenseNetBC190_40',
             'ResNext50', 'ResNext101',
             'SqueezeNetA', 'SqueezeNetB',
             'SE_Resnet_50', 'SE_Resnet_101']

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
DEFAULT_IMAGE_BYTES = HEIGHT * WIDTH * NUM_CHANNELS
# The record is the image plus a one-byte label
RECORD_BYTES = DEFAULT_IMAGE_BYTES + 1
NUM_CLASSES = 10
NUM_DATA_FILES = 5
NUM_IMAGES = {'train': 50000, 'validation': 10000, }

net_layers = {
  'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}
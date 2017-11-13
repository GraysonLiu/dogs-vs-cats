import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


class ModelWrapper(object):
    def __init__(self, learning_rate, img_size, tensorboard_dir='./log'):
        self.description = '6conv'
        self.learning_rate = learning_rate
        self.img_size = img_size
        self.tensorboard_dir = tensorboard_dir
        self.name = 'dogs_vs_cats-{}-{}-{}'.format(self.description, self.learning_rate,
                                                   self.img_size)
        self.model = self.create_model_architecture()

    def create_model_architecture(self):
        convnet = input_data(shape=[None, self.img_size, self.img_size, 1], name='input')

        convnet = conv_2d(convnet, 32, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 32, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 32, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=self.learning_rate,
                             loss='categorical_crossentropy',
                             name='targets')

        return tflearn.DNN(convnet, tensorboard_dir=self.tensorboard_dir)

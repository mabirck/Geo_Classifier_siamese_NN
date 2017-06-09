from core.convolutional import CNN
from keras.models import Sequential
from keras.layers import Dense, Merge, Dropout
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import LeakyReLU

class Siamese:
    def __init__(self, args):
        self.args = args
        self.arch = args.arch
        self.domain = args.domain

        self.heads = ['N', 'E', 'S', 'W']
        self.headsMap = {i: None for i in self.heads}
        self.model = self.build()
        print "I am the loop"

    def build(self):

        # Getting the heads ready
        for i in self.heads:
            self.headsMap[i] = CNN(self.args)

        # Define 4-siamese model from heads
        print self.headsMap['N'].summary()
        model = (concatenate([self.headsMap['N'].outputs, self.headsMap['E'].outputs, \
                         self.headsMap['S'].outputs, self.headsMap['W'].outputs]))

        model.add(Dense(1024, name='fc_merge'))
        model.add(Dropout(0.5))
        model.add(LeakyReLU(0.1))
        model.add(Dense(4, name='fc_final'))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        return model

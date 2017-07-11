from core.convolutional import CNN
from keras.models import Sequential, Model
from keras.layers import Dense, Merge, Dropout, Concatenate, Activation, concatenate
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
            self.headsMap[i] = CNN(self.args, i)

        # Define 4-siamese model from heads
        print self.headsMap['N'].summary()

	combine  = concatenate([self.headsMap['N'].layers[-1].output, self.headsMap['E'].layers[-1].output, \
                         self.headsMap['S'].layers[-1].output, self.headsMap['W'].layers[-1].output])

        fc_merge = (Dense(1024, name='fc_merge'))(combine)
        dropout = (Dropout(0.5))(fc_merge)
        LK = (LeakyReLU(0.1))(dropout)
        final_fc = (Dense(4, name='fc_final'))(LK)

        classifier = (Activation('softmax'))(final_fc)
	
	model = Model(inputs = [self.headsMap['N'].input, self.headsMap['E'].input, \
                                self.headsMap['S'].input, self.headsMap['W'].input],
		      outputs = [classifier])
	
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        return model
    def fit(self, X, Y, epochs):
	# Select archictecture 'cause it will change between keras and tensorflow
	if self.arch == 'AlexNet':
	   
	   # Selecting pretrained domain weights	   
	   if self.domain == 'imagenet':
	      return model.fit(X, Y, epochs = epochs)

	   elif self.domain == 'places':
	      sess = tf.session()
	      sess.run(model)
	elif self.arch == 'VGG16':
            if self.domain == 'imagenet':
	        return self.model.fit(X, Y, epochs=epochs)   	      

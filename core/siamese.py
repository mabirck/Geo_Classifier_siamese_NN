from core.convolutional import CNN
from keras.models import Sequential, Model
from keras.layers import Dense, Merge, Dropout, Concatenate, Activation, concatenate, Input
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import Callback, ModelCheckpoint, LambdaCallback
from keras.optimizers import Adadelta
from core.utils import SaveMetrics

class Siamese:
    def __init__(self, args):
        self.args = args
        self.arch = args.arch
        self.domain = args.domain

        self.heads = ['N', 'E', 'S', 'W']
        self.headsMap = {i: None for i in self.heads}
        self.model = None
	self.build()
        #print "I am the loop"

    def build(self):

        # Getting the heads ready
        for i in self.heads:
	    self.headsMap[i] = CNN(self.args, i)

        # Define 4-siamese model from heads
        print self.headsMap['N'].summary()
	

	Nhead = self.headsMap['N'].outputs
	#Nhead = Dense(1024, name='northDense')(Nhead)
	Nhead = Dropout(0)(Nhead)
	Nhead = LeakyReLU(0.1)(Nhead)
	TN = Model(input=self.headsMap['N'].input, output=Nhead)

	Ehead = self.headsMap['E'].outputs
	#Ehead = Dense(1024, name='eastDense')(Ehead)
	Ehead = Dropout(0)(Ehead)
	Ehead = LeakyReLU(0.1)(Ehead)
	TE = Model(inputs=self.headsMap['E'].input, outputs=Ehead)

	Shead = self.headsMap['S'].outputs
	#Shead = Dense(1024, name='southDense')(Shead)
	Shead = Dropout(0)(Shead)
	Shead = LeakyReLU(0.1)(Shead)
	TS = Model(input=self.headsMap['S'].input, output=Shead)

	Whead = self.headsMap['W'].outputs
	#Whead = Dense(1024, name='weastDense')(Whead)
	Whead = Dropout(0)(Whead)
	Whead = LeakyReLU(0.1)(Whead)
	TW = Model(input=self.headsMap['W'].input, output=Whead)



	
	self.model = Sequential()
#	self.model.add(Input(shape=(277, 277, 3)))
	self.model.add(Merge([TN, TE, TS, TW], mode='concat'))

        self.model.add(Dense(512, name='fc_merge'))
        self.model.add(Dropout(0.5, name='dp'))
        self.model.add(LeakyReLU(0.1, name='shi'))
        self.model.add(Dense(4, name='fc_final'))
	self.model.add(Activation('softmax', name='activia'))
	
	#self.model = Model(inputs = [self.headsMap['N'].input, self.headsMap['E'].input, \
        #                        self.headsMap['S'].input, self.headsMap['W'].input],
	#	      outputs = classifier)

	optimizer = Adadelta(lr=0.002)	

        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	
	if self.args.load != None:
	    self.model.load_weights(self.args.load)
        

    def fit(self, X, Y, epochs, saveit, e):
	
	
	checkpoint = ModelCheckpoint('weights.'+str(e)+'.hdf5', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)
	
	#saveMetrics = SaveMetrics() 
	
	callbacks = [checkpoint]	

	# Select archictecture 'cause it will change between keras and tensorflow
	if self.arch == 'AlexNet':
	   
	   # Selecting pretrained domain weights	   
	   if self.domain == 'imagenet':
	      return self.model.fit(X, Y, verbose=0, epochs = epochs)

	   elif self.domain == 'places':
	      sess = tf.session()
	      sess.run(model)
	elif self.arch == 'VGG16':
            if self.domain == 'imagenet':
			
		if(saveit == True):
	            return self.model.fit(X, Y, verbose = 0, epochs=epochs, callbacks=callbacks)   	      
		else:
		    return self.model.fit(X, Y, verbose = 0, epochs=epochs)
    def evaluate(self, VAL):
	ls = self.model.evaluate(VAL[0], VAL[1], verbose=0)        
        #print 'printing ls:',ls
	return ls

    def predict(self, VAL):
	ls = self.model.predict(VAL[0], verbose=0)        
        #print 'printing ls:',ls
	return ls
	
	

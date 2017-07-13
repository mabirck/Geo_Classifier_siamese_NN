import glob
import json
from scipy.misc import imread, imsave, imresize
import numpy as np
from keras.utils import np_utils
from keras.layers import Input, Convolution2D
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from customlayers import convolution2Dgroup, crosschannelnormalization, splittensor, Softmax4D
from keras.callbacks import Callback, ModelCheckpoint


def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False
    return model
def makeByPass(model):

    head = model.outputs
    

    return head

def AlexNet(include_top = False, weights = 'imagenet'):

    inputs = Input(shape=(227,227, 3))

    conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)
    conv_2 = merge([
        Convolution2D(128,5,5,activation="relu",name='conv_2_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_2)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)
    conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)
    conv_4 = merge([
        Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_4)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

    conv_5 = ZeroPadding2D((1,1))(conv_4)
    conv_5 = merge([
        Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_5)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)


    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu',name='dense_1')(dense_1)
    dense_2 = Dropout(0.25)(dense_1)
    dense_2 = Dense(4096, activation='relu',name='dense_2')(dense_2)
    dense_3 = Dropout(0.25)(dense_2)
    dense_3 = Dense(1000,name='dense_3')(dense_3)
    prediction = Activation("softmax",name="softmax")(dense_3)


    model = Model(input=inputs, output=prediction)
    if weights == 'imagenet':
        model.load_weights(weights_path)
    elif weights == 'places':
        print "I ensure you i will try to implement it"

    pop_layer(model)

    return model

def vgg16(include_top = False, weights = 'imagenet'):
    return VGG16(include_top, weights)


def freezeAndRename(model,card):
    for i in model.layers:
        i.name = i.name+'_'+card
        i.trainable = False
    return model

############# PREPROCESSING IMAGES FUNCS ########################
def loadCardinal(path, eval=False):
    #print path, 'why in hell this is a tuple'  
    if eval == False:  
        path = '_'.join(path.split('_')[:-1])
    
    card = ['_0.jpg', '_90.jpg', '_180.jpg', '_270.jpg']
    #labelToOneHot = {'blue' : 0, 'green' : 1, 'orange' : 2, 'red':3}
    labelToOneHot = {'blue' : 0, 'red' : 1}

    label = labelToOneHot[path.split('/')[-2]]
        
    n = np.array(imresize(imread(path+card[0]), (224, 224, 3)))
    e = np.array(imresize(imread(path+card[1]), (224, 224, 3)))
    s = np.array(imresize(imread(path+card[2]), (224, 224, 3)))
    w = np.array(imresize(imread(path+card[3]), (224, 224, 3)))
    #print n.shape, e.shape, s.shape, w.shape

    #print type(n)
    return n, e, s, w, label
    
def getImages(args, valPoints, eval=False):

    mode = args.mode
    path = str(args.data)
    batch_size = args.batch_size
    
    if(eval==True):
        #print "The path is in here ..............", str(path)
        #images_list = glob.glob(path+'/test/*/*')
        #print images_list
        batch_list = valPoints

        images = []
        labels = []

        for b in batch_list:
     	    #print b, len(images)
            n, e, s, w, y = loadCardinal(b, eval)
            images.append(np.array([n, e, s, w]))
            labels.append(y) 

        return np.array(images), labels
	    

    elif(mode == 'train'):
        #print "The path is in here ..............", str(path)
        images_list = glob.glob(path+'/train/*/*')
        #print images_list
        batch_list = np.random.choice(images_list, batch_size)

        images = []
        labels = []
        
        for b in batch_list:
            n, e, s, w, y = loadCardinal(b)
	    #print b,  len(images)
            images.append(np.array([n, e, s, w]))
            labels.append(y) 

        return np.array(images), labels

def augment(args, X, Y):
    datagen = ImageDataGenerator(
		       rescale=1./255,
                       shear_range=0.2,
                       zoom_range=0.2,
                       horizontal_flip=True)
        
    for x, y in datagen.flow(X, Y, batch_size=args.batch_size, shuffle=False):
		new = np_utils.to_categorical(y, 4)
		return x, new
def augmentVal(args, X, Y):
    datagen = ImageDataGenerator(
                       rescale=1./255)

    for x, y in datagen.flow(X, Y, shuffle=False):
                new = np_utils.to_categorical(y, 4)
                return x, new

    


def genBatch(args):

    images, labels = getImages(args, valPoints=None, eval=False)      	
    return images, labels
        

def genVal(args, valPoints, eval):

    images, labels = getImages(args,valPoints, eval)            
    return images, labels

    

       
class SaveMetrics(Callback):
    def on_epoch_begin(self, epoch, logs):
	try:
	    with open('metrics.json', 'r') as f:   
 	        self.metrics = json.load(f)
        except:
	    with open('metrics.json', 'w') as f:
                self.metrics = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
		json.dump(self.metrics, f)
			
    def on_epoch_end(self, epoch, logs):
        
        self.metrics['loss'].append(logs.get('loss'))
	self.metrics['acc'].append(logs.get('acc'))
	
	with open('metrics.json', 'w') as f:    
    	    data = json.dump(self.metrics, f)
	
def getValSet(path):
    images_list = glob.glob(path+'*/*')
    
    unicList = set()

    for i in images_list:
	point  = '_'.join(i.split('_')[:-1])
	#print point
	unicList.add(point)

    return unicList

def initLog():
    with open('metrics.json', 'w') as f:
        metrics = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
        json.dump(metrics, f)


from keras.utils import np_utils
import sys, os, argparse
import numpy as np
from core.siamese import Siamese as SM
from core.utils import genBatch, augment

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', help='Phase: Can be train, val or test')
    parser.add_argument('--arch', type=str, default='VGG16', help='Passes the architecture to be learned')
    parser.add_argument('--load', action='store_true', default=False, help='Turn on to load the pretrained model')
    parser.add_argument('--domain', type=str, default='imagenet', help='Chose domain weights to be load')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--data', type=str, default='./data')
    
    args = parser.parse_args()
    print args

    if args.mode == "train":
        Siamese = SM(args)

        for e in range(args.epochs):
            print "Epoch", e

            for s in range(args.steps):
				
                X, Y = genBatch(args)
		#print 'type of batch', type(X), type(Y)
          	#print 'This is the Y batch', Y
                #print "len of sub levels", len(X), len(X[0]), len(X[0][0]), len(X[0][0][0])
		#print "type os sub levels", type(X), type(X[0]), type(X[0][0]), type(X[0][0][0])
		N, Y = augment(args, X[:,0], Y)
#		NE = [i.reshape(1, 224, 224, 3) for i in N[:,3]]
		E = augment(args, X[:,1], Y)[0]
                S = augment(args, X[:,2], Y)[0]
		#print 'shapiisssssssssssss', len(N), len(E), len(S)    
                W = augment(args, X[:,3], Y)[0]
		#print 'images list size', N.shape, E.shape 
		#Y = np_utils.to_categorical(np.array(batch[:][1]), 4)
		data  = [N, E, S, W]
#		print batch[0][1]		
                Siamese.fit(data, Y, args.epochs)
    else:
    	val = Siamese.evaluate(x, y)
        print "The accuracy is", val



if __name__=="__main__":
    main()

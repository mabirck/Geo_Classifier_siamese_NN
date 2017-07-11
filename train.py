from keras.utils import np_utils
import sys, os, argparse
import numpy as np
from core.siamese import Siamese as SM
from core.utils import genBatch

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', help='Phase: Can be train, val or test')
    parser.add_argument('--arch', type=str, default='VGG16', help='Passes the architecture to be learned')
    parser.add_argument('--load', action='store_true', default=False, help='Turn on to load the pretrained model')
    parser.add_argument('--domain', type=str, default='imagenet', help='Chose domain weights to be load')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--data', type=str, default='./data')
    
    args = parser.parse_args()
    print args

    if args.mode == "train":
        Siamese = SM(args)

        for e in range(args.epochs):
            print "Epoch", e

            for s in range(args.steps):

                batch = genBatch(args)
		print 'type of batch', type(batch), type(batch[0])
                print 'len of batch', len(batch), len(batch[0]), len(batch[0][0])
		
#		N = batch[:][0]
#		E = batch[:][1]
#               S = batch[:][2]
#               W = batch[:][3]

		Y = np_utils.to_categorical(np.array(batch[:][1]), 4)

		
                Siamese.fit(batch[:][0][0], batch[:][1], args.epochs)
    else:
    	val = Siamese.evaluate(x, y)
        print "The accuracy is", val



if __name__=="__main__":
    main()

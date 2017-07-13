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
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data', type=str, default='./data')    
    parser.add_argument('--save_period', type=int, default=50)

    args = parser.parse_args()
    print args

    if args.mode == "train":
        Siamese = SM(args)

        for e in range(args.epochs):
            print "Epoch", e

            for s in range(2):
				
                X, Y = genBatch(args)

		N, Y = augment(args, X[:,0], Y)
		E = augment(args, X[:,1], Y)[0]
                S = augment(args, X[:,2], Y)[0]
                W = augment(args, X[:,3], Y)[0]
 
		data  = [N, E, S, W]		
                Siamese.fit(data, Y, 1)
    else:
    	val = Siamese.evaluate(x, y)
        print "The accuracy is", val



if __name__=="__main__":
    main()

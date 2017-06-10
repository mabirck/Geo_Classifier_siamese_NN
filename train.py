import sys, os, argparse
import numpy as np
from core.siamese import Siamese as S

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', help='Phase: Can be train, val or test')
    parser.add_argument('--arch', type=str, default='VGG16', help='Passes the architecture to be learned')
    parser.add_argument('--load', action='store_true', default=False, help='Turn on to load the pretrained model')
    parser.add_argument('--domain', type=str, default='imagenet', help='Chose domain weights to be load')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--steps', type=int, default=100)


    args = parser.parse_args()
    print args

    if args.mode == "train":
        Siamese = S(args)

        for i in range(args.epochs):
            print "Epoch", i
            Siamese.fit(x, y, 1, metrics)
    else:

	val = Siamese.evaluate(x, y)
        print "The accuracy is", val



if __name__=="__main__":
    main()

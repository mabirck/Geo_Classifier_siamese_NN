import sys, os, argparse
import numpy as np
from core.siamese import Siamese as S

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', help='Phase: Can be train, val or test')
    parser.add_argument('--arch', type=str, default='VGG16', help='Passes the architecture to be learned')
    parser.add_argument('--load', action='store_true', default=False, help='Turn on to load the pretrained model')
    parser.add_argument('--domain', type=str, default='imagenet', help='Chose domain weights to be load')
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--steps', type=int, default=100)


    args = parser.parse_args()
    print args

    if args.mode == "train":
        model = S(args)

        for i in epochs:
            print "Epoch", i
            model.fit(x, y, epochs, metrics)
    else:
        print "I will evaluate"



if __name__=="__main__":
    main()

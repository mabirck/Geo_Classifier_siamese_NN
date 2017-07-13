import time
import random
import sys, os, argparse
import numpy as np
from core.siamese import Siamese as SM
from core.utils import genBatch, augment, genVal, augmentVal, getValSet, initLog

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', help='Phase: Can be train, val or test')
    parser.add_argument('--arch', type=str, default='VGG16', help='Passes the architecture to be learned')
    parser.add_argument('--load', action='store_true', default=False, help='Turn on to load the pretrained model')
    parser.add_argument('--domain', type=str, default='imagenet', help='Chose domain weights to be load')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data', type=str, default='./data')    
    parser.add_argument('--save_period', type=int, default=50)

    args = parser.parse_args()
    print args

    if args.mode == "train":
        Siamese = SM(args)
	ValImageList = getValSet(args.data+'/test/')
	metrics = {}	
	initLog()	

        for e in range(args.epochs):
            print "############################# Epoch -", e,'##############################'
	    start = time.time()
	    acc = []
	    loss = []	    
 	    val_acc = []
	    val_loss = []	    

            for s in range(2):
				
                X, Y = genBatch(args)

		N, Y = augment(args, X[:,0], Y)
		E = augment(args, X[:,1], Y)[0]
                S = augment(args, X[:,2], Y)[0]
                W = augment(args, X[:,3], Y)[0]
				 
		data  = [N, E, S, W]		
                train_log = Siamese.fit(data, Y, 1, e)

		loss.append(train_log.history['loss'][0])
            	acc.append(train_log.history['acc'][0])

            while len(ValImageList) > 0:
		
		valPoints = random.sample(ValImageList, 32)		

            	X_V, Y_V = genVal(args, valPoints, eval=True)

        	testN, y_testN = augmentVal(args, X_V[:,0], Y_V)
        	testE = augmentVal(args, X_V[:,1], Y_V)[0]

        	testS = augmentVal(args, X_V[:,2], Y_V)[0]
        	testW = augmentVal(args, X_V[:,3], Y_V)[0]


	        validation_data = ([testN, testE, testS, testW], y_testN)

		vals = Siamese.evaluate(validation_data)
          
                val_loss.append(vals[0])
		val_acc.append(vals[1])

	    loss = np.mean(loss)
	    acc = np.mean(acc)
	
	    val_loss = np.mean(val_loss)
	    val_acc = np.mean(val_acc)
	    

	    with open('metrics.json', 'r') as f:   
                metrics = json.load(f)

		metrics['acc'].append(acc)
		metrics['loss'].append(loss)

		metrics['val_acc'].append(val_acc)
		metrics['val_loss'].append(val_loss)

	    with open('metrics.json', 'w') as f:    
                data = json.dump(metrics, f)
	    
	    end = time.time()
            print 'Epoch metrics ->  loss:', loss, 'acc:',acc, 'val_loss:', val_loss, 'val_acc:', val_acc, 'time_elapsed:', end-start

    else:
    	val = Siamese.evaluate(x, y)
        print "The accuracy is", val



if __name__=="__main__":
    main()

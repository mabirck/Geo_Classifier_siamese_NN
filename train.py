import json
import time
import random
import sys, os, argparse
import numpy as np
import keras.backend as K
from core.siamese import Siamese as SM
from core.utils import genBatch, augment, genVal, augmentVal, getValSet, initLog
import tensorflow as tf
from pandas_ml import ConfusionMatrix

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', help='Phase: Can be train, val or test')
    parser.add_argument('--arch', type=str, default='VGG16', help='Passes the architecture to be learned')
    parser.add_argument('--load', type=str, default=None, help='Turn on to load the pretrained model')
    parser.add_argument('--domain', type=str, default='imagenet', help='Chose domain weights to be load')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=497)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data', type=str, default='./data')    
    parser.add_argument('--save_period', type=int, default=10)

    args = parser.parse_args()
    print args

    if args.mode == "train":
        Siamese = SM(args)
	testList = getValSet(args.data+'/test/')
	trainList = getValSet(args.data+'/train/')
	metrics = {}	
	initLog()	

        for e in range(args.epochs):
            print "############################# Epoch -", e,'##############################'
	    start = time.time()
	    acc = []
	    loss = []	    
	    val_acc = []
	    val_loss = []	    
	    ValImageList = list(testList)
	    TrainImageList = list(trainList)
            
            if(e % args.save_period == 0):
	        saveit = True
	    else:
	        saveit = False	    

                 #	    print 'STARTIIIIING TRAIIIIN PART _>>>>>>>>>>>>>>>>>>>

            
	    
	    for s in range(0, len(trainList), 16):
		
    		sys.stdout.write('.')
	    	sys.stdout.flush()		

		if(len(TrainImageList) < 16):
		    bs = len(TrainImageList)
		else:
		    bs = 16		

		trainPoints = random.sample(TrainImageList, bs)		        

		X, Y = genVal(args, trainPoints, eval=False)
		
	    	N, Y = augment(args, X[:,0], Y)
	    	E = augment(args, X[:,1], Y)[0]
	        S = augment(args, X[:,2], Y)[0]
	        W = augment(args, X[:,3], Y)[0]
						
		TrainImageList = [image for image in TrainImageList if image not in trainPoints]
		#TrainImageList.delete(trainPoints)
			
		data  = [N, E, S, W]
    		train_log = None		    
            	with tf.device('/gpu:0'):
	     	    train_log = Siamese.fit(data, Y, 1, saveit, e)
		
		saveit = False
		
		loss.append(train_log.history['loss'][0])
    	 	acc.append(train_log.history['acc'][0])

        	#print len(ValImageList)
            for i in range(0, len(testList), 16):
		
		if(len(ValImageList) < 16):
                    bs = len(ValImageList)
                else:
                    bs = 16 
	
                valPoints = random.sample(ValImageList, bs)		
	        #print valPoints
	        X_V, Y_V = genVal(args, valPoints, eval=True)

	        testN, y_testN = augmentVal(args, X_V[:,0], Y_V)
	        testE = augmentVal(args, X_V[:,1], Y_V)[0]

	        testS = augmentVal(args, X_V[:,2], Y_V)[0]
	        testW = augmentVal(args, X_V[:,3], Y_V)[0]
		

	        validation_data = ([testN, testE, testS, testW], y_testN)
	        vals = None
	        
		with tf.device('/gpu:0'):
	            vals = Siamese.evaluate(validation_data)
		  
	        val_loss.append(vals[0])
	        val_acc.append(vals[1])
		    
                ValImageList = [image for image in ValImageList if image not in valPoints]
	
	
	    val_loss = np.mean(val_loss)
	    val_acc = np.mean(val_acc)
	    loss = np.mean(loss)
	    acc = np.mean(acc)	    

            with open('metrics.json', 'r') as f:   
	        metrics = json.load(f)

	        metrics['acc'].append(acc)
	        metrics['loss'].append(loss)

	        metrics['val_acc'].append(val_acc)
	        metrics['val_loss'].append(val_loss)

	    with open('metrics.json', 'w') as f:    
	        data = json.dump(metrics, f)
	    end = time.time()
	    print '\nEpoch metrics ->  loss:', loss, 'acc:',acc, 'val_loss:', val_loss, 'val_acc:', val_acc, 'time_elapsed:', end-start, '\n'

    else:

        Siamese = SM(args)	
	testList = getValSet(args.data+'/test/')
    
        ValImageList = list(testList)


	final_labels = []
	real_labels = []	

	for i in range(0, len(testList), 16):
		pred = None		

		if(len(ValImageList) < 16):
                    bs = len(ValImageList)
                else:
                    bs = 16 
	
                valPoints = random.sample(ValImageList, bs)		
	        #print valPoints
	        X_V, Y_V = genVal(args, valPoints, eval=True)
		real_labels.append(Y_V)

	        testN, y_testN = augmentVal(args, X_V[:,0], Y_V)
	        testE = augmentVal(args, X_V[:,1], Y_V)[0]

	        testS = augmentVal(args, X_V[:,2], Y_V)[0]
	        testW = augmentVal(args, X_V[:,3], Y_V)[0]
		

	        validation_data = ([testN, testE, testS, testW], y_testN)
	        vals = None
	        
		with tf.device('/gpu:0'):
	             pred = Siamese.predict(validation_data)
		      
	        #val_loss.append(vals[0])
	        #val_acc.append(vals[1])
		    
                ValImageList = [image for image in ValImageList if image not in valPoints]

		final_labels.append(np.argmax(np.array(pred), axis=1))	
		
		#print np.argmax(pred, axis=1)
	final_labels = np.array([item for sublist in final_labels for item in sublist])
	real_labels = np.array([item for sublist in real_labels for item in sublist])	

    	#val = Siamese.evaluate(x, y)
        #print "The final_labels is", final_labels
	
	confusionmatrix = ConfusionMatrix(real_labels, final_labels)
	print confusionmatrix
	print confusionmatrix.print_stats()



if __name__=="__main__":
    main()

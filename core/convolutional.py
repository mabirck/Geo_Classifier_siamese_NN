from utils import AlexNet, vgg16, pop_layer, freezeAndRename # We inport externally because there is no native AlexNet in keras

def CNN(args, n):
    if args.arch == "AlexNet":

        if args.domain == 'imagenet':
            model = AlexNet(include_top=True, weights='imagenet')
	    model = freezeAndRename(model, n) 	
            model = pop_layer(model)
	    model.layers.pop()
	    return model
        else:
            print "I will load from a TF file gotten in a caffemodel"

    elif args.arch == "VGG16":
        if args.domain == 'imagenet':
            model = vgg16(include_top=True, weights='imagenet')
            model = pop_layer(model)
            model = freezeAndRename(model, n)
	    return model
        elif args.domain == 'places':
            print 'Sure I will try weights form place also'

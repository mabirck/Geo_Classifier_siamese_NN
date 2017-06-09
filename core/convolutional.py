from utils import AlexNet, vgg16, pop_layer # We inport externally because there is no native AlexNet in keras

def CNN(args):
    if args.arch == "AlexNet":

        if args.domain == 'imagenet':
            model = AlexNet(include_top=True, weights='imagenet')
            return pop_layer(model)
        else:
            print "I will load from a TF file gotten in a caffemodel"

    elif args.arch == "VGG16":
        if args.domain == 'imagenet':
            model = vgg16(include_top=True, weights='imagenet')
            return pop_layer(model)
        elif args.domain == 'places':
            print 'Sure I will try weights form place also'

# Import Libraries
from __future__ import print_function
import argparse
import chainer
from chainer import iterators, training, optimizers, datasets, serializers
from chainer.dataset import concat_examples
from chainer.datasets import LabeledImageDataset
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.function_hooks import TimerHook

from itertools import chain
from PIL import Image
import glob
import os

# Model VGG16
class VGG16(chainer.Chain): #Define the model
    def __init__(self, num_class, train=True):
        w = chainer.initializers.HeNormal()
        super(VGG16, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 64, 3, stride=1, pad=1, initialW=w)
            self.conv1_2 = L.Convolution2D(64, 64, 3, stride=1, pad=1, initialW=w)
            self.conv2_1 = L.Convolution2D(64, 128, 3, stride=1, pad=1, initialW=w)
            self.conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1, initialW=w)
            self.conv3_1 = L.Convolution2D(128, 256, 3, stride=1, pad=1, initialW=w)
            self.conv3_2 = L.Convolution2D(256, 256, 3, stride=1, pad=1, initialW=w)
            self.conv3_3 = L.Convolution2D(256, 256, 3, stride=1, pad=1, initialW=w)
            self.conv4_1 = L.Convolution2D(256, 512, 3, stride=1, pad=1, initialW=w)
            self.conv4_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=w)
            self.conv4_3 = L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=w)
            self.conv5_1 = L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=w)
            self.conv5_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=w)
            self.conv5_3 = L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=w)
            self.fc6 = L.Linear(512*7*7, 4096, initialW=w)
            self.fc7 = L.Linear(4096, 4096, initialW=w)
            self.fc8 = L.Linear(4096, num_class, initialW=w)

    def __call__(self, x):
        # resize image 
        h = F.resize_images(x,(224,224))
        
        # block 1: 64
        h = F.relu(self.conv1_1(h))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # block 2: 128
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # block 3: 256
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # block 4: 512
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # block 5: 512
        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(F.local_response_normalization(h), ksize=2, stride=2)

        # fc
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)
        return h

# Parse functions
parser = argparse.ArgumentParser(description='Chainer - Hoook time')
parser.add_argument('--dataset', '-dt', 
                    default = '/media/phongphu/HHD1_Doctor_Ubuntu/00_ADC_Project/02_Code_Implement/03_Classification_Patches/01_ADC_Speicific_Class/02_Output_DataPreprocessing/01_Patches/', 
                    help='Path to the dataset')
parser.add_argument('--batchsize', '-b', type=int, default=32, help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=20, help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='weights', help='Directory to output the result')
parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
parser.add_argument('--unit', '-u', type=int, default=8, help='Number of output layer units')
args = parser.parse_args(['-g','0']) #The negative device number means CPU.

print('GPU: {}'.format(args.gpu))
print('# unit: {}'.format(args.unit))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('')

# Import model
model = VGG16(args.unit)
classifier_model = L.Classifier(model)
if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use() # Make a specified GPU current
    classifier_model.to_gpu() # Copy the model to the GPU

# Optimizer
optimizer = chainer.optimizers.Adam() #Adam 
optimizer.setup(classifier_model)

# Dataset
def load_dataset_train(datatype = 'train'):
    def transform(data):
        img, label = data
        img = img / 255.
        return img, label
    
    IMG_DIR = str(args.dataset) + datatype + '/'
         
    dnames = glob.glob('{}/*'.format(IMG_DIR))
    fnames = [glob.glob('{}/*.bmp'.format(d)) for d in dnames]
    fnames = list(chain.from_iterable(fnames))
    
    labels = [os.path.basename(os.path.dirname(fn)) for fn in fnames]
    dnames = [os.path.basename(d) for d in dnames]
    labels = [dnames.index(l) for l in labels]
    d = LabeledImageDataset(list(zip(fnames, labels)))
    d = chainer.datasets.TransformDataset(d, transform)
    return d

# Load the dataset
train = load_dataset_train('train')
test = load_dataset_train('valid')

train_iter = chainer.iterators.SerialIterator(train, args.batchsize) #Set Training data batch iterater
test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

# Forward the test data after each of training to calcuat the validation loss/arruracy.
# Updater Trainer
updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

# Extension objects
trainer.extend(extensions.Evaluator(test_iter, classifier_model, device=args.gpu))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
trainer.extend(extensions.ProgressBar())


if args.resume:
    # Resume from a snapshot
    serializers.load_npz(args.resume, trainer)
trainer.run()
serializers.save_npz('{}/ADC.model'.format(args.out), model) #Save the model

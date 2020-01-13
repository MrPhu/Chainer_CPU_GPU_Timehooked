# Import Libraries
from __future__ import print_function
import argparse
import chainer
import numpy as np
from chainer import iterators, training, optimizers, datasets, serializers
from chainer.dataset import concat_examples
from chainer.datasets import LabeledImageDataset
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.function_hooks import TimerHook

import cupy as cp
from itertools import chain
from PIL import Image
import glob
import os

# Create VGG16 model
class VGG16(chainer.Chain): #Define the model
    def __init__(self, num_class, train=True):
        self.time=[0]*16
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
        
        # 64 channel blocks:S
        hook=TimerHook()
        with hook:
            h = F.relu(self.conv1_1(h))
        self.time[0]=self.time[0]+hook.total_time()

        hook1=TimerHook()
        with hook1:
            h = F.relu(self.conv1_2(h))
            h = F.max_pooling_2d(h, ksize=2, stride=2)
        self.time[1]=self.time[1]+hook1.total_time()

        # 128 channel blocks:
        hook2=TimerHook()
        with hook2:
            h = F.relu(self.conv2_1(h))
        self.time[2]=self.time[2]+hook2.total_time()

        hook3=TimerHook()
        with hook3:
            h = F.relu(self.conv2_2(h))
            h = F.max_pooling_2d(h, ksize=2, stride=2)
        self.time[3]=self.time[3]+hook3.total_time()

        # 256 channel blocks:
        hook4=TimerHook()
        with hook4:
            h = F.relu(self.conv3_1(h))
        self.time[4]=self.time[4]+hook4.total_time()

        hook5=TimerHook()
        with hook5:
            h = F.relu(self.conv3_2(h))
        self.time[5]=self.time[5]+hook5.total_time()

        hook6=TimerHook()
        with hook6:
            h = F.relu(self.conv3_3(h))
            h = F.max_pooling_2d(h, ksize=2, stride=2)
        self.time[6]=self.time[6]+hook6.total_time()

        # 512 channel blocks:
        hook7=TimerHook()
        with hook7:
            h = F.relu(self.conv4_1(h))
        self.time[7]=self.time[7]+hook7.total_time()

        hook8=TimerHook()
        with hook8:
            h = F.relu(self.conv4_2(h))
        self.time[8]=self.time[8]+hook8.total_time()
        
        hook9=TimerHook()
        with hook9:
            h = F.relu(self.conv4_3(h))
            h = F.max_pooling_2d(h, ksize=2, stride=2)
        self.time[9]=self.time[9]+hook9.total_time()

        # 512 channel blocks:
        hook10=TimerHook()
        with hook10:
            h = F.relu(self.conv5_1(h))
        self.time[10]=self.time[10]+hook10.total_time()
        hook11=TimerHook()
        with hook11:
            h = F.relu(self.conv5_2(h))
        self.time[11]=self.time[11]+hook11.total_time()
        hook12=TimerHook()
        with hook12:
            h = F.relu(self.conv5_3(h))
            h = F.max_pooling_2d(
                    F.local_response_normalization(h), 
                    ksize=2, stride=2)
        self.time[12]=self.time[12]+hook12.total_time()

        # classifier
        hook13=TimerHook()
        with hook13:
            h = F.dropout(F.relu(self.fc6(h)))
        self.time[13]=self.time[13]+hook13.total_time()
        
        hook14=TimerHook()
        with hook14:
            h = F.dropout(F.relu(self.fc7(h)))
        self.time[14]=self.time[14]+hook14.total_time()
        hook15=TimerHook()
        
        with hook15:
            h = self.fc8(h)
        self.time[15]=self.time[15]+hook15.total_time()
        return h
# Parse functions
parser = argparse.ArgumentParser(description='Chainer - Hoook time')
parser.add_argument('--dataset', '-dt', 
                    default = '/media/phongphu/HHD1_Doctor_Ubuntu/00_ADC_Project/02_Code_Implement/03_Classification_Patches/01_ADC_Speicific_Class/02_Output_DataPreprocessing/01_Patches/', 
                    help='Path to the dataset')
parser.add_argument('--out', '-o', default='weights/ADC.model', help='Directory to output the result')
parser.add_argument('--unit', '-u', type=int, default=8, help='Number of output layer units')
parser.add_argument('--gpu', '-g', type=bool, default=False, help='Using GPU')

args = parser.parse_args() #The negative device number means CPU.

print('Using GPU: {}'.format(args.gpu))
print('# unit: {}'.format(args.unit))
print('')

# Dataset
def load_dataset_test(datatype = 'train'):
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

test = load_dataset_test('test')

# Load trained model
model = VGG16(args.unit)
xp = np # No GPU => Numpy

if args.gpu == True:
    chainer.cuda.get_device('0').use() # Make a specified GPU current
    model.to_gpu() # Copy the model to the GPU
    xp = cp # Have GPU => Cupy
    
# Load Dataset
serializers.load_npz(args.out, model)

# Test first 160 images
for i in range(160):
    x = chainer.Variable(xp.asarray([test[i][0]])) # Images
    t = chainer.Variable(xp.asarray([test[i][1]])) # Labels
    y = model(x) # Reults 

for count, i in enumerate(model.time):
    print('Layer {}: {} sec'.format(str(count + 1),str(round(i,4))))


# Chainer_CPU_GPU_Allocation
This project is a homework of Parallel Computing and Deep Learning System course.

Requirements:
+ Chainer
+ Cupy

To run the training code:
  cd ./path~
  python hw2_train.py --dataset [linktodataset] --out [foldercontainmodel] --unit [outputclass] --batchsize [sizeofbatch] --epoch [numberofepoch]

To run the inference code:
+ Download Test Data:
  Link: https://drive.google.com/drive/folders/18k1W-dUwal-zJJ0EEmfs5wl0URTEQmeg?usp=sharing
+ Test data folder:
  ~/data/test/
+ Trained model:
  ~/weights/ADC.model
+ GPU:
  cd ./path~
  python hw2_inference.py --dataset [linktodataset] --out [foldercontainmode] --unit [outputclass] --gpu True
+ CPU:
  cd ./path~
  python hw2_inference.py --dataset [linktodataset] --out [foldercontainmode] --unit [outputclass] 
  
Results:

![The Graph shows the comparison btw running on GPU and CPU](https://github.com/MrPhu/Chainer_CPU_GPU_Allocation/blob/master/results/comparison_result.PNG)

Requirements:
+ Chainer
+ Cupy

To run the training code:
  cd ./path~
  python hw2_train_1.py --dataset [linktodataset] --out [foldercontainmodel] --unit [outputclass] --batchsize [sizeofbatch] --epoch [numberofepoch]

To run the inference code:
+ Download Trained Model and Test Data:
  Link: https://drive.google.com/drive/folders/18k1W-dUwal-zJJ0EEmfs5wl0URTEQmeg?usp=sharing
+ Test data folder
  ~/data/test/
+ Trained model
  ~/weights/ADC.model
+ GPU:
  cd ./path~
  python hw2_inference.py --dataset [linktodataset] --out [foldercontainmode] --unit [outputclass] --gpu True
+ CPU:
  cd ./path~
  python hw2_inference.py --dataset [linktodataset] --out [foldercontainmode] --unit [outputclass] 
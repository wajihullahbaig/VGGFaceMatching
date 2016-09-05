# VGGFaceMatching

This my attempt at learning how the VGG-16 deep learning feature extraction works on a small dataset.
I have found numerous sources including the the official version of VGG, claiming that features extracted
from the 16 layer deep neural network are good enough for face verification.

Though this project grew out of frustration as I could not find a decent test that would show the accuracy of 
the features on a small dataset.

Please note that I have followed the FVC protocol (Fingerprint Verfication Competition) to produce my results.
The LFW dataset was prunned and reduced to a 100 individual set. Each individual containing 8 images of the same 
person. This is important as I had to follow the FVC protocol.

I have used the HeadHunter algorithm to detect, crop and resize the face images in the LFW dataset. Let me say
that the algorithm is extremely impressive.

Caffe deep learning library is used to extract VGG-16 features. The last classification layer is removed to
leave a 4096 dimensional vector describing an image.

You can see the References section to read more about caffe,vgg,face detect etc. 


## Running the code
	
Make sure you have correct paths for

	1. LFW 224x224 folder
	2. Outputs folder
	3. EER-FVCProtocol folder
	
From FeatureExtraction run Main.py file. This will run untill all the 800 images from the lfw 224x224.
The outcoming feature vectors must be saved in *Outputs* folder. 
Then run GenuineMatcher.py and ImposterMatcher.py from EER-FVC folder.
This will result in genuines.npy and imposters.npy 

Finally run EERCalculation and wait for the output graphs and watch the console for EER.
	

## Matching Results

| EER (100%)	|
| ------------- |:------------------------------:|
| 6.4%      	| Without mean image subtraction |
| 5.75%         | With mean image subtraction    |

	Total Genuine Matches = 2800
	Total Imposter Matches = 4950

![alt tag](https://github.com/wajihullahbaig/VGGFaceMatching/blob/master/ScreenShots/genuin-imposter-distribution.jpg)
![alt tag](https://github.com/wajihullahbaig/VGGFaceMatching/blob/master/ScreenShots/threshold.jpg)
![alt tag](https://github.com/wajihullahbaig/VGGFaceMatching/blob/master/ScreenShots/roc.jpg)

	*A genuine match is as follows*
	   1_1 vs 1_2
	   1_1 vs 1_2
	   ...
	   1_1 vs 1_8
   
	*An imposter match is as follow*
	   1_1 vs 2_1
	   1_1 vs 3_1
	   ...
	   99_1 vs 100_1
   
Note that LFW dataset does not have images named 1_1.jpg, rather as firstname_lastname_0001.jpg. The code takes care of following the FVC
matching protocol. This was possible because LFW data has images of individuals with 8 or images.

   
Caffe researchers claim to have better accuracy on mean image substraction. In my tests, I have subtracted mean of the image from itself to produce
better accuracies. The claim seems to hold in my tests.

##  Point of interest
During face extraction, Hugh Grant folder ended with a image of Sandra Bullock. This was face detection output from an image where Hugh Grant and Sandra Bullock
were together. So I removed Sandra Bullock's image and replaced it with that of Hugh Grant's.


## References
		http://vis-www.cs.umass.edu/lfw/
		http://markusmathias.bitbucket.org/2014_eccv_face_detection/
		http://www.vlfeat.org/matconvnet/pretrained/
		http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz
		https://github.com/musicfish1973/vgg_face_matconvnet
		https://github.com/PatienceKai/VGG_Face_Caffe_Model
		https://github.com/eglxiang/vgg_face
		https://github.com/jesu9/VGGFeatExtract

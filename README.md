# VGGFaceMatching

This my attempt at learning how the VGG-16 deep learning feature extraction works on a small dataset.
I have found numerous sources including the the official version of VGG, claiming that features extracted
from the 16 layer deep neural network are good enough for face vecirification.

Though this project grew out of frustration as I could not find a decent test that would show the accuracy of 
the features on a small dataset.

Please note that I have followed the FVC protocol (Fingerprint Verfication Competition) to produce my results.
The LFW dataset was prunned and reduced to a 100 individual set. Each individual containing 8 images of the same 
person. This is important as I had to follow the FVC protocol.

I have used the HeadHunter algorithm to detect, crop and resize the face images in the LFW dataset. Let me say
that the algorithm is extremely impressive. 


## Running the code
	

You can see the References section to read more about caffe,vgg,face detect etc.


## References

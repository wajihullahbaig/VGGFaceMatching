import numpy as np
import caffe

class VGGExtractor(caffe.Net):
		
	def __init__(self,modelFile,pretrainedFile):
		caffe.Net.__init__(self,modelFile, pretrainedFile, caffe.TEST)
		caffe.set_mode_cpu()
	def GetFeature(self,inputImage,blobs=['fc7']):
		feats = {}
		for blob in blobs:
			feats[blob] = []
		inputImage[:,:,0] = inputImage[:,:,0] - np.mean(inputImage[:,:,0])
		inputImage[:,:,1] = inputImage[:,:,1] - np.mean(inputImage[:,:,1])
		inputImage[:,:,2] = inputImage[:,:,2] - np.mean(inputImage[:,:,2])
		inputImage = inputImage.transpose((2,0,1)) # RGB to BGR
		inputImage = inputImage*255		
		inputImage = inputImage[None,:] # add singleton dimension			
		out = self.forward_all(**{self.inputs[0]: inputImage, 'blobs': blobs})  			
		for blob in blobs:
			feat = out[blob]
			feats[blob].append(feat.flatten())
		return feats
			

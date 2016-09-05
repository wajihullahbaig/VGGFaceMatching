from VGGFeatureExtractor import VGGExtractor
import os
import skimage
import numpy as np
import caffe
if __name__ == '__main__':
	model_file = '/media/wajih/Disk1 500 GB/Onus/RnD/DataSet/caffe models/VGG_FACE_deploy.prototxt'
  	pretrained_file = '/media/wajih/Disk1 500 GB/Onus/RnD/DataSet/caffe models/VGG_FACE.caffemodel'
  	outputPath = '/media/wajih/Disk1 500 GB/Onus/RnD/Dev/Deep Learning/EclipseWorkSpace/VGGFaceRecognitionTest/Outputs/'
  	vggExtractor = VGGExtractor(model_file,pretrained_file)
  	lfw_224x244Path = '/media/wajih/Disk1 500 GB/Onus/RnD/DataSet/face images/lfw_home/lfw 224x224/'
  	folderList = os.listdir(lfw_224x244Path)
  	fileCount = 0;
  	for i in range(0,len(folderList)):
  		fullPath = lfw_224x244Path +folderList[i]
  		fileList = os.listdir(fullPath)
  		for j in range(0,len(fileList)):
  			img = caffe.io.load_image(fullPath+'/'+fileList[j])	 	
  	 	  	feature = vggExtractor.GetFeature(img)
  	 	  	fileName, fileExtension = os.path.splitext(fileList[j])
  	 	  	np.save(outputPath+fileName+'.npy',feature )
  	 	  	print 'Processed file:'+fileList[j]
  	 	  	fileCount +=1  	 	  	
  	 	print 'Processed:' + folderList[i] 	
  	 	print 'Total files processed:' + str(fileCount)
  	 	
  	print 'Processing complete...'
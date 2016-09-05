import numpy as np
import matplotlib.pyplot as plt
from bokeh.models.annotations import Legend
outputPath = '../EER-FVCProtocol/'

def ClassSeperability(genuines,imposters):
    maxVal = np.max(genuines +imposters )
    minVal = np.min(genuines + imposters)
    dIndex = DecidabilityIndex(genuines,imposters)
    
    step = 0.03*(maxVal-minVal)
    binSize = np.int(np.ceil((maxVal-minVal)/step))
    count = 0                   
    g = np.zeros((len(genuines),1),dtype = np.float64)
    g[:,0] = genuines[:]
    distGenuines,bins = np.histogram(g,binSize)
    bins = np.expand_dims(bins, axis=1)
    distGenuines = np.expand_dims(distGenuines, axis=1)
    i = np.zeros((len(imposters),1),dtype = np.float64)
    i[:,0] = imposters[:]    
    distImposters,bins = np.histogram(i,binSize)
    bins = np.expand_dims(bins, axis=1)
    distImposters = np.expand_dims(distImposters, axis=1)
    plt.plot(bins[0:binSize,0],distGenuines,color='blue', label = 'Genuine')
    plt.plot(bins[0:binSize,0],distImposters,color='red',label = 'Imposter')
    plt.legend(loc = 'upper right')
    plt.xlabel('Matching Score')
    plt.ylabel('Occurance')
    plt.title('Genuine and Imposter Distribution')
    plt.figure(1)    
    plt.draw()
    
    nGen = len(genuines)
    nImp = len(imposters)    
    npt = 10*(nGen+nImp)
    v = [i for i in range(0, npt)]
    t = (maxVal + (minVal-maxVal)* np.true_divide(v,npt)).T
    t = t[None,:]
    t = np.fliplr(t)
    fmr = np.ones((npt,1),dtype = np.float64)
    fnmr = np.ones((npt,1),dtype = np.float64)
    
    for k in range(0,npt):
        fmr[k,0] = np.true_divide(len(np.where(imposters >= t[0,k])[0]),nImp)
        fnmr[k,0] = np.true_divide(len(np.where(genuines <= t[0,k])[0]),nGen)
    
    fmr = fmr * 100
    fnmr = fnmr * 100
    
    eer = abs(fmr-fnmr)
    mindex = np.argmin(eer)
    eer1 = fmr[mindex,0]
    eer2 = fnmr[mindex,0]
    eer = (eer1+eer2)/2.0
    
    plt.figure(2)
    plt.plot(t[0,:],fmr[:,0],color='blue')
    plt.plot(t[0,:],fnmr[:,0],color='red')
    plt.title('Threshold vs FMR & FNMR')
    plt.xlabel('Threshold')
    plt.ylabel('Percentage')
    plt.draw()
    
    plt.figure(3)
    plt.plot(fmr,fnmr,color='blue')
    plt.title('Receiver Operating Curve (ROC)')
    plt.xlabel('False Match Rate')
    plt.ylabel('False Non-Match Rate')
    plt.draw()
    
    print 'EER = ' + str(eer) + ' dIndex = ' + str(dIndex)
    plt.show()
          

def DecidabilityIndex(genuines,imposters):
    meanGenuines = np.mean(genuines)
    meanImposters = np.mean(imposters)
    
    varGenuines = np.var(genuines)
    varImposters = np.var(imposters)
    
    num = np.abs(meanGenuines-meanImposters)
    den = np.sqrt((varGenuines+varImposters)/2.0)
    
    return (num/den)

if __name__ == '__main__':
    
    gens = np.load(outputPath+'genuines.npy')
    imps = np.load(outputPath+'imposters.npy')
    g = gens.item().values()
    i = imps.item().values()

    ClassSeperability(g,i)



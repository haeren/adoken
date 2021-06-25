import numpy as np
import keras
import nibabel as nib

class customMriGenerator(keras.utils.Sequence):

    def __init__(self, mriFilePaths, labels, batchSize):
        self.mriFilePaths = mriFilePaths
        self.labels = labels
        self.batchSize = batchSize

    def __len__(self):
        return (np.ceil(len(self.mriFilePaths)/float(self.batchSize))).astype(np.int)

    def __getitem__(self, idx):
        batchX = self.mriFilePaths[idx*self.batchSize : (idx+1)*self.batchSize]
        batchY = self.labels[idx*self.batchSize : (idx+1)*self.batchSize]

        mriBatch = []

        for filePath in batchX:
            img = nib.load(filePath)
            imgData = img.get_fdata()
            imgDataArr = np.asarray(imgData)
            mriBatch.append(imgDataArr)
        
        mriBatch = np.array(mriBatch)

        rows, cols, depth = mriBatch[0].shape
        channel = 1
    
        mriBatch = mriBatch.reshape(-1, rows, cols, depth, channel)

        return mriBatch, np.array(batchY)

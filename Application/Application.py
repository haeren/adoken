from PyQt5 import QtWidgets
import sys
from MainWindow import Ui_MainWindow
from PyQt5.QtGui import QIcon, QPixmap, QTextCursor
from pathlib import PurePath, Path
import splitfolders as sf
import pandas as pd
import os
from arff2pandas import a2p
from sklearn.model_selection import train_test_split
from Model import myConv3d, myConv2dLstm
import nibabel as nib
from deepbrain import Extractor
import intensity_normalization as ino
import numpy as np
from keras.models import model_from_json, load_model
from CustomGenerator import customMriGenerator
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from nilearn.image import resample_img
from sklearn.metrics import confusion_matrix, classification_report
import csv
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='invalid value encountered in double_scalars')
warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice')
warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')
warnings.filterwarnings('ignore', message='invalid value encountered in double_scalars')
warnings.filterwarnings('ignore', message='NaNs or infinite values are present in the data passed to resample')
warnings.filterwarnings('ignore', category=FutureWarning)

class myApp(QtWidgets.QMainWindow):

    def __init__(self):
        self.info = ''
        self.model = None
        self.trainedModel = None
        self.dataTypeList = ['MRI', 'Feature Vector']
        self.modelArchList = ['Conv3D', 'Conv2D + LSTM']
        self.paddingOptions = ['valid', 'same']
        self.activationOptions = ['relu', 'sigmoid', 'tanh']
        self.preprocessMriPathOptions = ['Select', 'Train', 'Validation', 'Test']
        self.resampleOptions = ['No Resample', 'Scale', 'Target Shape']
        self.preprocessPathValid = False
        self.optimizerOptions = ['adam', 'adadelta']
        self.lossOptions = ['categorical_crossentropy']
        self.metricsOptions = ['accuracy']
        self.modelCompiled = False
        self.modelParamLoaded = False
        self.labelEncoder = LabelEncoder()
        self.xTrainFilePaths = []
        self.xValFilePaths = []
        self.yTrain = []
        self.yVal = []
        self.yTrainEncoded = None
        self.yValEncoded = None
        self.isLabelEncoded = False
        self.isModelTrained = False
        self.dataType = self.dataTypeList[0]
        
        super(myApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setWindowTitle('ADOKEN')
        self.setWindowIcon(QIcon('icon.png'))
        self.setFixedSize(self.size())

        # Info Box
        self.ui.teInfo.setReadOnly(True)
        self.ui.teInfo.textChanged.connect(self.infoBoxTextChanged)

        # Select Data Tab
        self.ui.cbDataType.addItems(self.dataTypeList)
        self.ui.cbDataType.currentTextChanged.connect(self.dataTypeChanged)

        self.ui.btnSelectDataSet.clicked.connect(self.selectDataPath)
        self.ui.btnSelectTrainSet.clicked.connect(self.selectDataPath)
        self.ui.btnSelectValSet.clicked.connect(self.selectDataPath)
        self.ui.btnSelectTestSet.clicked.connect(self.selectDataPath)
        self.ui.btnSplitDataSet.clicked.connect(self.splitDataSet)

        self.ui.leDataSetPath.setReadOnly(True)
        self.ui.leTrainSetPath.setReadOnly(True)
        self.ui.leValSetPath.setReadOnly(True)
        self.ui.leTestSetPath.setReadOnly(True)

        self.ui.dsbTrainRatio.setDecimals(1)
        self.ui.dsbValRatio.setDecimals(1)
        self.ui.dsbTestRatio.setDecimals(1)

        self.ui.dsbTrainRatio.setSingleStep(0.1)
        self.ui.dsbValRatio.setSingleStep(0.1)
        self.ui.dsbTestRatio.setSingleStep(0.1)

        self.ui.dsbTrainRatio.setMaximum(1)
        self.ui.dsbValRatio.setMaximum(1)
        self.ui.dsbTestRatio.setMaximum(1)

        self.ui.dsbTrainRatio.setValue(0.8)
        self.ui.dsbValRatio.setValue(0.1)
        self.ui.dsbTestRatio.setValue(0.1)

        self.ui.leTrainSetPath.textChanged.connect(self.trainPathChanged)
        self.ui.leValSetPath.textChanged.connect(self.valPathChanged)
        self.ui.leTestSetPath.textChanged.connect(self.testPathChanged)

        # Pre-process Tab
        self.ui.cbPreprocessMriPath.addItems(self.preprocessMriPathOptions)
        self.ui.cbPreprocessMriPath.currentTextChanged.connect(self.preprocessMriPathChanged)

        self.ui.cbResampleOption.addItems(self.resampleOptions)
        self.ui.cbResampleOption.currentTextChanged.connect(self.resampleOptionChanged)

        self.ui.btnPreprocessMri.clicked.connect(self.preprocessMri)

        self.ui.dsbResampleAffine.setDecimals(1)
        self.ui.dsbResampleAffine.setSingleStep(0.1)
        self.ui.dsbResampleAffine.setValue(4)

        self.ui.dsbResampleAffine.setMinimum(0.1)
        self.ui.sbResampleRows.setMinimum(100)
        self.ui.sbResampleCols.setMinimum(150)
        self.ui.sbResampleDepth.setMinimum(100)

        self.ui.dsbResampleAffine.setMaximum(10)
        self.ui.sbResampleRows.setMaximum(500)
        self.ui.sbResampleCols.setMaximum(500)
        self.ui.sbResampleDepth.setMaximum(500)

        self.ui.dsbResampleAffine.setEnabled(False)
        self.ui.sbResampleRows.setEnabled(False)
        self.ui.sbResampleCols.setEnabled(False)
        self.ui.sbResampleDepth.setEnabled(False)

        self.ui.tabPreprocess.setTabEnabled(1, False)
        self.ui.tabPreprocess.setTabToolTip(1, 'This feature is under construction and will be added in upcoming version.')

        # Create Model Tab
        self.ui.cbModelArch.addItems(self.modelArchList)
        self.ui.cbModelArch.currentTextChanged.connect(self.modelArchChanged)
        
        self.ui.cbPadding.addItems(self.paddingOptions)
        self.ui.cbActivation.addItems(self.activationOptions)
        self.ui.btnCreateModel.clicked.connect(self.createModel)
        self.ui.btnSaveModelParam.clicked.connect(self.saveModelParameters)

        self.ui.sbLstmUnit.setVisible(False)

        self.ui.sbRows.setMinimum(1)
        self.ui.sbCols.setMinimum(1)
        self.ui.sbDepth.setMinimum(1)
        self.ui.sbFilters.setMinimum(1)
        self.ui.sbKernelSize.setMinimum(1)
        self.ui.sbLstmUnit.setMinimum(1)
        self.ui.sbPoolSize.setMinimum(1)
        self.ui.sbClasses.setMinimum(1)

        self.ui.sbRows.setMaximum(500)
        self.ui.sbCols.setMaximum(500)
        self.ui.sbDepth.setMaximum(500)
        self.ui.sbLstmUnit.setMaximum(500)
        self.ui.dsbDropout.setMaximum(0.5)

        self.ui.dsbDropout.setSingleStep(0.01)

        # Train Model Tab
        self.ui.btnLoadModelParam.clicked.connect(self.loadModelParameters)

        self.ui.cbOptimizer.addItems(self.optimizerOptions)
        self.ui.cbLoss.addItems(self.lossOptions)
        self.ui.cbMetrics.addItems(self.metricsOptions)

        self.ui.sbBatchSize.setMinimum(1)
        self.ui.sbEpoch.setMinimum(1)

        self.ui.btnCompileModel.clicked.connect(self.compileModel)
        self.ui.btnEncodeTrainLabels.clicked.connect(self.encodeTrainLabels)
        self.ui.btnFitModel.clicked.connect(self.fitModel)
        self.ui.btnSaveTrainedModel.clicked.connect(self.saveTrainedModel)

        # Test Model Tab
        self.ui.btnLoadTrainedModel.clicked.connect(self.loadTrainedModel)
        self.ui.btnLoadEncodeInfo.clicked.connect(self.loadEncodeInfo)
        self.ui.btnPredict.clicked.connect(self.predict)

    # Info Box
    def infoBoxTextChanged(self):
        self.ui.teInfo.moveCursor(QTextCursor.End)

    # Select Data Tab
    def dataTypeChanged(self):
        self.dataType = self.ui.cbDataType.currentText()
        self.ui.teInfo.append(f'Data type is changed to {self.dataType}')

        self.ui.leDataSetPath.setText('')
        self.ui.leTrainSetPath.setText('')
        self.ui.leValSetPath.setText('')
        self.ui.leTestSetPath.setText('')

        tabIndexPreprocess = self.dataTypeList.index(self.dataType)
        #self.ui.tabPreprocess.setCurrentIndex(tabIndexPreprocess)

    def selectDataPath(self):
        sender = self.sender()
        dialog = QtWidgets.QFileDialog()

        if self.dataType == 'MRI':            
            path = dialog.getExistingDirectory(self, "Select Folder")
        elif self.dataType == 'Feature Vector':
            path = dialog.getOpenFileName(self, 'Select File', filter='(*.csv *.arff *.xlsx)')[0]

        #objectName() can be used instead of text()
        if sender.text() == 'Browse Data Set':
            self.ui.leDataSetPath.setText(path)
        elif sender.text() == 'Browse Train Set':
            self.ui.leTrainSetPath.setText(path)
        elif sender.text() == 'Browse Validation Set':
            self.ui.leValSetPath.setText(path)
        elif sender.text() == 'Browse Test Set':
            self.ui.leTestSetPath.setText(path)

    def splitDataSet(self):
        trainRatio = self.ui.dsbTrainRatio.value()
        valRatio = self.ui.dsbValRatio.value()
        testRatio = self.ui.dsbTestRatio.value()

        if self.ui.leDataSetPath.text() == '':
            self.ui.teInfo.append('Please select path of the data set')
            return

        if trainRatio + valRatio + testRatio != 1.0:
            self.ui.teInfo.append('Sum of the ratios must be equal to 1')
            return

        shuffleChk = False
        shuffleSeed = 1337

        if self.ui.chkShuffle.isChecked():
            shuffleChk = True
            shuffleSeed = None

        if self.dataType == 'MRI':        
            path = PurePath(self.ui.leDataSetPath.text())
            folderName = path.name
            parentFolder = path.parent
            outputPath = parentFolder.joinpath(folderName + '_split')
            ratios = (trainRatio, valRatio, testRatio)

            sf.ratio(path, output=outputPath, seed=shuffleSeed, ratio=ratios)            
            self.ui.leTrainSetPath.setText(str(outputPath.joinpath('train')))
            self.ui.leValSetPath.setText(str(outputPath.joinpath('val')))
            self.ui.leTestSetPath.setText(str(outputPath.joinpath('test')))
            self.ui.teInfo.append('Split completed')

        if self.dataType == 'Feature Vector':
            path = PurePath(self.ui.leDataSetPath.text())
            dataPath = os.path.splitext(path)[0]
            dataExtension = os.path.splitext(path)[1]
            dataFrame = None

            if dataExtension == '.csv':
                dataFrame = pd.read_csv(path)

            elif dataExtension == '.arff':
                with open(path) as df:
                    dataFrame = a2p.load(df)

            elif dataExtension == '.xlsx':
                dataFrame = pd.read_excel(path)

            dfTrain, dfValTest = train_test_split(dataFrame, test_size=(valRatio+testRatio), shuffle=shuffleChk)
            dfVal, dfTest = train_test_split(dfValTest, test_size=testRatio/(valRatio+testRatio), shuffle=shuffleChk)

            dfTrain.to_csv(dataPath + '_train.csv')
            dfVal.to_csv(dataPath + '_val.csv')
            dfTest.to_csv(dataPath + '_test.csv')

    # Pre-process Tab
    def preprocessMriPathChanged(self):
        self.ui.lblVisualizeMri.clear()
        path = ''
        pathIndex = self.ui.cbPreprocessMriPath.currentIndex()
        
        if pathIndex == 0:
            path = ''
        elif pathIndex == 1:
            path = self.ui.leTrainSetPath.text()
        elif pathIndex == 2:
            path = self.ui.leValSetPath.text()
        elif pathIndex == 3:
            path = self.ui.leTestSetPath.text()

        filePath = self.findFirstFile(path ,'.nii')
        
        if filePath == '':
            self.preprocessPathValid = False
            self.ui.teInfo.append('Please select a valid path')
        else:
            self.preprocessPathValid = True
            self.updateDisplayedMri(filePath)

    def resampleOptionChanged(self):
        option = self.ui.cbResampleOption.currentText()
        
        if option == 'No Resample': # self.resampleOptions[] can be used
            self.ui.dsbResampleAffine.setEnabled(False)
            self.ui.sbResampleRows.setEnabled(False)
            self.ui.sbResampleCols.setEnabled(False)
            self.ui.sbResampleDepth.setEnabled(False)
        elif option == 'Scale':
            self.ui.dsbResampleAffine.setEnabled(True)
            self.ui.sbResampleRows.setEnabled(False)
            self.ui.sbResampleCols.setEnabled(False)
            self.ui.sbResampleDepth.setEnabled(False)
        elif option == 'Target Shape':
            self.ui.dsbResampleAffine.setEnabled(False)
            self.ui.sbResampleRows.setEnabled(True)
            self.ui.sbResampleCols.setEnabled(True)
            self.ui.sbResampleDepth.setEnabled(True)

    def preprocessMri(self):
        if self.preprocessPathValid == False:
            self.ui.teInfo.append('Please select a valid path')
            return

        skullStripSelected = self.ui.chkSkullStripping.isChecked()
        normalizationSelected = self.ui.chkNormalizationMri.isChecked()

        if (self.ui.cbResampleOption.currentText() == 'No Resample') and (not skullStripSelected) and (not normalizationSelected):
            self.ui.teInfo.append('Select a pre-process method')
            return

        ext = Extractor()
        path = self.ui.leTrainSetPath.text()
        pathIndex = self.ui.cbPreprocessMriPath.currentIndex()
        outputFolderPath = None

        affineScale = self.ui.dsbResampleAffine.value()
        targetRows = self.ui.sbResampleRows.value()
        targetCols = self.ui.sbResampleCols.value()
        targetDepth = self.ui.sbResampleDepth.value()
        targetShape = np.array((targetRows, targetCols, targetDepth))
        newResolution = [2,]*3
        newAffine = np.zeros((4,4))
        newAffine[:3,:3] = np.diag(newResolution)
        newAffine[:3,3] = targetShape*newResolution/2.*-1
        newAffine[3,3] = 1.

        if pathIndex == 2:
            path = self.ui.leValSetPath.text()
        elif pathIndex == 3:
            path = self.ui.leTestSetPath.text()

        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.nii'):
                    filePath = os.path.join(root, file)
                    niImg = nib.load(filePath)
                    func = nib.load(filePath)
                    rows, cols, depth = niImg.shape

                    if skullStripSelected:
                        output = np.zeros((rows,cols,depth))
                        niImg = niImg.get_fdata()
                        prob = ext.run(niImg) 
                        mask = prob > 0.5

                        for i in range(rows):
                            for j in range(cols):
                                for k in range(depth):
                                    if(mask[i,j,k] == True):
                                        output[i,j,k] = niImg[i,j,k]

                        niImg = nib.Nifti1Image(output, func.affine, func.header)

                    if self.ui.cbResampleOption.currentText() == 'Scale':
                        niImg = resample_img(niImg, target_affine=np.eye(3)*affineScale, interpolation='nearest')
                        niImg = nib.Nifti1Image(niImg.get_fdata(), niImg.affine, func.header)

                    elif self.ui.cbResampleOption.currentText() == 'Target Shape':
                        niImg = resample_img(niImg, target_affine=newAffine, target_shape=targetShape, interpolation='nearest')
                        niImg = nib.Nifti1Image(niImg.get_fdata(), newAffine, func.header)

                    if normalizationSelected:
                        out = ino.normalize.zscore.zscore_normalize(niImg)
                        niImg = nib.Nifti1Image(out.get_fdata(), out.affine, func.header)

                    cwd = os.getcwd()
                    pathParts = Path(root).parts
                    labelName = pathParts[-1]
                    setName = pathParts[-2]
                    outputFolderPath = os.path.join(cwd, 'files', 'preprocessed', setName, labelName)
                    Path(outputFolderPath).mkdir(parents=True, exist_ok=True)         
                    #mainFolder = Path(root).parent.parent
                    #fileNameWoExt = os.path.splitext(file)[0]
                    try:
                        nib.save(niImg, os.path.join(outputFolderPath, file))
                        print('Preprocess completed for:', file)
                    except:
                        self.ui.teInfo.append('Something went wrong when trying to save file:', file)
                    
        self.ui.teInfo.append('Preprocess completed. Path:')
        outputFolderPath = Path(outputFolderPath).parent
        outputFolderPath = str(outputFolderPath)
        self.ui.teInfo.append(outputFolderPath)

        if pathIndex == 1:
            self.ui.leTrainSetPath.setText(outputFolderPath)
            self.trainPathChanged()
        elif pathIndex == 2:
            self.ui.leValSetPath.setText(outputFolderPath)
            self.valPathChanged()
        elif pathIndex == 3:
            self.ui.leTestSetPath.setText(outputFolderPath)
        preprocessedFilePath = self.findFirstFile(outputFolderPath, '.nii')
        self.updateDisplayedMri(preprocessedFilePath)
        self.ui.teInfo.append('Selected path is updated.')

    def modelArchChanged(self):
        modelArch = self.ui.cbModelArch.currentText()
        if modelArch == 'Conv3D':
            self.ui.sbLstmUnit.setVisible(False)
        elif modelArch == 'Conv2D + LSTM':
            self.ui.sbLstmUnit.setVisible(True)

    def createModel(self):
        model = None
        modelArch = self.ui.cbModelArch.currentText()
        
        rows = self.ui.sbRows.value()
        cols = self.ui.sbCols.value()
        depth = self.ui.sbDepth.value()
        filters = self.ui.sbFilters.value()
        kernelSize = self.ui.sbKernelSize.value()
        padding = self.ui.cbPadding.currentText()
        lstmUnits = self.ui.sbLstmUnit.value()
        activation = self.ui.cbActivation.currentText()
        poolSize = self.ui.sbPoolSize.value()
        dropout = self.ui.sbPoolSize.value()
        numClasses = self.ui.sbClasses.value()

        try: # Try-except can be removed
            if modelArch == 'Conv3D':
                inputShape = (rows, cols, depth, 1)
                model = myConv3d(inputShape, numClasses, filters, kernelSize, padding, activation, poolSize, dropout)
            elif modelArch == 'Conv2D + LSTM':
                inputShape = (rows, cols, depth, 1)
                model = myConv2dLstm(inputShape, numClasses, filters, kernelSize, padding, lstmUnits, activation, poolSize, dropout)

            self.model = model

            modelSumStrList = []
            model.summary(print_fn=lambda x: modelSumStrList.append(x))
            modelSumStr = '\n'.join(modelSumStrList)

            self.ui.tbModelSummary.setText(modelSumStr)
            self.ui.tbModelInfo.setText(modelSumStr)

            self.modelParamLoaded = True
            self.modelCompiled = False
            self.isModelTrained = False
            self.ui.teInfo.append('Model created successfully')
        
        except:
            self.ui.teInfo.append('Please enter valid model parameters')
 
    def findFirstFile(self, folderPath, fileExtension):
        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.endswith(fileExtension):
                    return os.path.join(root, file)
        return ''

    def updateDisplayedMri(self, filePath):
        loadedFile = nib.load(filePath).get_fdata()
        imgArray = np.asarray(loadedFile)
        rows, cols, depth = imgArray.shape
        imgSlice = imgArray[:,:,depth//2]
        plt.imshow(imgSlice, cmap='gray')

        cwd = os.getcwd()
        outputFolderPath = os.path.join(cwd, 'files')
        Path(outputFolderPath).mkdir(parents=True, exist_ok=True)
        outputPath = os.path.join(outputFolderPath, 'mriSlice.png')
        plt.savefig(outputPath)
        pixmap = QPixmap(outputPath)
        self.ui.lblVisualizeMri.setPixmap(pixmap)
        #if os.path.exists(outputPath):
            #os.remove(outputPath)

    def saveModelParameters(self):
        cwd = os.getcwd()
        outputFolderPath = os.path.join(cwd, 'files')
        Path(outputFolderPath).mkdir(parents=True, exist_ok=True)

        try:
            jsonFile = open(os.path.join(outputFolderPath,'modelParameters.json'), 'w+')
            modelParameters = self.model.to_json()
            jsonFile.write(modelParameters)
            jsonFile.close()
            self.ui.teInfo.append('Model parameters were saved to JSON file successfully')
        
        except:
            self.ui.teInfo.append('There are no model parameters to save!')

    def loadModelParameters(self):
        dialog = QtWidgets.QFileDialog()
        path = dialog.getOpenFileName(self, 'Select File', filter='(*.json)')[0]
        
        try:
            jsonFile = open(path, 'r')
            loadedModelParameters = jsonFile.read()
            jsonFile.close()
            self.model = model_from_json(loadedModelParameters)
            modelSumStrList = []
            self.model.summary(print_fn=lambda x: modelSumStrList.append(x))
            modelSumStr = '\n'.join(modelSumStrList)
            self.ui.tbModelInfo.setText(modelSumStr)

            self.modelParamLoaded = True
            self.modelCompiled = False
            self.isModelTrained = False
            self.ui.teInfo.append('Model parameters are loaded from JSON file successfully')

        except:
            self.ui.teInfo.append('Unable to load model from JSON file')

    def compileModel(self):
        if self.modelParamLoaded == True:
            optimizer = self.ui.cbOptimizer.currentText()
            loss = self.ui.cbLoss.currentText()
            metrics = self.ui.cbMetrics.currentText()
            self.model.compile(optimizer = optimizer, loss = loss, metrics=[metrics])
            self.modelCompiled = True
            self.ui.teInfo.append('Model is compiled successfully')

        else:
            self.ui.teInfo.append('Load model parameters to compile!')

    def encodeTrainLabels(self):
        trainPath = self.ui.leTrainSetPath.text()
        valPath = self.ui.leValSetPath.text()

        self.xTrainFilePaths.clear()
        self.xValFilePaths.clear()
        self.yTrain.clear()
        self.yVal.clear()

        for root, dirs, files in os.walk(trainPath):
            for file in files:
                if file.endswith('.nii'):
                    filePath = os.path.join(root, file)
                    self.xTrainFilePaths.append(filePath)
                    pathParts = Path(root).parts
                    labelName = pathParts[-1]
                    self.yTrain.append(labelName)

        for root, dirs, files in os.walk(valPath):
            for file in files:
                if file.endswith('.nii'):
                    filePath = os.path.join(root, file)
                    self.xValFilePaths.append(filePath)
                    pathParts = Path(root).parts
                    labelName = pathParts[-1]
                    self.yVal.append(labelName)

        #X_train, y_train = shuffle(X_train, y_train)

        if not self.yTrain:
            self.ui.teInfo.append('Could not find any .nii file in train path')
            return

        if not self.yVal:
            self.ui.teInfo.append('Could not find any .nii file in validation path')
            return

        self.yTrainEncoded = self.labelEncoder.fit_transform(self.yTrain)
        self.yValEncoded = self.labelEncoder.transform(self.yVal)
        labelList = self.labelEncoder.classes_
        numClasses = len(labelList)
        
        self.yTrainEncoded = to_categorical(self.yTrainEncoded, num_classes = numClasses)
        self.yValEncoded = to_categorical(self.yValEncoded, num_classes = numClasses)

        self.isLabelEncoded = True
        cwd = os.getcwd()
        outputFolderPath = os.path.join(cwd, 'files')
        Path(outputFolderPath).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(outputFolderPath,'encodeInfo.npy'), self.labelEncoder.classes_)

        textFile = open(os.path.join(outputFolderPath,'encodeInfo.txt'), 'w+')
        content = str(self.labelEncoder.classes_)
        textFile.write(content)
        textFile.close()

        self.ui.teInfo.append('Encode completed and class info is saved.')
        
    def fitModel(self):
        if self.modelCompiled == True and self.isLabelEncoded == True:
            batchSize = self.ui.sbBatchSize.value()
            epoch = self.ui.sbEpoch.value()
            trainSize = len(self.yTrain)
            valSize = len(self.yVal)

            # Callbacks
            cwd = os.getcwd()
            outputFolderPath = os.path.join(cwd, 'files')
            Path(outputFolderPath).mkdir(parents=True, exist_ok=True)
            checkpointCb = ModelCheckpoint(os.path.join(outputFolderPath, 'modelCheckpoint.h5'), monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
            earlyStoppingCb = EarlyStopping(monitor='val_accuracy', patience=15, mode='max')
            callbacksList = [checkpointCb, earlyStoppingCb]

            myTrainBatchGen = customMriGenerator(self.xTrainFilePaths, self.yTrainEncoded, batchSize)
            myValBatchGen = customMriGenerator(self.xValFilePaths, self.yValEncoded, batchSize)

            try:
                self.model.fit_generator(generator=myTrainBatchGen,
                steps_per_epoch = int(trainSize // batchSize),
                epochs = epoch,
                verbose = 1,
                validation_data = myValBatchGen,
                validation_steps = int(valSize // batchSize),
                callbacks = callbacksList)

                modelSumStrList = []
                self.model.summary(print_fn=lambda x: modelSumStrList.append(x))
                modelSumStr = '\n'.join(modelSumStrList)
                self.ui.tbTrainedModelInfo.setText(modelSumStr)

                self.isModelTrained = True
                self.ui.teInfo.append('Training is completed successfully.')

            except:
                self.ui.teInfo.append('Something went wrong with the training.')

        else:
            self.ui.teInfo.append('Compile model and encode label to fit model!')

    def saveTrainedModel(self):
        if self.isModelTrained == True:
            cwd = os.getcwd()
            outputFolderPath = os.path.join(cwd, 'files')
            Path(outputFolderPath).mkdir(parents=True, exist_ok=True)
            try:
                self.model.save(os.path.join(outputFolderPath,'trainedModel.h5'))
                self.ui.teInfo.append('Trained model is saved successfully.')
            except:
                self.ui.teInfo.append('There is no trained model to save!')
        else:
            self.ui.teInfo.append('Model must be trained in order to save it!')

    def trainPathChanged(self):
        self.isLabelEncoded = False

        trainPath = self.ui.leTrainSetPath.text()
        rows, cols, depth = 1, 1, 1

        for root, dirs, files in os.walk(trainPath):
            for file in files:
                if file.endswith('.nii'):
                    filePath = os.path.join(root, file)
                    img = nib.load(filePath).get_fdata()
                    rows, cols, depth = img.shape
                    break

        self.ui.sbRows.setValue(rows)
        self.ui.sbCols.setValue(cols)
        self.ui.sbDepth.setValue(depth)

    def valPathChanged(self):
        self.isLabelEncoded = False

    def testPathChanged(self):
        pass # May be required in later versions

    def loadTrainedModel(self):
        dialog = QtWidgets.QFileDialog()
        path = dialog.getOpenFileName(self, 'Select File', filter='(*.h5)')[0]
        
        try:
            self.model = load_model(path)
            
            modelSumStrList = []
            self.model.summary(print_fn=lambda x: modelSumStrList.append(x))
            modelSumStr = '\n'.join(modelSumStrList)
            self.ui.tbTrainedModelInfo.setText(modelSumStr)
            self.ui.tbModelInfo.setText(modelSumStr)

            self.modelParamLoaded = True
            self.modelCompiled = True
            self.isModelTrained = True
            self.ui.teInfo.append('Trained model is loaded from H5 file successfully')

        except:
            self.ui.teInfo.append('Unable to load trained model from H5 file!')
   
    def loadEncodeInfo(self):
        dialog = QtWidgets.QFileDialog()
        path = dialog.getOpenFileName(self, 'Select File', filter='(*.npy)')[0]

        try:
            self.labelEncoder.classes_ = np.load(path)
            self.ui.teInfo.append('Class info for encoder is loaded from npy file successfully')
        except:
            self.ui.teInfo.append('Unable to load class info for encoder from npy file!')

    def predict(self):
        if self.isModelTrained == False:
            self.ui.teInfo.append('A trained model is needed for prediction')
            return

        testPath = self.ui.leTestSetPath.text()
        xTestFilePaths = []
        xTestFileNames = []
        yTest = []
        yTestEncoded = None
        yPred = []
        yPredDecoded = []
        formattedResults = []

        for root, dirs, files in os.walk(testPath):
            for file in files:
                if file.endswith('.nii'):
                    filePath = os.path.join(root, file)
                    xTestFilePaths.append(filePath)
                    xTestFileNames.append(file)
                    pathParts = Path(root).parts
                    labelName = pathParts[-1]
                    yTest.append(labelName)
        
        if not yTest:
            self.ui.teInfo.append('Could not find any .nii file in test path')
            return

        try:
            yTestEncoded = self.labelEncoder.transform(yTest)
            labelList = self.labelEncoder.classes_

            labelListString = '[ '
            for label in labelList:
                labelListString = labelListString + label + ' '
            labelListString = labelListString + ']'

            numClasses = len(labelList)            
            yTestEncoded = to_categorical(yTestEncoded, num_classes = numClasses)
            self.ui.tbPredictions.clear()
            self.ui.tbPredictions.append('Encode Info: ' + labelListString)
            self.ui.tbPredictions.append('File Name, Label, Prediction')

            for index, filePath in enumerate(xTestFilePaths):
                niImg = nib.load(filePath).get_fdata()
                niImgArr = np.asarray(niImg)
                rows, cols, depth = niImgArr.shape
                channel = 1
                niImgArr = niImgArr.reshape(-1, rows, cols, depth, channel)
                pred = self.model.predict(niImgArr)
                yPred.append(pred[0])
                yPredDecoded.append(labelList[np.argmax(yPred[index])])
                formattedResults.append([xTestFileNames[index], str(yTest[index]), str(yPredDecoded[index])]) # str() is required to print encoded label
                self.ui.tbPredictions.append(formattedResults[index][0] + ', ' + formattedResults[index][1] + ', ' + formattedResults[index][2])
                print('Prediction completed for:', filePath)

            # Write results and confusion matrix to text files

            cwd = os.getcwd()
            outputFolderPath = os.path.join(cwd, 'files')
            Path(outputFolderPath).mkdir(parents=True, exist_ok=True)

            csvFile = open(os.path.join(outputFolderPath,'testResults.txt'), 'w', newline='')
            csvWriter = csv.writer(csvFile, delimiter=',')
            csvWriter.writerow(['File Name','Label', 'Prediction'])
            csvWriter.writerows(formattedResults)
            csvFile.close()

            performanceMetrics = classification_report(yTest, yPredDecoded, labels=labelList, zero_division=0) # target_names can be used instead of labels to rename classes
            textFile = open(os.path.join(outputFolderPath,'performanceMetrics.txt'), 'w+')
            textFile.write(performanceMetrics)
            textFile.close()

            confusionMatrix = confusion_matrix(yTest, yPredDecoded, labels=labelList)
            confusionMatrix = np.asarray(confusionMatrix)
            np.savetxt(os.path.join(outputFolderPath,'confusionMatrix.txt'), confusionMatrix, delimiter=',', fmt='%1.2f')

            self.ui.teInfo.append('Test completed successfully')

        except:
            self.ui.teInfo.append('Something went wrong with test')
            print('Unexpected error:', sys.exc_info()[0]) # Print the error to the console

def app():
    app = QtWidgets.QApplication(sys.argv)
    win = myApp()
    win.show()
    sys.exit(app.exec_())

app()

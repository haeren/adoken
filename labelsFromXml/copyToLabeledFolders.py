import os
import shutil
import pandas as pd
from pathlib import Path

folderPath = os.getcwd()
fileCounter = 0
df = pd.read_csv('fileNamesAndLabels.txt')

for root, dirs, files in os.walk(folderPath):
    for file in files:
        if file.endswith('.nii'):
            mriPath = os.path.join(root,file)
            mriParentPath = Path(mriPath).parent
            for item in os.listdir(mriParentPath):
                path = os.path.join(mriParentPath, item)
                if os.path.isfile(path) and item.endswith('.xml'):
                    fileNameWoExt = os.path.splitext(item)[0]
                    index = df.index[df['sample'] == fileNameWoExt].tolist()[0]
                    label = df.iloc[index,1]
                    outputFolderPath = os.path.join(folderPath, 'formattedData', label)
                    Path(outputFolderPath).mkdir(parents=True, exist_ok=True)
                    shutil.copy('\\\\?\\' + mriPath, outputFolderPath)
                    fileCounter = fileCounter + 1
                    print('File', fileCounter, 'is copied.')

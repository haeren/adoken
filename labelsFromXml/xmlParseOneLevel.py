import xml.etree.ElementTree as ET
import csv
import os

folderPath = os.getcwd()

csvFile = open('fileNamesAndLabels.txt', 'w', newline='')
csvWriter = csv.writer(csvFile, delimiter=',')
csvWriter.writerow(['sample','label'])

for item in os.listdir(folderPath):
    path = os.path.join(folderPath, item)
    if os.path.isfile(path) and item.endswith('.xml'):
        rootNode = ET.parse(path).getroot()
        for tag in rootNode.findall('project/subject/subjectInfo'):
            value = tag.attrib['item']
            if value == 'DX Group':
                fileNameWoExt = os.path.splitext(item)[0]
                fileNameParts = fileNameWoExt.split('_')
                sampleName = fileNameParts[-2] + fileNameParts[-1]
                fileLabel = tag.text
                row = [sampleName, fileLabel]
                csvWriter.writerow(row)

csvFile.close()
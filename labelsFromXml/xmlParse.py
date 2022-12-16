import xml.etree.ElementTree as ET
import csv
import os

folderPath = os.getcwd()

csvFile = open('fileNamesAndLabels.txt', 'w')
csvWriter = csv.writer(csvFile, delimiter=',')

for root, dirs, files in os.walk(folderPath):
    for file in files:
        if file.endswith('.xml'):
            xmlPath = os.path.join(root, file)
            rootNode = ET.parse(xmlPath).getroot()
            for tag in rootNode.findall('project/subject/subjectInfo'):
                value = tag.attrib['item']
                if value == 'DX Group':
                    fileNameWoExt = os.path.splitext(file)[0]
                    fileLabel = tag.text
                    row = [fileNameWoExt, fileLabel]
                    csvWriter.writerow(row)

csvFile.close()
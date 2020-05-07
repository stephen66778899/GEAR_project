from PIL import Image
from pathlib import Path
import numpy as np
import sys
import os
import csv

#Useful function
def createFileList(myDir, format='.jpg'):
    filelabelList = []
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                filepath = Path(os.path.join(root, name))
                directory = filepath.parent.parent
                directories = str(directory).split('/')
                label = directories[len(directories)-1]
                fullName = os.path.join(root, name)
                filelabelList.append([str(fullName), str(label)])
    return filelabelList

data_dir = 'OOWL_in_the_wild/train'
file_name = 'train.csv'

myFileList = createFileList(data_dir)
with open(file_name, 'w', newline='') as F:
    for file in myFileList:
        writer = csv.writer(F)
        writer.writerow(file) 

import xml.etree.ElementTree as ET
from os import getcwd
from os import walk
from os.path import join


#請依照順序改為自己的類別
classes = ["fish"]


def convert_annotation(image_id, list_file):
    in_file = open('own_data/xml/%s.xml'%(image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()
own_data_name=[]


    
from os import listdir
from os.path import isfile, isdir, join


mypath = "own_data/img"
files = listdir(mypath)


for f in files:
    fullpath = join(mypath, f)
    if isfile(fullpath):
        print("檔案：", f)
        own_data_name.append(f)
print(own_data_name)

    
    
    
          
list_file = open('own_datapath.txt', 'w')
for image_id in own_data_name:
    list_file.write('own_data/img/%s.jpg'%(image_id.split('.')[0]))
    convert_annotation(image_id.split('.')[0], list_file)
    list_file.write('\n')
list_file.close()


#!/usr/bin/python3

_description='''
This tool is used to generate pre-loadable(SW defined prototxt format <schema/DlaInterface.proto>) from caffe prototxt (<schema/caffe.proto>)
'''

import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np

def main():    
    i = 0

    cmd = ('rm ./txt/*')
    os.system(cmd)

    for root, dirs, files in os.walk('./output'):
        for file in files:
            if os.path.splitext(file)[1] == '.dat':
                f = open('./output/' + file,"rb")
                num = os.path.splitext(file)[0][14:20]
                file = open('./txt/data'+str(num)+'.txt','a')
                contents = f.readlines()
                #print(contents)
                for line in contents:
                    if not line.strip():continue
                    p = line.split('double_data:')
                    if len(p) == 2:
                        tmp = p[1][0:7]              
                        file.write(str(p[1][0:6]))                                                               
                file.close() 
                f.close()
                i = i + 1
    return

if __name__ == "__main__":
    main()

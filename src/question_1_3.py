import numpy as np
import glob
import cv2
from skimage.transform import integral_image
from skimage.feature import hessian_matrix_det, peak_local_max
import matplotlib.pyplot as plt
import math
import json
import pandas as pd

def get_ratio(a, b, c):

    return (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)

def distance(x1, y1, x2, y2):

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def make_float(arr):

    arr1 = []
    for i in range(len(arr)):
        arr1.append([float(arr[i][0]), float(arr[i][1]), float(arr[i][2])])
    return arr1

def clean(arr):

    arr1 = []

    for i in range(len(arr)):
        if (arr[i][2] == 0):
            continue
        else:
            arr1.append(arr[i])

    return arr1

def remove_redundancy(blobs, overlap=0.5):

    for b in range(len(blobs)):
       
        x = blobs[b][0]
        y = blobs[b][1]
        sig = blobs[b][2]

        for b1 in range(len(blobs)):

            if (b != b1):

                x1 = blobs[b1][0]
                y1 = blobs[b1][1]
                sig1 = blobs[b1][2] 

                dis = distance(x1, y1, x, y)

                if (dis > sig + sig1):
                    continue

                elif (dis <= abs(sig - sig1)):
                    if (sig > sig1):
                        blobs[b1][2] = 0
                    else: 
                        blobs[b][2] = 0

                elif ((dis < (sig + sig1)) and (dis > abs(sig - sig1))):
                    ratio1 = np.clip(get_ratio(dis, sig, sig1), -1, 1)
                    ratio2 = np.clip(get_ratio(dis, sig1, sig), -1, 1)
                    acos1 = math.acos(ratio1)
                    acos2 = math.acos(ratio2)
                    
                    a1 = -dis + sig1 + sig
                    a2 = dis - sig1 + sig
                    a3 = dis + sig1 - sig
                    a4 = dis + sig + sig1
                    a = (sig**2.*acos1 + sig1**2.*acos2 - 0.5*math.sqrt(abs(a1*a2*a3*a4)))
                    a_ = a / (math.pi * (min(sig**2, sig1**2.)))
                    
                    if (a_ >= overlap):
                        if (sig > sig1):
                            blobs[b1][2] = 0
                        else:
                            blobs[b][2] = 0

    return blobs

def make_features(img, features, img_name):

    fig, ax = plt.subplots()
    ax.imshow(img)

    for i in range(len(features)):
        y, x, r = features[i][0], features[i][1], features[i][2]
        c = plt.Circle((x, y), r, color='k', fill=False)
        ax.add_patch(c)

    ax.plot()
    plt.savefig('./database_surf/' + img_name.split('/')[-1].split('.')[0] + '.png')
    plt.close()

def get_feature_vector_database(img):

    integral_img = integral_image(img)
    features = []

    for sig in (sigmas):

        determinant_hessian = hessian_matrix_det(img, sigma=sig)        
        coordinates = peak_local_max(determinant_hessian, num_peaks=50)
        
        for i in range(coordinates.shape[0]):
            features.append([coordinates[i][0], coordinates[i][1], sig])

    return features

def main_database():

    img_database = glob.glob('./images/*jpg')
    json_file = {}

    img_counter = 0
    
    for im in img_database:

        img = cv2.imread(im)
        img = cv2.resize(img, (256, 256))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        fea = get_feature_vector_database(img_gray)
        fea = clean(remove_redundancy(fea))
    
        make_features(img, fea, im)

        json_file[im] = make_float(fea)

        img_counter += 1
        print (img_counter)
    
    with open('surf.json', 'w') as f:
        json.dump(json_file, f)

def query():

    query_data = glob.glob('./train/query/*.txt')
    img_counter = 0
    json_file = {}

    for qu in query_data:

        with open(qu, 'r') as g:
            for line in g:
                a = line.split(' ')[0][5:]

        img_name = './images/' + a + '.jpg'
        
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (256, 256))

        fea = get_feature_vector_database(img)
        fea = clean(remove_redundancy(fea))
    
        json_file[qu] = make_float(fea)

        img_counter += 1
        print (img_counter)
    
    with open('surf.json', 'w') as f:
        json.dump(json_file, f)


if (__name__ == '__main__'):

    sigmas = np.arange(2, 6, 0.25)
    
    main_database()

    #query()
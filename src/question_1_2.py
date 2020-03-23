import numpy as np
import glob
import cv2
import math
from scipy import ndimage
import copy
import matplotlib.pyplot as plt
import json

def distance(x1, y1, x2, y2):

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def get_ratio(a, b, c):

    return (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)

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

def make_img(img, blobs, img_name):

    blobs = np.asarray(blobs)

    fig, ax = plt.subplots()
    ax.imshow(img)

    for i in range(len(blobs)):
        y, x, r = blobs[i, 0], blobs[i, 1], intial_sigma * (blobs[i, 2] + 1)
        c = plt.Circle((x, y), r, color='k', fill=False)
        ax.add_patch(c)

    ax.plot()
    plt.savefig('./database_blobs/' + img_name.split('/')[-1].split('.')[0] + '.png')

def get_laplacian_of_gaussian_filter(f_s, sigma):

    n = np.ceil(sigma * kernel_constant)
     
    a = np.ogrid[-int(n/2) : int(n/2) + 1, -int(n/2) : int(n/2) + 1]
    y = a[0]
    x = a[1]

    y1 = np.exp(-(y*y / (2 * sigma * sigma)))
    x1 = np.exp(-(x*x / (2 * sigma * sigma)))
    
    log = (-(2*sigma**2) + (x**2 + y**2) ) * (x1*y1) * (1 / (2*math.pi*sigma**4))
    
    return log

def features_log(img):
    
    conv_imgs = np.zeros((num_log, img_size, img_size))

    for i in range(num_log):
        
        sigma = intial_sigma * (i+1)
        log = get_laplacian_of_gaussian_filter(sigma * kernel_constant, sigma)
        #conv_img = ndimage.convolve(img, log)
        conv_img = cv2.filter2D(img, -1, log)
        conv_img = np.square(conv_img)

        conv_imgs[i, :, :] = conv_img

    return conv_imgs

def nms(c_i):

    points_nms = []

    for scales in range(1, num_log-1):

        for i in range(1, img_size-1):
            for j in range(1, img_size-1):

                neighbors = []
                
                for d in directions:
                    neighbors.append(c_i[scales, i + d[0], j + d[1]])
                    neighbors.append(c_i[scales-1, i + d[0], j + d[1]])
                    neighbors.append(c_i[scales+1, i + d[0], j + d[1]])
                neighbors.append(c_i[scales-1, i, j])
                neighbors.append(c_i[scales+1, i, j])

                if (c_i[scales, i, j] >= max(neighbors)):
                    if (c_i[scales, i, j] >= 0.03):
                        points_nms.append([i, j, scales])

    return points_nms

def remove_redundancy(blobs, overlap=0.5):

    for b in range(len(blobs)):
       
        x = blobs[b][0]
        y = blobs[b][1]
        s = blobs[b][2] + 1
        sig = intial_sigma * s

        for b1 in range(len(blobs)):

            if (b != b1):

                x1 = blobs[b1][0]
                y1 = blobs[b1][1]
                s1 = blobs[b1][2] + 1
                sig1 = intial_sigma * s1

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

def main():

    img_database = glob.glob('./images/*.jpg')

    json_file = {}
    img_counter = 0

    for im in img_database:

        img = cv2.imread(im)
        img = cv2.resize(img, (img_size, img_size))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        conv_imgs = features_log(img_gray)
        conv_imgs_nms = nms(conv_imgs)
        conv_imgs_red = clean(remove_redundancy(conv_imgs_nms))
        
        make_img(img, conv_imgs_red, im)
        json_file[im] = make_float(conv_imgs_red)

        img_counter += 1
        print (img_counter)

    with open('blobs.json', 'w') as f:
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
        img = cv2.resize(img, (img_size, img_size))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        conv_imgs = features_log(img_gray)
        conv_imgs_nms = nms(conv_imgs)
        conv_imgs_red = clean(remove_redundancy(conv_imgs_nms))
        
        json_file[qu] = make_float(conv_imgs_red)

        img_counter += 1
        print (img_counter)

    with open('blobs.json', 'w') as f:
        json.dump(json_file, f)


if (__name__ == '__main__'):

    directions = [[-1, 0], [1, 0], [0, -1], [0, 1], [1, 1], [-1, -1], [-1, 1], [1, -1]]
    img_size = 128
    num_log = 5
    intial_sigma = 1
    kernel_constant = 6

    query()



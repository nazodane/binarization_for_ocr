# License: LGPL v2.1 or later (for now)
# (C) 2014 Toshimitsu Kimura
#
# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
# cimport numpy as np
import math

if 1:
#def main():
#    c_main()

#cdef c_main():

    if len(sys.argv) != 3:
        print "./binarization_for_ocr.py <in_file> <out_file>"
        quit()

    # cdef unicode
    filename = sys.argv[1]
    # cdef unicode
    outfile = sys.argv[2]

    # cdef np.ndarray[DTYPE_t, ndim=2]
    image = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    if image is None:
        print "input file is not found"
        quit()

    process(image, outfile, True)

def process(image, outfile, retry):
    # cdef np.ndarray[DTYPE_t, ndim=2]
    mask = np.zeros([image.shape[0], image.shape[1] ], dtype=np.uint8)

    # cdef np.ndarray[np.int32_t, ndim=2]
    cluster_idx = np.zeros([image.shape[0] * image.shape[1], 1], dtype=np.int32)
    #cdef np.ndarray[np.float32_t, ndim=2]
    points = np.zeros([image.shape[0] * image.shape[1], 1], dtype=np.float32)

    # cdef size_t
    i, j, x = 0, 0, 0

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            points[x] = float(image[i,j]) #<float> image[i,j]
            x += 1

    #cdef np.ndarray[np.float32_t, ndim=2] centers

    _,cluster_idx,centers = cv2.kmeans(points, 3, (cv2.cv.CV_TERMCRIT_ITER | cv2.cv.CV_TERMCRIT_EPS, 10, 1.0), 1, cv2.KMEANS_PP_CENTERS)

    #cdef size_t
    wi = 0

    for i in range(1, centers.shape[0]):
        if centers[i] > centers[wi]:
            wi = i

    x = 0
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if cluster_idx[x][0] == wi:
                image[i, j] = 255
            x += 1

    img_bw = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 3)

    if retry:
        contours, hierarchy = cv2.findContours(np.copy(img_bw), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maxc = None
        maxsz = image.shape[0]*image.shape[1]
        pghalf = (image.shape[0]*image.shape[1])/2
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if w*h > pghalf and w*h < maxsz:
                maxsz = w*h
                maxc = contour
        if maxc is not None:
            x,y,w,h = cv2.boundingRect(maxc)
            cv2.rectangle(mask,(x,y),(x+w,y+h),255,-1)
            for i in range(0, image.shape[0]):
                for j in range(0, image.shape[1]):
                    if mask[i,j] != 255:
                        image[i,j] = 255
            process(image, outfile, False)
            quit()

    cv2.imwrite(outfile, img_bw)


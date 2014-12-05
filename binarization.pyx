# License: AGPLv3 (for now)
# (C) 2014 Nazo
#
# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
cimport numpy as np
import math
cimport cython


cdef unicode filename = u"face.jpg"
cdef unicode outfile = u"out.jpg"

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t

# 周囲からカラーをもらう。値: 0～255
@cython.boundscheck(False)
cdef float get_around(np.ndarray[DTYPE_t, ndim=3] ar, np.uint8_t my, size_t i, size_t j):
    cdef np.uint16_t sumi_score = <np.uint16_t>ar[i-1,j-1][0] + ar[i-1,j+1][0] + ar[i+1,j-1][0] + ar[i+1,j+1][0] # 四隅
    cdef float score = (sumi_score/1.4142) + (<np.uint16_t>ar[i,j-1][0] + ar[i,j+1][0] + ar[i-1,j][0] + ar[i+1,j][0]) # 四隅 + 上下左右
# undecidedが影響を与えるのは望ましくないのでは?
    cdef float over = (((math.sqrt(2.0)/2.0) ** 2) * 3.1416 - 1.0) # ピクセルの外周円の面積よりピクセルの面積を引く

    return (score*(over/(4.0/1.4142+4.0)) + my)/ (over+1.0)

cdef np.uint8_t black_white_or_undecided(float i, float kw, float kb, np.uint8_t threshed, np.uint8_t wh_min, np.uint8_t bk_max):
    return wh_min if i>kw else bk_max if i<kb else threshed

# 未定の閾値を狭めてくべき

#cdef unsigned char black_or_white(float i, np.uint8_t threshed, np.uint8_t wh_min, np.uint8_t bk_max):
#    return wh_min if i>threshed else bk_max

# 0 = black, 255 = white, 127 = undecided


def main():
    main_c()

@cython.boundscheck(False)
cdef main_c():
    cdef np.ndarray[DTYPE_t, ndim=3] image = cv2.imread(filename)

    #size = image.shape[0], image.shape[1], image.shape[2] # width, height, depth
    #newimage = np.zeros(image.shape, dtype=image.dtype)

    cdef np.ndarray[DTYPE_t, ndim=3] newimage = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype=np.uint8)

    cdef np.ndarray[np.int32_t, ndim=2] cluster_idx = np.zeros([image.shape[0] * image.shape[1], 1], dtype=np.int32)
    cdef np.ndarray[np.float32_t, ndim=2] points = np.zeros([image.shape[0] * image.shape[1], 1], dtype=np.float32)

    cdef size_t i, j

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            points[i*j] = <float> image[i,j][0];
#    points = points.reshape(3, image.shape[0]*image.shape[1]);

    cdef np.ndarray[np.float32_t, ndim=2] centers

    # 第二引数は3以上。4はてきとう。
    _,cluster_idx,centers = cv2.kmeans(points, 4, (cv2.cv.CV_TERMCRIT_ITER | cv2.cv.CV_TERMCRIT_EPS, 10, 1.0), 1, cv2.KMEANS_PP_CENTERS)

    cdef size_t wi = centers[0]
    cdef size_t bi = centers[0]

    for i in range(1, centers.shape[0]):
        if centers[i] > centers[wi]:
            wi = i
        if centers[i] < centers[bi]:
            bi = i

    cdef np.uint8_t wh_min = 255
    cdef np.uint8_t bk_max = 0
    cdef np.uint8_t pt
    for i in range(0, points.shape[0]):
        pt = <np.uint8_t>points[i][0] # XXX
        if cluster_idx[i][0] == wi and wh_min > pt:
           wh_min = pt;
        if cluster_idx[i][0] == bi and bk_max < pt:
           bk_max = pt;

    print "White Level: %s"%wh_min
    print "Black Level: %s"%bk_max

    cdef np.uint8_t kb_base = bk_max#10 #　ソースの黒レベル
    cdef np.uint8_t kw_base = wh_min#227 #　ソースの白レベル
    cdef np.uint8_t k_threshed = (kb_base + kw_base) / 2
    cdef float kb = kb_base
    cdef float kw = kw_base
    cdef size_t mk = 10
    cdef float kbstep = ((k_threshed+1)-kb)/mk
    cdef float kwstep = (k_threshed-(255-kw))/mk

# 細かいものは積極的に選択して、そうでないものは無視すべき

    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            newimage[i,j] = black_white_or_undecided(image[i,j][0], kw, kb, k_threshed, wh_min, bk_max)

    cdef float k
    for k in range(1, mk):
        print "path %s..."%k
        kw -= kwstep#(k_threshed-(255-kw))/2
        kb += kbstep#((k_threshed+1)-kb)/2
        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                if newimage[i,j][0] == k_threshed:
                    newimage[i,j] = black_white_or_undecided(get_around(newimage, image[i,j][0], i, j), kw, kb, k_threshed, wh_min, bk_max)

    cv2.imwrite(outfile, newimage)

    cv2.imshow(u'test', newimage)
    cv2.waitKey()


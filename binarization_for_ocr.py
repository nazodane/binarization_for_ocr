# License: LGPL v2.1 or later (for now)
# (C) 2014 Toshimitsu Kimura
#
# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
# cimport numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import colors as clr
# from statistics import mode
from scipy.stats import mode

def main():
#    c_main()

#cdef c_main():

    if len(sys.argv) != 3:
        print "./binarization_for_ocr.py <in_file> <out_file>"
        quit()

    # cdef unicode
    filename = sys.argv[1]
    # cdef unicode
    outfile = sys.argv[2]

    # cv2.ocl.setUseOpenCL(True)

    # cdef np.ndarray[DTYPE_t, ndim=2]
    image_color = cv2.imread(filename, cv2.IMREAD_COLOR)

    if image_color is None:
        print "input file is not found"
        quit()

    channels = cv2.split(image_color)
    image = channels[1] # drop blue channel for yellowish books

    mode_0 = mode(channels[0].flat)[0]
    mode_1 = mode(channels[1].flat)[0]
    mode_2 = mode(channels[2].flat)[0]

    channels[0] = cv2.absdiff(channels[0], mode_0)
    channels[1] = cv2.absdiff(channels[1], mode_1)
    channels[2] = cv2.absdiff(channels[2], mode_2)
    img_diff = cv2.max(channels[0], channels[1])
    img_diff = cv2.max(img_diff, channels[2])

    img_diff = cv2.fastNlMeansDenoising(img_diff, 100, 7, 21) ###

    process(image, img_diff, outfile, True, 3)

def process(image, img_diff, outfile, retry, kn):
    # cdef np.ndarray[DTYPE_t, ndim=2]
    mask = np.zeros([image.shape[0], image.shape[1] ], dtype=np.uint8)

    # cdef np.ndarray[np.int32_t, ndim=2]
    cluster_idx = np.zeros([image.shape[0] * image.shape[1] , 1], dtype=np.int32) # / 128 +1

    # cdef size_t
    i, j, x = 0, 0, 0

    img_diff = np.float32(img_diff)

    c0 = image.shape[0]/2.0
    c1 = image.shape[1]/2.0

    ax = np.arange(-c0,image.shape[0]-c0, dtype=np.float32)
    ax = np.uint8(np.absolute(ax / c0) * 255.0)

    ay = np.arange(-c1,image.shape[1]-c1, dtype=np.float32)
    ay = np.uint8(np.absolute(ay / c1) * 255.0)

    ax = np.repeat(ax, image.shape[1])
    ay = np.tile(ay, image.shape[0])

    points = 255 - np.maximum(ax, ay)

    sz = 256

    df_flat = img_diff.flat

    grid = np.zeros([sz, sz], dtype=np.uint8)
#    grid = np.zeros([256, 256], dtype=np.uint32)
    for i in range(0, points.shape[0]):
        if grid[points[i], df_flat[i]] != 255: grid[points[i], df_flat[i]] += 1
        # dither-aware (+0, +1 or -1)
#        if df_flat[i] != 255:
#            if grid[points[i], df_flat[i] + 1] != 255: grid[points[i], df_flat[i] + 1] += 1
#        if df_flat[i] != 0:
#            if grid[points[i], df_flat[i] - 1] != 255: grid[points[i], df_flat[i] - 1] += 1

    cv2.imwrite("%s_grid.png"%outfile, grid)

#    center = 127 # XXX

    grid_cluster_idx = np.zeros([image.shape[0] * image.shape[1] , 2], dtype=np.int32) # / 128 +1
    _,grid_cluster_idx,centers = cv2.kmeans(np.float32(grid.flat), 2, grid_cluster_idx, (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 10, 1.0), 3, cv2.KMEANS_PP_CENTERS)

    wi = 0 if centers[0][0] > centers[1][0] else 1
    grid_m = np.ma.array(np.copy(grid), mask=(grid_cluster_idx != wi))
    center = np.ma.min(grid_m)
    print (center)

    _, grid_mask = cv2.threshold(np.uint8(grid), center, 255, cv2.THRESH_BINARY)

#    grid_mask = cv2.fastNlMeansDenoising(grid_mask, 100, 7, 21) ###

    cv2.imwrite("%s_grid_mask.png"%outfile, grid_mask)

    points_n = np.reshape(points, (image.shape[0], image.shape[1]))



    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if grid_mask[points_n[i, j], img_diff[i, j]] != 0:
                image[i, j] = 255

    cv2.imwrite("%s_gridded.png"%outfile, image)
#    quit()




#    points = np.dstack((img_diff.flat, points))[0]



#    points = points[0:x]

    #cdef np.ndarray[np.float32_t, ndim=2] centers

#    _,cluster_idx,centers = cv2.kmeans(points, kn, cluster_idx, (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 10, 1.0), 3, cv2.KMEANS_PP_CENTERS)

# memory problem
#    from sklearn.cluster import SpectralClustering, spectral_clustering
#
#    sc = SpectralClustering(n_clusters = 3, eigen_solver='arpack')
#    print "predict!"
#    labels = sc.fit_predict(points)

#    labels = spectral_clustering(points, n_clusters=3, eigen_solver='arpack')

#    import sklearn.cluster
#    from sklearn.neighbors import kneighbors_graph
#    connectivity = kneighbors_graph(points, n_neighbors=10)
#    connectivity = 0.5 * (connectivity + connectivity.T)
#    ward = sklearn.cluster.AgglomerativeClustering(n_clusters=3, linkage='ward', connectivity=connectivity)
#    labels = ward.fit_predict(points)

# my mistakes
#    import sklearn.cluster
#    il = np.zeros([2, 2], dtype=np.float32)
#    il[1][0],il[1][1] = 1.0, 1.0
#    ap = sklearn.cluster.AffinityPropagation(damping=.9,preference = il)
#    labels = ap.fit_predict(points)

#--
#    import sklearn.cluster
#    from sklearn.neighbors import kneighbors_graph
#    connectivity = kneighbors_graph(points, n_neighbors=10)
#    connectivity = 0.5 * (connectivity + connectivity.T)
#    ag = sklearn.cluster.AgglomerativeClustering(linkage="average",
#                                                 affinity="cityblock", n_clusters=10,
#                                                 connectivity=connectivity)
#    labels = ag.fit_predict(points)
#--

##    from sklearn.cluster import DBSCAN
##    dbs = DBSCAN()
##    labels = dbs.fit_predict(points)

#    for i in range(0, labels.shape[0]):
#         plt.scatter(points[i,0], points[i,1], c=['r', 'g', 'b', 'y', 'm', 'c', 'k'][labels[i]%6])
##        if(x%128 == 0): plt.scatter(points[i,0], points[i,1], c=['r', 'g', 'b'][labels[i]])
#    plt.xlabel('Color'),plt.ylabel('Distance')
#    plt.show()
#    quit()

    #cdef size_t
    wi = 0
    ei = 0

    _,cluster_idx,centers = cv2.kmeans(np.float32(image.flat), 2, cluster_idx, (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 10, 1.0), 3, cv2.KMEANS_PP_CENTERS)


    for i in range(1, centers.shape[0]):
        if centers[i][0] > centers[wi][0]: # white
            wi = i

    image_m = np.ma.array(np.copy(image), mask=(np.reshape(cluster_idx, (image.shape[0], image.shape[1])) != wi))
    center = np.ma.min(image_m)
    print (center)

    image_m = np.ma.array(np.copy(image), mask=(np.reshape(cluster_idx, (image.shape[0], image.shape[1])) == wi))
    image = np.ma.filled(image_m, 255)

#    cmap = clr.ListedColormap(['r', 'g', 'b'], 3)
#    plt.scatter(points[:,0], points[:,1], c=cluster_idx, cmap = cmap)
#    plt.xlabel('Color'),plt.ylabel('Distance')
#    plt.show()

    cv2.imwrite("%s_clean.png"%outfile, image)
    print "cleaned"

#    if not retry:
#        cv2.imwrite(outfile, image)
#        quit()

    image = cv2.fastNlMeansDenoising(image, 100, 7, 21)

    img_bw = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 3)

    img_mode = mode(image.flat)[0]

#    while retry:
#    if retry:
    if False:
        edges = cv2.Canny(image, img_mode-50, img_mode+50)#cv2.Canny(image, centers[wi]-50, centers[wi]+50)

        blurred_edges = cv2.blur(edges,(7,7))
        _, edges = cv2.threshold(blurred_edges, 1, 255, cv2.THRESH_BINARY)

        cv2.imwrite(u"%s_debug.png"%outfile, edges)

        _, contours, hierarchy = cv2.findContours(np.copy(edges), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        maxc = None
        maxsz = image.shape[0]*image.shape[1]
        maxl = None
        maxlsz = image.shape[0]*image.shape[1]/2
        maxr = None
        maxrsz = image.shape[0]*image.shape[1]/2

        pghalf = (image.shape[0]*image.shape[1])/2
        pgquarter = (image.shape[0]*image.shape[1])/4

        for contour in contours:
            area = cv2.contourArea(contour)
#            cv2.drawContours(mask, [contour], 0, 255,4)
            if area > pghalf and area < maxsz:
                maxsz = area
                maxc = contour
            if area > pgquarter and area < maxlsz and max([(0 if p[0][0] < image.shape[0]/2 else 1) for p in contour]) == 0:
                maxlsz = area
                maxl = contour
            if area > pgquarter and area < maxrsz and max([(0 if p[0][0] > image.shape[0]/2 else 1) for p in contour]) == 0:
                maxrsz = area
                maxr = contour

        if maxl is not None and maxr is not None:
# not working...
            print "2page..."
            cv2.drawContours(mask, [maxl], 0, 255,-1)
            cv2.drawContours(mask, [maxl], 0, 0, 7)
            cv2.drawContours(mask, [maxr], 0, 255,-1)
            cv2.drawContours(mask, [maxr], 0, 0, 7)
            cv2.imwrite(u"%s_debug2.png"%outfile, mask)
            for i in range(0, image.shape[0]):
                for j in range(0, image.shape[1]):
                    if mask[i,j] != 255:
                        image[i,j] = 255
            process(image, img_diff, outfile, False, 3)
            quit()

        if maxc is not None:
            cv2.drawContours(mask, [maxc], 0, 255,-1)
            cv2.drawContours(mask, [maxc], 0, 0, 7)
            print "1page..."
            cv2.imwrite(u"%s_debug2.png"%outfile, mask)
            for i in range(0, image.shape[0]):
                for j in range(0, image.shape[1]):
                    if mask[i,j] != 255:
                        image[i,j] = 255
            process(image, img_diff, outfile, False, 3)
            quit()

#        if (maxl is None or maxr is None) and maxc is None:
#            process(image, outfile, False, 2)
#            quit()

    cv2.imwrite(outfile, img_bw)

main()

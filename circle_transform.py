# License: Public Domain
# (C) 2014 Toshimitsu Kimura

import sys
import cv2
import numpy as np
import math

def main():
    if len(sys.argv) != 3:
        print "./circle_transform.py <in_file> <out_file>"
        quit()

    filename = sys.argv[1]
    outfile = sys.argv[2]

    image = cv2.imread(filename, cv2.IMREAD_COLOR)

    if image is None:
        print "input file is not found"
        quit()

    circle = np.zeros([image.shape[0], image.shape[1], 2], dtype=np.float)
    center = [image.shape[0]/2.0, image.shape[1]/2.0]

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            pos = [i-center[0], j-center[1]]
            if pos[0] == 0: continue
            n = pos[1]/pos[0] # y = nx, x = y/n
            if n == 0: continue
            circle[i,j,0] = 1.0 - math.sqrt(pos[0]**2 + pos[1]**2) / math.sqrt(min(center[0]**2 + (n*center[0])**2, center[1]**2 + (center[1]/n)**2))
            circle[i,j,1] = math.atan2(pos[1], pos[0])

    img_out_size = max(image.shape[0], image.shape[1]) / 2.0
    img_out = np.zeros([img_out_size, img_out_size, image.shape[2]], dtype=np.uint8)

    out_half = [img_out.shape[0]/2.0, img_out.shape[1]/2.0]
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            dist = circle[i,j,0]
            rad = circle[i,j,1]
            idx_x = (math.cos(rad)*dist*out_half[0] + out_half[0])%img_out.shape[0]
            idx_y = (math.sin(rad)*dist*out_half[1] + out_half[1])%img_out.shape[1]
            img_out[idx_x, idx_y] = image[i, j]

    cv2.imwrite(outfile, img_out)

main()

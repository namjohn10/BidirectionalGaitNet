## This is code for converting from c3d file to bvh file.

import sys
import os
import numpy as np
import math
import bvh
import c3d
from IPython import embed
import copy

def load_c3d(c3d_path):
    with open(c3d_path, 'rb') as handle:
        reader = c3d.Reader(handle)
        labels = reader.point_labels
        total_points = []
        for i, points, analog in reader.read_frames():
            ps = []
            for point in points:
                p = np.zeros(3)

                p[0] = 0.001 * point[1]
                p[1] = 0.001 * point[2]
                p[2] = 0.001 * point[0]
                
                ps.append(p)

            total_points.append(copy.deepcopy(ps))
        print(labels)
        print('Succesfully loaded c3d file')
        return labels, total_points, int(reader.header.frame_rate)







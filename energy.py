# Image Completion using Statistics of Patch Offsets
# Author: Pranshu Gupta and Shrija Mishra

import cv2, numpy as np, sys, math, operator, maxflow, random
from scipy import ndimage
from itertools import count, combinations

class Optimizer(object):
    def __init__(self, image, mask, labels, tau):
        self.image = image
        self.mask = mask
        self.labels = labels
        self.tau = tau

    def D(self, site, offset):
        i, j = site[0] + offset[0], site[1] + offset[1]
        try:
            if self.mask[i][j] == 0:
                return 0
            return float('inf')
        except:
            return float('inf')

    def V(self, site1, site2, alpha, beta):
        x1a, y1a = site1[0] + alpha[0], site1[1] + alpha[1]
        x2a, y2a = site2[0] + alpha[0], site2[1] + alpha[1]
        x1b, y1b = site1[0] + beta[0], site1[1] + beta[1]
        x2b, y2b = site2[0] + beta[0], site2[1] + beta[1]
        try:
            if self.mask[x1a, y1a] == 0 and self.mask[x1b, y1b] == 0 and self.mask[x2a, y2a] == 0 and self.mask[x2a, y2a] == 0:
                return np.sum((self.image[x1a, y1a] - self.image[x1b, y1b])**2) + np.sum((self.image[x2a, y2a] - self.image[x2b, y2b])**2)
            return float('inf')
        except:
            return float('inf')

    def EnergyCalculator(self, sites, labelling):
        i, e = 0, 0
        while i < len(sites) and e < float('inf'):
            e = e + self.D(sites[i], self.labels[labelling[i]])
            neighbors = self.GetNeighbors(sites[i])
            for n in neighbors:
                if n in sites:
                    if sites.index(n) > i:
                        e = e + self.V(sites[i], n, self.labels[labelling[i]], self.labels[labelling[sites.index(n)]])
            i += 1
        return e

    def GetNeighbors(self, site):
        return [[site[0]-1, site[1]], [site[0], site[1]-1], [site[0]+1, site[1]], [site[0], site[1]+1]]

    def AreNeighbors(self, site1, site2):
        if np.abs(site1[0]-site2[0]) < 2 and np.abs(site1[1]-site2[1]) < 2:
            return True
        return False 

    def InitializeLabelling(self, sites):
        labelling = [None]*len(sites)
        for i in xrange(len(sites)):
            perm = np.random.permutation(len(self.labels))
            for j in perm:
                if self.D(sites[i], self.labels[j]) < float('inf'):
                    labelling[i] = j
                    break
        return labelling


    def OptimizeLabelling(self):
        x, y = np.where(self.mask != 0)
        sites = [[i, j] for (i, j) in zip(x, y)]
        labelling = self.InitializeLabelling(sites)
        
        return labelling
        
        

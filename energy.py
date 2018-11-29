# Image Completion using Statistics of Patch Offsets
# Author: Pranshu Gupta and Shrija Mishra

import cv2, numpy as np, sys, math, operator, maxflow, random, config as cfg
from scipy import ndimage
from time import time
from itertools import count, combinations

class Optimizer(object):
    def __init__(self, image, mask, labels):
        self.image = image/255.0
        self.mask = mask
        self.labels = labels
        x, y = np.where(self.mask != 0)
        sites = [[i, j] for (i, j) in zip(x, y)]
        self.sites = sites
        self.neighbors = []
        # self.vmem = np.zeros((len(sites), len(sites), len(labels), len(labels)))
        self.dmem = np.zeros((len(sites), len(labels)))

        self.InitializeD()
        # self.InitializeV()
        self.InitializeNeighbors()

    # def InitializeV():
    #     for i in xrange(len(self.sites)):
    #         for j in range(i,len(self.sites)):
    #             for l in xrange(len(self.labels)):
    #                 for m in range(l,len(self.labels)):
    #                     self.vmem[i,j,l,m] = self.V(self.sites[i], self.sites[j], self.labels[l], self.labels[m])
    
    def InitializeD(self):
        for i in xrange(len(self.sites)):
            for j in xrange(len(self.labels)):
                self.dmem[i,j] = self.D(self.sites[i], self.labels[j])
    
    def InitializeNeighbors(self):
        start = time()
        for i in xrange(len(self.sites)):
            ne = []
            neighbors = self.GetNeighbors(self.sites[i])
            for n in neighbors:
                if n in self.sites:
                    ne.append(self.sites.index(n))
            self.neighbors.append(ne)
        end = time()
        print "InitializeNeighbors execution time: ", end - start

    def D(self, site, offset):
        i, j = site[0] + offset[0], site[1] + offset[1]
        try:
            if self.mask[i][j] == 0:
                return 0
            return float('inf')
        except:
            return float('inf')

    def V(self, site1, site2, alpha, beta):
        start = time()
        x1a, y1a = site1[0] + alpha[0], site1[1] + alpha[1]
        x2a, y2a = site2[0] + alpha[0], site2[1] + alpha[1]
        x1b, y1b = site1[0] + beta[0], site1[1] + beta[1]
        x2b, y2b = site2[0] + beta[0], site2[1] + beta[1]
        try:
            if self.mask[x1a, y1a] == 0 and self.mask[x1b, y1b] == 0 and self.mask[x2a, y2a] == 0 and self.mask[x2a, y2a] == 0:
                return np.sum((self.image[x1a, y1a] - self.image[x1b, y1b])**2) + np.sum((self.image[x2a, y2a] - self.image[x2b, y2b])**2)
            return 1000000.0
        except:
            return 1000000.0
        

    def EnergyCalculator(self, labelling):
        start = time()
        num_labelling = self.dmem.shape[-1]
        
        # Sum of the unary terms.
        unary = np.sum([self.dmem[labelling==i,i].sum() for i in range(num_labelling)])

        # Binary terms.
        binary = 0
        for i in xrange(len(self.sites)):
            for j in self.neighbors[i]:
                binary += self.V(self.sites[i], self.sites[j], self.labels[labelling[i]], self.labels[labelling[j]])
        
        end = time()
        #print "EnergyCalculator execution time: ", end - start
        return unary + binary

    def GetNeighbors(self, site):
        return [[site[0]-1, site[1]], [site[0], site[1]-1], [site[0]+1, site[1]], [site[0], site[1]+1]]

    def AreNeighbors(self, site1, site2):
        if np.abs(site1[0]-site2[0]) < 2 and np.abs(site1[1]-site2[1]) < 2:
            return True
        return False 

    def InitializeLabelling(self, sites):
        start = time()
        labelling = [None]*len(sites)
        for i in xrange(len(sites)):
            perm = np.random.permutation(len(self.labels))
            for j in perm:
                if self.D(sites[i], self.labels[j]) < 1000000.0:
                    labelling[i] = j
                    break
        end = time()
        print "InitializeLabelling execution time: ", end - start
        return labelling

    def CreateGraph(self, alpha, beta, sites, labelling):
        start = time()
        ps = [i for i in range(len(sites)) if (labelling[i] == alpha or labelling[i] == beta)]
        v = len(ps)
        g = maxflow.Graph[float](v, 3*v)
        nodes = g.add_nodes(v)
        for i in range(v):
            pixel_pos = sites[ps[i]]
            # add the data terms here
            ta, tb = self.D(pixel_pos, self.labels[alpha]), self.D(pixel_pos, self.labels[beta])
            # add the smoothing terms here
            neighbor_list = self.GetNeighbors(pixel_pos)
            for neighbor in neighbor_list:
                try:
                    ind = sites.index(neighbor)
                    gamma, j = labelling[ind], ps.index(ind)
                    if gamma == beta and j > i:
                        epq = self.V(pixel_pos, neighbor, self.labels[alpha], self.labels[gamma])
                        g.add_edge(nodes[i], nodes[j], epq, epq)
                    elif gamma == alpha and j > i:
                        g.add_edge(nodes[i], nodes[j], 0, 0)
                    else:
                        ea = self.V(pixel_pos, neighbor, self.labels[alpha], self.labels[gamma])
                        eb = self.V(pixel_pos, neighbor, self.labels[beta], self.labels[gamma])
                        ta, tb = ta + ea, tb + eb
                except:
                    pass                                    
            g.add_tedge(nodes[i], ta, tb)
        end = time()
        #print "CreateGraph execution time: ", end - start
        return g, nodes

    def OptimizeLabelling(self):
        x, y = np.where(self.mask != 0)
        sites = [[i, j] for (i, j) in zip(x, y)]
        labelling = self.InitializeLabelling(sites)
        E1 = self.EnergyCalculator(labelling)
        iter_count = 0
        while(True):
            success = 0
            for alpha, beta in combinations(range(len(self.labels)), 2):
                ps = [i for i in range(len(sites)) if (labelling[i] == alpha or labelling[i] == beta)]
                if len(ps) > 0:
                    g, nodes = self.CreateGraph(alpha, beta, sites, labelling)
                    # start = time()
                    flow = g.maxflow()
                    # end = time()
                    # print "MaxFlow execution time: ", end - start
                    temp_labelling = labelling
                    for i in range(len(ps)):
                        if g.get_segment(nodes[i]) == 0:
                            temp_labelling[ps[i]] = alpha
                        elif g.get_segment(nodes[i]) == 1:
                            temp_labelling[ps[i]] = beta
                    E2 = self.EnergyCalculator(temp_labelling)
                    if  E2 < E1:
                        print(E1, E2)
                        E1 = E2
                        # Set the new labelling
                        labelling = temp_labelling
                        #print(labelling)
                        success = 1
            print labelling
            if success != 1 or iter_count >= cfg.MAX_ITER:
                return labelling
            iter_count += 1
            print("Iterations: ", iter_count)
        
        

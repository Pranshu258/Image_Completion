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
        self.dmem = np.zeros((len(sites), len(labels)))
        self.InitializeD()
        self.InitializeNeighbors()

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
        
    def IsLowerEnergy(self, nodes, labelling1, labelling2):
        updatedNodes = np.where(labelling1 != labelling2)[0]
        diff = 0.0
        for node in updatedNodes:
            if self.D(self.sites[node], self.labels[labelling2[node]]) < float('inf'):
                for n in self.neighbors[node]:
                    if n in updatedNodes:
                        if n > node:
                            diff += self.V(self.sites[node], self.sites[n], self.labels[labelling2[node]], self.labels[labelling2[n]]) - self.V(self.sites[node], self.sites[n], self.labels[labelling1[node]], self.labels[labelling1[n]])
                    else:
                        diff += self.V(self.sites[node], self.sites[n], self.labels[labelling2[node]], self.labels[labelling2[n]]) - self.V(self.sites[node], self.sites[n], self.labels[labelling1[node]], self.labels[labelling1[n]])
            else:
                return False
        if diff < 0:
            return True
        return False

    def GetNeighbors(self, site):
        return [[site[0]-1, site[1]], [site[0], site[1]-1], [site[0]+1, site[1]], [site[0], site[1]+1]]

    def AreNeighbors(self, site1, site2):
        if np.abs(site1[0]-site2[0]) < 2 and np.abs(site1[1]-site2[1]) < 2:
            return True
        return False 

    def InitializeLabelling(self):
        start = time()
        labelling = [None]*len(self.sites)
        for i in xrange(len(self.sites)):
            perm = np.random.permutation(len(self.labels))
            for j in perm:
                if self.D(self.sites[i], self.labels[j]) < 1000000.0:
                    labelling[i] = j
                    break     
        self.sites = [self.sites[i] for i in range(len(self.sites)) if labelling[i] != None]
        labelling = [label for label in labelling if label != None]         
        end = time()
        print "InitializeLabelling execution time: ", end - start
        return self.sites, np.array(labelling)

    def CreateGraphABS(self, alpha, beta, ps, labelling):
        start = time()
        v = len(ps)
        g = maxflow.Graph[float](v, 3*v)
        nodes = g.add_nodes(v)
        for i in range(v):
            # add the data terms here
            ta, tb = self.D(self.sites[ps[i]], self.labels[alpha]), self.D(self.sites[ps[i]], self.labels[beta])
            # add the smoothing terms here
            neighbor_list = self.neighbors[ps[i]]
            for ind in neighbor_list:
                try:
                    a, b, j = labelling[ps[i]], labelling[ind], ps.index(ind)
                    if j > i and (b == alpha or b == beta):
                        epq = self.V(self.sites[ps[i]], self.sites[ps[j]], self.labels[alpha], self.labels[beta])
                        g.add_edge(nodes[i], nodes[j], epq, epq)
                    else:
                        ea = self.V(self.sites[ps[i]], self.sites[ps[j]], self.labels[alpha], self.labels[b])
                        eb = self.V(self.sites[ps[i]], self.sites[ps[j]], self.labels[beta], self.labels[b])
                        ta, tb = ta + ea, tb + eb
                except Exception as e:
                    pass                                  
            g.add_tedge(nodes[i], ta, tb)
        end = time()
        #print "CreateGraph execution time: ", end - start
        return g, nodes

    def CreateGraphAE(self, alpha, labelling):
        start = time()
        v = len(self.sites)
        g = maxflow.Graph[float](2*v, 4*v)
        nodes = g.add_nodes(v)
        for i in range(v):
            ta, tb = self.D(self.sites[i], self.labels[alpha]), float('inf')
            if labelling[i] != alpha:
                tb = self.D(self.sites[i], self.labels[labelling[i]])
            g.add_tedge(nodes[i], ta, tb)
            neighbor_list = self.neighbors[i]
            for j in neighbor_list:
                try:
                    if labelling[i] == labelling[j] and j > i:
                        epq = self.V(self.sites[i], self.sites[j], self.labels[labelling[i]], self.labels[alpha])
                        g.add_edge(nodes[i], nodes[j], epq, epq)
                    elif j > i:
                        aux_nodes = g.add_nodes(1)
                        epa = self.V(self.sites[i], self.sites[j], self.labels[labelling[i]], self.labels[alpha])
                        eaq = self.V(self.sites[i], self.sites[j], self.labels[labelling[j]], self.labels[alpha])
                        epq = self.V(self.sites[i], self.sites[j], self.labels[labelling[i]], self.labels[labelling[j]])
                        g.add_edge(nodes[i], aux_nodes[0], epa, epa)
                        g.add_edge(nodes[j], aux_nodes[0], eaq, eaq)
                        g.add_tedge(aux_nodes[0], float('inf'), epq)
                except Exception as e:
                    print(e)
        end = time()
        #print "CreateGraph execution time: ", end - start
        return g, nodes            

    def OptimizeLabellingABS(self, labelling):
        labellings = np.zeros((2, len(self.sites)), dtype=int)
        labellings[0] = labellings[1] = np.copy(labelling)
        iter_count = 0
        while(True):
            start = time()
            success = 0
            for alpha, beta in combinations(range(len(self.labels)), 2):
                ps = [i for i in range(len(self.sites)) if (labellings[0][i] == alpha or labellings[0][i] == beta)]
                if len(ps) > 0:
                    g, nodes = self.CreateGraphABS(alpha, beta, ps, labellings[0])
                    flow = g.maxflow()
                    for i in range(len(ps)):
                        gamma = g.get_segment(nodes[i])
                        labellings[1, ps[i]] = alpha*(1-gamma) + beta*gamma
                    if self.IsLowerEnergy(ps, labellings[0], labellings[1]):
                        labellings[0, ps] = labellings[1, ps] 
                        success = 1
                    else:
                        labellings[1, ps] = labellings[0, ps]                      
            iter_count += 1
            end = time()
            print "ABS Iteration " + str(iter_count) + " execution time: ", str(end - start) 
            if success != 1 or iter_count >= cfg.MAX_ITER:
                break
        return labellings[0]

    def OptimizeLabellingAE(self, labelling):
        labellings = np.zeros((2, len(self.sites)), dtype=int)
        labellings[0] = labellings[1] = np.copy(labelling)
        iter_count = 0
        while(True):
            start = time()
            success = 0
            for alpha in xrange(len(self.labels)):
                g, nodes = self.CreateGraphAE(alpha, labellings[0])
                flow = g.maxflow()
                for i in range(len(self.sites)):
                    gamma = g.get_segment(nodes[i])
                    labellings[1, i] = alpha*(1-gamma) + labellings[1, i]*gamma
                if self.IsLowerEnergy(range(len(self.sites)), labellings[0], labellings[1]):
                    labellings[0] = labellings[1] 
                    success = 1
                else:
                    labellings[1] = labellings[0]     
            iter_count += 1
            end = time()
            print "AE Iteration " + str(iter_count) + " execution time: ", str(end - start) 
            if success != 1 or iter_count >= cfg.MAX_ITER:
                break
        return labellings[0]
        
        

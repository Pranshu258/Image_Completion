# Image Completion using Statistics of Patch Offsets
# Author: Pranshu Gupta and Shrija Mishra

import cv2, numpy as np, sys, math, operator, maxflow, random
from scipy import ndimage
from itertools import count, combinations

class Optimizer(object):
    def __init__(self, image, mask, labels):
        self.image = image
        self.mask = mask
        self.labels = labels

    def D(self, site, offset):
        i, j = site[0] + offset[0], site[1] + site[1]
        try:
            if self.mask[int(t_x)][int(t_y)] == 0:
                return 0
        except:
            return float('inf')

    def V(self, site1, site2, alpha, beta):
        x1a, y1a = site1[0] + alpha[0], site1[1] + alpha[1]
        x2a, y2a = site2[0] + alpha[0], site2[1] + alpha[1]
        x1b, y1b = site1[0] + beta[0], site1[1] + beta[1]
        x2b, y2b = site2[0] + beta[0], site2[1] + beta[1]
        try:
            if mask[x1a, y1a] == 0 and mask[x1b, y1b] == 0 and mask[x2a, y2a] == 0 and mask[x2a, y2a] == 0:
                return np.sum((image[x1a, y1a] - image[x1b, y1b])**2) + np.sum((image[x2a, y2a] - image[x2b, y2b])**2)
        except:
            return float('inf')

def EnergyCalculator(sites, offsets, labelling, height, width, mask, image):
	e = 0
	for i in range(len(sites)):
		e = e + D(sites[i], offsets[labelling[i]], height, width, mask)
		neighbors = get_neighbors(sites[i])
		for n in neighbors:
			if n in sites:
				if sites.index(n) > i:
					e = e + V(sites[i], n, offsets[labelling[i]], offsets[labelling[sites.index(n)]], height, width, image)
	return e

def GetNeighbors(pix):
	return [list(pix + np.array([-1,0])), list(pix + np.array([1,0])), list(pix + np.array([0,-1])), list(pix + np.array([0,1]))]

def AreNeighbors(pix1, pix2):
	return (pix1 == pix2 + np.array([-1,0])).all() or (pix1 == pix2 + np.array([1,0])).all() or (pix1 == pix2 + np.array([0,-1])).all() or (pix1 == pix2 + np.array([0,1])).all()

def OptimizeLabels(offsets, mask, image, tau):
	#Start with arbitrary labelling
    height, width = mask.shape
    global sites
    sites, labelling = [], []
    for i in range(height*width):
        x, y = math.floor(i/width), i%width
        if mask[int(x)][int(y)] != 0:
            sites.append([x,y])
            label = len(offsets)-1
            while D([x,y],offsets[label], height, width, mask) == float('inf') and label >= 0:
                label -= 1
            labelling.append(label)
    sea = np.array([[int(-2*tau),0],[0,int(-2*tau)],[0,int(2*tau)],[int(2*tau),0]])
    for i in range(len(labelling)):
		if labelling[i] == -1:
			# print("-1 label detected ")
			for j in range(len(sea)):
				if D([x,y],sea[j], height, width) != float('inf'):
					labelling[i] = len(offsets)+j
					#print("Label Improved: " ,len(offsets)+j)
					break
    print type(offsets), type(sea)
    print offsets.shape, sea.shape
    offsets = np.concatenate((offsets, sea), axis = 0)
    # for s in sea:
    #     offsets = np.concatenate((offsets, s), axis = 0)
        #offsets.append(s)

	# Calculate the initial energy
    E1 = energy_calculator(sites, offsets, labelling, height, width, mask, image)
	#print("Initial Labels: \n", labelling)
	#print("Initial Energy: ", E1)
	# MRF optimization using alpha-beta swap moves
    iter_count = 0
    while(True):
        print("--------------------------------------------------")
        success = 0
        for alpha, beta in combinations(range(len(offsets)), 2):
            pa = [i for i in range(len(sites)) if labelling[i] == alpha]
            pb = [i for i in range(len(sites)) if labelling[i] == beta]
            ps = pa + pb
            v, e = len(ps), 3*len(ps)-1
            if v > 0:
				#print(" ")
				# Construct the graph
				g = maxflow.Graph[float](v, e)
				nodes = g.add_nodes(v)
				#print("Nodes: ", v)
				# Add the edges to the source and sink
				for i in range(v):
					pixel_pos = sites[ps[i]]
					# add the data terms here
					ta = D(pixel_pos, offsets[alpha], height, width, mask) 
					tb = D(pixel_pos, offsets[beta], height, width, mask) 
					# add the smoothing terms here
					neighbor_list = get_neighbors(pixel_pos)
					for neighbor in neighbor_list:
						if neighbor in sites:
							ind = sites.index(neighbor)
							gamma = labelling[ind]
							if ind not in ps:
								ta = ta + V(pixel_pos, neighbor, offsets[alpha], offsets[gamma], height, width, image)
								tb = tb + V(pixel_pos, neighbor, offsets[beta], offsets[gamma], height, width, image)
					if ta != float('inf') or tb != float('inf'):
						g.add_tedge(nodes[i], ta, tb)
						#print("Added edges, src, sink: ", ta, tb)
				#print("Added the edges to source and sink")
				# Add the edges to neighbors
				for p, q in combinations(range(len(ps)), 2):
					pixel_pos1, pixel_pos2 = sites[ps[p]], sites[ps[q]]
					# If these two pixels are neighbors then add an edge between them
					if are_neighbors(pixel_pos1, pixel_pos2):
						epq = V(pixel_pos1, pixel_pos2, offsets[alpha], offsets[beta], height, width, image)
						if epq != float('inf'):
							g.add_edge(nodes[p], nodes[q], epq, epq)
							#print("Added edges, neigh: ", epq)
				#print("Added edges for neighbors")
				# Now execute the max flow algorithm
				flow = g.maxflow()
				#print("Computed MaxFlow", flow)
				# Get the labelling for the sites in ps
				# if get_segment() gives 0 then alpha, if get_segment() gives 1 then beta
				# Find the new energy
				temp_labelling = [i for i in labelling]
				for i in range(v):
					if g.get_segment(nodes[i]) == 0:
						temp_labelling[ps[i]] = alpha
					elif g.get_segment(nodes[i]) == 1:
						temp_labelling[ps[i]] = beta
				#print(temp_labelling)
				E2 = energy_calculator(sites, offsets, temp_labelling, height, width, mask, image)
				del g
				if E2 < E1:
					print("Graph (l1,l2,flow): ", alpha, beta, flow)
					print("Energy Minimized: ", E1, " -> ", E2)
					E1 = E2
					# Set the new labelling
					labelling = [i for i in temp_labelling]
					#print(labelling)
					success = 1
        if success != 1 or iter_count >= max_iter:
			return labelling, E1
        iter_count += 1
        print("Iterations: ", iter_count)

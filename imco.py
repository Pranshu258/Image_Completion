# Image Completion using Statistics of Patch Offsets
# Author: Pranshu Gupta
# -----------------------------------------------------------------------------

import cv2, numpy as np, sys, math, operator, maxflow, random
from scipy import ndimage
from itertools import count, combinations

# -----------------------------------------------------------------------------

if len(sys.argv) == 2:
	name = str(sys.argv[1])
	filename = "img/" + name
	maskname = "mask/" + name
	output = "res/" + name
else:
	print("usage: python3 imco.py filename")
	exit()

print(filename, maskname)

# Read the image and mask files
img = cv2.imread(filename)
mask = cv2.imread(maskname, 0)
height, width = mask.shape

K = 60 					# Number of offsets to keep in the final set
max_iter = 10 			# Maximum iterations in labelling optimization
random.seed()
tau = None

# -----------------------------------------------------------------------------

def get_mask_bb():
	h1, h2, w1, w2 = height, 0, width, 0
	for i in range(height):
		for j in range(width):
			if mask[i][j] != 0:
				if i > h2:
					h2 = i
				if i < h1:
					h1 = i
				if j > w2:
					w2 = j
				if j < w1:
					w1 = j
	# return h2-h1, w2-w1
	global tau
	tau = max(h2-h1, w2-w1)/15
	return h1, h2, w1, w2


def patchmatch(valid_patch):
	# Randomly Initialize the offsets
	print('Randomly Initialize the offsets ...')
	offsets = [[0,0] for i in range(height*width)]
	dist = [float('inf') for i in range(height*width)]
	for i in range(height*width):
		x, y = math.floor(i/width), i%width
		if valid_patch[x][y] == 1:
			p, q = random.randrange(0,height), random.randrange(0,width)
			# Keep doing this until we get a valid offset
			while valid_patch[p][q] != 1 or np.linalg.norm(np.array([p,q])-np.array([x,y])) < tau:
				p, q = random.randrange(0,height), random.randrange(0,width)
			offsets[i] = np.array([p-x, q-y])
			patch1 = (np.array(img[x-4:x+4, y-4:y+4])).flatten()
			patch2 = (np.array(img[p-4:p+4, q-4:q+4])).flatten()
			dist[i] = np.linalg.norm(patch1-patch2)
	# print(offsets)
	# Propagation and random search
	print("Patch Offset Propagation Begins ...")
	sea = np.array([[-1,0],[0,-1],[0,1],[1,0]])
	for it in range(5):
		for i in range(height*width):
			x, y = math.floor(i/width), i%width
			if valid_patch[x][y] == 1:
				# print(i, offsets[i])
				patch1 = (np.array(img[x-4:x+4, y-4:y+4])).flatten()
				new_off, new_dist = offsets[i], dist[i]
				# Propagation
				m = 1
				for s in sea:
					test_off = offsets[i] + m*np.array(s)
					nx, ny = x+test_off[0], y+test_off[1]
					# if the patch at new offset position is valid and a better match than what we have, update
					if nx >= 0 and nx < height and ny >= 0 and ny < width and np.linalg.norm(np.array([nx,ny])-np.array([x,y])) > tau:
						if valid_patch[nx][ny] == 1:
							patch2 = (np.array(img[nx-4:nx+4, ny-4:ny+4])).flatten()
							test_dist = np.linalg.norm(patch1-patch2)
							if test_dist < dist[i]:
								new_off = test_off
								new_dist = test_dist

				offsets[i], dist[i] = new_off, new_dist
				# print(i, offsets[i])
				# Random Search
				m = 2
				while m < 1000:
					s = sea[random.randrange(0,4)]
					test_off = offsets[i] + m*np.array(s)
					nx, ny = x+test_off[0], y+test_off[1]
					if nx >= 0 and nx < height and ny >= 0 and ny < width and np.linalg.norm(np.array([nx,ny])-np.array([x,y])) > tau:
						if valid_patch[nx][ny] == 1:
							patch2 = (np.array(img[nx-4:nx+4, ny-4:ny+4])).flatten()
							test_dist = np.linalg.norm(patch1-patch2)
							if test_dist < dist[i]:
								new_off = test_off
								new_dist = test_dist
					m = m*2

				offsets[i], dist[i] = new_off, new_dist
				
		print("Iterations Done: ", it+1)
	return offsets

def patch_match(validity):
	# offsets are nitialized to (0,0)
	offsets = [[0,0] for i in range(height*width)]
	for i in range(height*width):
		x, y = math.floor(i/width), i%width
		if mask[x][y] == 0:
			if validity[x][y] == 1:
				patch = (np.array(img[x-4:x+4, y-4:y+4])).flatten()
				r1, c1 = max(0, int(x-bb_h*1.5)), max(0, int(y-bb_w*1.5))
				r2, c2 = min(int(r1+bb_h*3), height), min(int(c1+bb_w*3), width)
				d = float('inf')
				tau = max(r2-r1, c2-c1)/15
				for r in range(r1,r2):
					for c in range(c1,c2):
						if mask[r][c] == 0 and np.linalg.norm(np.array([r,c])-np.array([x,y])) > tau:
							p = (np.array(img[r-4:r+4, c-4:c+4])).flatten()
							v = ((np.array(mask[r-4:r+4, c-4:c+4])).flatten()).sum()
							if v == 0 and len(p) == 192:
								dist = np.linalg.norm(patch-p)
								if d > dist:
									d = dist
									offsets[i] = [x-r, y-c]
									#print(";;;;;;;;;;;;;;;;", i, ": ", offsets[i])
		if i % width == 0 and i != 0:
			print("Matching Done: ", int(i/width), " rows")
	return offsets

# pick_offsets: returns the 60 peak offsets in the image
def peak_offsets(offsets):
	x, y = [], []
	for point in offsets:
		x.append(point[0])
		y.append(point[1])
	bins = [[i for i in range(-1*height,height)],[i for i in range(-1*width,width)]]
	hist = (np.histogram2d(x,y,bins))[0]
	# Remove the [0,0] matches
	# print(hist[height][width])
	hist[height][width] = 0
	# print(hist[height][width])
	smooth_hist = ndimage.filters.gaussian_filter(hist,2**0.5)
	# Pick the best 60 local maxima
	peaks = {}
	for i in range(4,len(smooth_hist)):
		for j in range(4,len(smooth_hist[0])):
			max_in_box = np.max(smooth_hist[i-4:i+4, j-4:j+4])
			if max_in_box != smooth_hist[i][j]:
				peaks[(i-height,j-width)] = smooth_hist[i][j]
	# Sort these local maxima and return top 60
	sorted_peaks = sorted(peaks.items(), key = operator.itemgetter(1))
	sorted_peaks = sorted_peaks[-1*K:]
	peak_offsets = [p[0] for p in sorted_peaks]
	# Now remove those which are not suitable as offsets
	# peaks = []
	# for offset in peak_offsets:
	# 	if not (offset[0]+h1 < 0 or offset[0]+h2 >= height or offset[1]+w1 < 0 or offset[1]+w2 >= width):
	# 		peaks.append(offset)
	return peak_offsets

def D(pixel_pos, offset):
	x,y = pixel_pos[0], pixel_pos[1]
	x_off, y_off = offset[0], offset[1]
	t_x, t_y = x+x_off, y+y_off
	# Check if t_x, t_y are within the image
	if t_x < height and t_y < width and t_x > -1 and t_y > -1:
		if mask[t_x][t_y] != 0:
			return float('inf')
		return 0
	return float('inf')

def V(pixel_pos1, pixel_pos2, alpha, beta):
	x1,y1 = pixel_pos1[0]+alpha[0], pixel_pos1[1]+alpha[1]
	x2,y2 = pixel_pos2[0]+beta[0], pixel_pos2[1]+beta[1]
	if x1 > -1 and x1 < height and y1 > -1 and y1 < width and x2 > -1 and x2 < height and y2 > -1 and y2 < width:
		return np.linalg.norm(img[x1][y1]-img[x2][y2])
	#print("V returned inifinity")
	return float('inf')


def energy_calculator(sites, offsets, labelling):
	e = 0
	for i in range(len(sites)):
		e = e + D(sites[i], offsets[labelling[i]])
		neighbors = get_neighbors(sites[i])
		for n in neighbors:
			if n in sites:
				if sites.index(n) > i:
					e = e + V(sites[i], n, offsets[labelling[i]], offsets[labelling[sites.index(n)]])
	return e

def get_neighbors(pix):
	return [list(pix + np.array([-1,0])), list(pix + np.array([1,0])), list(pix + np.array([0,-1])), list(pix + np.array([0,1]))]

def are_neighbors(pix1, pix2):
	return (pix1 == pix2 + np.array([-1,0])).all() or (pix1 == pix2 + np.array([1,0])).all() or (pix1 == pix2 + np.array([0,-1])).all() or (pix1 == pix2 + np.array([0,1])).all()

def optimize_labels(offsets, valid_patch):
	#Start with arbitrary labelling
	global sites
	sites, labelling = [], []
	for i in range(height*width):
		x, y = math.floor(i/width), i%width
		if mask[x][y] != 0:
			sites.append([x,y])
			label = len(offsets)-1
			while D([x,y],offsets[label]) == float('inf') and label >= 0:
				label -= 1
			labelling.append(label)
	sea = np.array([[int(-2*tau),0],[0,int(-2*tau)],[0,int(2*tau)],[int(2*tau),0]])
	for i in range(len(labelling)):
		if labelling[i] == -1:
			# print("-1 label detected ")
			for j in range(len(sea)):
				if D([x,y],sea[j]) != float('inf'):
					labelling[i] = len(offsets)+j
					#print("Label Improved: " ,len(offsets)+j)
					break
	for s in sea:
		offsets.append(s)

	# Calculate the initial energy
	E1 = energy_calculator(sites, offsets, labelling)
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
					ta = D(pixel_pos, offsets[alpha]) 
					tb = D(pixel_pos, offsets[beta]) 
					# add the smoothing terms here
					neighbor_list = get_neighbors(pixel_pos)
					for neighbor in neighbor_list:
						if neighbor in sites:
							ind = sites.index(neighbor)
							gamma = labelling[ind]
							if ind not in ps:
								ta = ta + V(pixel_pos, neighbor, offsets[alpha], offsets[gamma])
								tb = tb + V(pixel_pos, neighbor, offsets[beta], offsets[gamma])
					if ta != float('inf') or tb != float('inf'):
						g.add_tedge(nodes[i], ta, tb)
						#print("Added edges, src, sink: ", ta, tb)
				#print("Added the edges to source and sink")
				# Add the edges to neighbors
				for p, q in combinations(range(len(ps)), 2):
					pixel_pos1, pixel_pos2 = sites[ps[p]], sites[ps[q]]
					# If these two pixels are neighbors then add an edge between them
					if are_neighbors(pixel_pos1, pixel_pos2):
						epq = V(pixel_pos1, pixel_pos2, offsets[alpha], offsets[beta])
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
				E2 = energy_calculator(sites, offsets, temp_labelling)
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

def complete(img, sites, offsets, Labels):
	final_img = img
	for i in range(len(sites)):
		x,y = sites[i][0], sites[i][1]
		n_x, n_y = x + offsets[Labels[i]][0], y + offsets[Labels[i]][1]
		final_img[x][y] = img[n_x][n_y]
	return final_img

def get_validity():
	valid_patch = [[0 for i in range(width)] for j in range(height)]
	for i in range(height*width):
		x, y = math.floor(i/width), i%width
		if mask[x][y] == 0:
			patch = (np.array(img[x-4:x+4, y-4:y+4])).flatten()
			valid = ((np.array(mask[x-4:x+4, y-4:y+4])).flatten()).sum()
			if valid == 0 and len(patch) == 192:
				valid_patch[x][y] = 1
	return valid_patch

# -----------------------------------------------------------------------------

print("Image Size: ", height, width)
h1, h2, w1, w2 = get_mask_bb()
bb_h, bb_w = h2-h1, w2-w1
print("BB Size: ", bb_h, bb_w)
#print("Computing the Offsets ...")
validity = get_validity()
offsets = patch_match(validity)
peaks = peak_offsets(offsets)
print("Offsets: ", len(peaks))
Labels, E = optimize_labels(peaks, validity)
print("Final Labels: \n", Labels)
print('Final Energy: ', E)
# Construct the image now
final_img = complete(img, sites, peaks, Labels)
cv2.imwrite(output, final_img)

# -----------------------------------------------------------------------------
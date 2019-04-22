import numpy as np

res_x = 480
res_y = 360
num_pixels = res_x * res_y

def raytrace():
	view = np.zeros((num_pixels, 3))
	view[:,2] = 1.0

raytrace()

'''
Author:			Manuel Vasquez
Date:			January 17, 2019
Images:			image1.png, image2.png
Task:			Filter -> show, explain what functions the filter performed,
				and explain the differences between the original and filtered image.
Utilities:		Convolution built-in functions
'''


from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import time


def pm(a):
	for b in a:
		for c in b:
			print(f'{c:0.2f}', '' if c > 9 else ' ', end='| ')
		print()
	print('-'*100)


def g(x, y, sigma):
	return (1/(2*np.pi*sigma**2))*np.e**(-1*(x**2+y**2)/(2*sigma**2))


def apply_kernel(input, size, kernel, div=1):
	k = size//2
	step = []
	for i in range(size):
		yroll = np.copy(input)
		yroll = np.roll(yroll, i-k, axis=0)
		for j in range(size):
			xroll = np.copy(yroll)
			xroll = np.roll(xroll, j-k, axis=1)*kernel[i, j]
			step.append(xroll)
	
	step = np.array(step)
	stepsum = np.sum(step, axis=0) / div
	return stepsum


def my_box(input, size):
	input = np.array(input, dtype=np.float64)
	kernel = np.ones((size, size), dtype=np.float64)
	return apply_kernel(input, size, kernel, np.sum(kernel))


def my_gaussian(input, sigma):
	input = np.array(input, dtype=np.float64)
	size = 6*sigma+1
	k = size//2
	kernel = [[0]*size for _ in range(size)]
	for y in range(-1*k, k+1):
		for x in range(-1*k, k+1):
			kernel[x+k][y+k] = g(x, y, sigma)
	kernel = np.array(kernel, dtype=np.float64)
	return apply_kernel(input, size, kernel, np.sum(kernel))


def my_gradient(input, difference, axis):
	kernel = None
	if difference == 'backward':
		if axis == 'x':
			kernel=np.array([[-1,1,0],[-1,1,0],[-1,1,0]], dtype=np.float64)
		elif axis == 'y':
			kernel=np.array([[-1,-1,-1],[1,1,1],[0,0,0]], dtype=np.float64)
	elif difference == 'forward':
		if axis == 'x':
			kernel=np.array([[0,1,-1],[0,1,-1],[0,1,-1]], dtype=np.float64)
		elif axis == 'y':
			kernel=np.array([[0,0,0],[1,1,1],[-1,-1,-1]], dtype=np.float64)
	elif difference == 'central':
		if axis == 'x':
			kernel=np.array([[1,0,-1],[1,0,-1],[1,0,-1]], dtype=np.float64)
		elif axis == 'y':
			kernel=np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=np.float64)
	return apply_kernel(input, 3, kernel, 3)


def my_sobel(input):
	x = apply_kernel(input, 3, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64))
	y = apply_kernel(input, 3, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64))
	return x,y


def gu(u, sigma):
	return (1/(np.sqrt(2*np.pi*sigma**2)))*np.e**(-1*(u**2)/(2*sigma**2))


def my_fastgaussian(input, sigma):
	input = np.array(input, dtype=np.float64)
	size = 6*sigma+1
	k = size//2
	kernel = [0]*size
	for i in range(-1*k, k+1):
		kernel[i+k] = gu(i, sigma)
	kernel = np.array(kernel, dtype=np.float64)
	
	'''
	temp = np.copy(input)
	for row in range(k, len(input)-k):
		for col in range(len(input[0])):
			sum = 0
			for i in range(-1*k,k+1):
				sum += temp[row+i][col]*kernel[i+k]
			temp[row][col] = sum//np.sum(kernel)
	
	for row in range(len(input)):
		for col in range(k, len(input[0])-k):
			sum = 0
			for i in range(-1*k, k+1):
				sum += temp[row][col+i]*kernel[i+k]
			temp[row][col] = sum//np.sum(kernel)
	'''

	step = []
	for i in range(size):
		roll = np.copy(input)
		step.append(np.roll(roll, i-k, axis=0)*kernel[i])
	step = np.array(step)
	stepsum = np.sum(step, axis=0) / np.sum(kernel)

	step = []
	for i in range(size):
		roll = np.copy(stepsum)
		step.append(np.roll(roll, i-k, axis=1)*kernel[i])
	step = np.array(step)
	stepsum = np.sum(step, axis=0) / np.sum(kernel)
	
	return stepsum


def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data), len(data[0])))
    for i in range(len(data)):
        for j in range(len(data[0])):
            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for _ in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])
            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final


def my_median(input, size):
	k = size // 2
	input = np.array(input, dtype=np.float64)
	res = np.empty_like(input, dtype=np.float64)
	for i in range(k, len(input)-k):
		for j in range(k, len(input[0])-k):
			res[i][j] = np.median(input[i-k:i+k+1, j-k:j+k+1])
	return res[k:-k, k:-k]


def my_magn_orien(xgradient, ygradient):
	magnitude = np.empty_like(xgradient)
	orientation = np.empty_like(xgradient)
	for row in range(len(xgradient)):
		for col in range(len(xgradient[0])):
			magnitude[row][col] = np.sqrt(xgradient[row][col]**2 + ygradient[row][col]**2)
			angle = np.abs(np.arctan2(ygradient[row][col], xgradient[row][col]) * 180/np.pi)
			if angle >= 0 and angle <= 22.5 or angle > 157.5 and angle <= 180:
				angle = 0
			elif angle > 22.5 and angle <= 67.5:
				angle = 45
			elif angle > 67.5 and angle <= 112.5:
				angle = 90
			elif angle > 112.5 and angle <= 157.5:
				angle = 135
			else:
				print('Something broke '*5)
			orientation[row][col] = angle
	return magnitude, orientation


def max(a, b, c):
	if b >= a and b >= c:
		return True
	return False


def my_nonmaxsuppression(magnitude, orientation):
	res = np.copy(magnitude)
	for row in range(len(magnitude)):
		for col in range(len(magnitude[0])):
			# west east
			if orientation[row][col] == 0:
				if col == 0:
					if not max(magnitude[row][-1], magnitude[row][col], magnitude[row][col+1]):
						res[row][col] = 0
				elif col == len(magnitude[0])-1:
					if not max(magnitude[row][col-1], magnitude[row][col], magnitude[row][0]):
						res[row][col] = 0
				elif not max(magnitude[row][col-1], magnitude[row][col], magnitude[row][col+1]):
					res[row][col] = 0
				elif max(magnitude[row][col-1], magnitude[row][col], magnitude[row][col+1]):
					res[row][col] = res[row][col]
				else:
					print('west east error')

			# southwest northeast
			elif orientation[row][col] == 45:
				if row == 0:
					if col == 0:
						if not max(magnitude[row+1][-1], magnitude[row][col], magnitude[-1][col+1]):
							res[row][col] = 0
					elif col == len(magnitude[0])-1:
						if not max(magnitude[row+1][col-1], magnitude[row][col], magnitude[-1][0]):
							res[row][col] = 0
					elif not max(magnitude[row+1][col-1], magnitude[row][col], magnitude[-1][col+1]):
						res[row][col] = 0
				elif row == len(magnitude)-1:
					if col == 0:
						if not max(magnitude[0][-1], magnitude[row][col], magnitude[row-1][col+1]):
							res[row][col] = 0
					elif col == len(magnitude[0])-1:
						if not max(magnitude[0][col-1], magnitude[row][col], magnitude[row-1][0]):
							res[row][col] = 0
					elif not max(magnitude[0][col-1], magnitude[row][col], magnitude[row-1][col+1]):
						res[row][col] = 0
				elif col == 0:
					if not max(magnitude[row+1][-1], magnitude[row][col], magnitude[row-1][col+1]):
						res[row][col] = 0
				elif col == len(magnitude[0])-1:
					if not max(magnitude[row+1][col-1], magnitude[row][col], magnitude[row-1][0]):
						res[row][col] = 0
				elif not max(magnitude[row+1][col-1], magnitude[row][col], magnitude[row-1][col+1]):
					res[row][col] = 0
				elif max(magnitude[row+1][col-1], magnitude[row][col], magnitude[row-1][col+1]):
					res[row][col] = res[row][col]
				else:
					print('southwest northeast error')

			# south north
			elif orientation[row][col] == 90:
				if row == 0:
					if not max(magnitude[-1][col], magnitude[row][col], magnitude[row+1][col]):
						res[row][col] = 0
				elif row == len(magnitude)-1:
					if not max(magnitude[row-1][col], magnitude[row][col], magnitude[0][col]):
						res[row][col] = 0
				elif not max(magnitude[row-1][col], magnitude[row][col], magnitude[row+1][col]):
					res[row][col] = 0
				elif max(magnitude[row-1][col], magnitude[row][col], magnitude[row+1][col]):
					res[row][col] = res[row][col]
				else:
					print('south north error')

			# southeast northwest
			elif orientation[row][col] == 135:
				if row == 0:
					if col == 0:
						if not max(magnitude[-1][-1], magnitude[row][col], magnitude[row+1][col+1]):
							res[row][col] = 0
					elif col == len(magnitude[0])-1:
						if not max(magnitude[-1][col-1], magnitude[row][col], magnitude[row+1][0]):
							res[row][col] = 0
					elif not max(magnitude[-1][col-1], magnitude[row][col], magnitude[row+1][col+1]):
						res[row][col] = 0
				elif row == len(magnitude)-1:
					if col == 0:
						if not max(magnitude[row-1][-1], magnitude[row][col], magnitude[0][col+1]):
							res[row][col] = 0
					elif col == len(magnitude[0])-1:
						if not max(magnitude[row-1][col-1], magnitude[row][col], magnitude[0][0]):
							res[row][col] = 0
					elif not max(magnitude[row-1][col-1], magnitude[row][col], magnitude[0][col+1]):
						res[row][col] = 0
				elif col == 0:
					if not max(magnitude[row-1][-1], magnitude[row][col], magnitude[row+1][col+1]):
						res[row][col] = 0
				elif col == len(magnitude[0])-1:
					if not max(magnitude[row-1][col-1], magnitude[row][col], magnitude[row+1][0]):
						res[row][col] = 0
				elif not max(magnitude[row-1][col-1], magnitude[row][col], magnitude[row+1][col+1]):
					res[row][col] = 0
				elif max(magnitude[row-1][col-1], magnitude[row][col], magnitude[row+1][col+1]):
					res[row][col] = res[row][col]
				else:
					print('southeast northwest error')
	return res
				


def my_canny(input, sigma):
	smoothed = my_fastgaussian(input, sigma)
	xsobel, ysobel = my_sobel(smoothed)
	magnitude, orientation = my_magn_orien(xsobel, ysobel)
	suppressed = my_nonmaxsuppression(magnitude, orientation)

	return suppressed


def my_histogram(input, bins, window=None):
	input = np.array(input, dtype=np.int8)
	interval = 256//bins
	input = input.flatten()
	hist = [0]*bins
	if window!=None:
		plt.figure(window)
	plt.gray()
	for i in input:
		hist[i//interval] += 1
	plt.bar(np.arange(bins), np.array(hist, dtype=np.int64))

def show(name=None, test=None):
	t1 = time.time()

	input = None
	if test == None:
		pic = np.array(Image.open(name))
		input = np.array(pic)
		fig = plt.figure()
		plt.gray()
		plt.subplots_adjust(hspace=.5)
	else:
		input = test
	
	fl = []

	fl.append((input, 'Unaltered'))
	# fl.append((my_box(input, 3), 'Box'))
	# fl.append((my_median(input, 3), 'Median'))
	# fl.append((median_filter(input, 3), 'Median v2'))
	# fl.append((my_gaussian(input, 3), 'Gaussian'))
	# fl.append((my_fastgaussian(input, 3), 'Fast Gaussian'))
	# fl.append((my_gradient(input, 'forward', 'x'), 'Gradient X'))
	# fl.append((my_gradient(input, 'forward', 'y'), 'Gradient Y'))
	fl.append((my_canny(input, 3), 'Edgeeeesss'))
	# fl.append((my_sobel(input), 'Sobel'))

	if test == None:
		count = len(fl) + (0 if len(fl) % 2 == 0 else 1)
		for i in range(1, len(fl)+1):
			temp = fig.add_subplot(count//2, 2, i)
			temp.set_title(fl[i-1][1])
			plt.imshow(fl[i-1][0])
		print(f'{(time.time() - t1):0.2f} s')
		plt.show()
	else:
		for i in range(len(fl)):
			print(fl[i][0])
			pm(fl[i][0])
		print(f'{(time.time() - t1):0.2f} s')


def main():
	
	input = [
		[0,0,0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0,0,0],
		[0,0,0,90,90,90,90,90,0,0],
		[0,0,0,90,90,90,90,90,0,0],
		[0,0,0,90,90,90,90,90,0,0],
		[0,0,0,90,0,90,90,90,0,0],
		[0,0,0,90,90,90,90,90,0,0],
		[0,0,0,0,0,0,0,0,0,0],
		[0,0,90,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0,0,0]
	]

	input1 = [
		[10,10,20,20,20],
		[10,10,20,20,20],
		[10,10,20,20,20],
		[10,10,20,20,20],
		[10,10,20,20,20]
	]

	
	# my_histogram('image1.png', 64)
	# show(test=input)
	# show(name='image1.png')
	show(name='red_square_blue_B.png')
	# my_canny(None, 3)

if __name__ == '__main__':
	main()

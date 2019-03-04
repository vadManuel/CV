import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import hog
from sklearn.preprocessing import normalize
from skimage import exposure
from skimage.color import rgb2gray


def do_hog(image):
    # image = rgb2gray(image)
    image = normalize(image)
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=False)
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 1))
    # thresh = cv2.threshold(hog_image, 0.001, 255, cv2.THRESH_BINARY)[1]
    # thresh = cv2.threshold(hog_image, 120, 155, cv2.THRESH_BINARY)[1]
    # return thresh
    return hog_image


def auto_canny(image, sigma=(1/3)):
    med = np.median(image)
    low = int(max(0, (1-sigma) * med))
    high = int(min(255, (1+sigma) * med))
    can = cv2.Canny(cv2.GaussianBlur(image, (3, 3), sigma), low, high)
    return can


def midpoint(first, last):
    x=(first[0]+last[0])//2
    y=(first[1]+last[1])//2
    return (y, x)


def crop(canny, margin):
    val=np.argwhere(canny > 0)
    if not val.any():
        return None
    mid=midpoint(val[0], val[-1])
    top_left=(mid[0]-margin, mid[1]-margin)
    bottom_right=(mid[0]+margin, mid[1]+margin)
    return canny[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]


def args_crop(canny, margin):
    val=np.argwhere(canny > 0)
    if not val.any():
        return None
    mid=midpoint(val[0], val[-1])
    top_left=(mid[0]-margin, mid[1]-margin)
    bottom_right=(mid[0]+margin, mid[1]+margin)
    return [top_left, bottom_right]


def get_canny(image, sigma, src=True):
    im=cv2.imread(image)
    im=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im=cv2.GaussianBlur(im, (5, 5), sigma)
    canny=auto_canny(im, sigma)
    return canny


def get_cropped_canny(image, sigma=3, margin=25):
    im=cv2.imread(image)
    im=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    canny=get_canny(image, sigma)
    ind=args_crop(canny, margin)
    if ind is None and sigma != 0:
        return get_cropped_canny(image, sigma=sigma-.01, margin=margin)
    if ind is None:
        return im
    return im[ind[0][1]:ind[1][1], ind[0][0]:ind[1][0]]

from scipy import ndimage
# images = ['image04.jpg', 'image05.jpg',
        #   'image06.jpg', 'image07.jpg',]
#   'image08.jpg', 'image09.jpg']
image='Cross.png'
cropped_canny=get_cropped_canny(image, margin=86)
plt.subplot(1, 3, 1)
plt.imshow(cv2.imread(image))
plt.subplot(1, 3, 2)
rotated_img = ndimage.rotate(cropped_canny, 180)
plt.imshow(auto_canny(rotated_img, 1.3))
plt.subplot(1,3,3)
plt.imshow(do_hog(rotated_img))
# plt.imshow(auto_canny(cropped_canny, 1.3))
plt.show()

# cannys = [get_cropped_canny(i) for i in images]

# plt.subplots_adjust(hspace=.3, wspace=0)
# for i in range(len(cannys)):
#     plt.subplot(2, 2, i+1)
#     plt.imshow(auto_canny(cannys[i], 1))
#     # plt.imshow(do_hog(cannys[i]))
#     plt.title(images[i])
# plt.show()


# image = cv2.imread('image07.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.GaussianBlur(image, (5,5), 3)
# canny = auto_canny(image, 3)

# val = np.argwhere(canny > 0)
# mid = midpoint(val[0], val[-1])

# margin = 25
# top_left = (mid[0]-margin, mid[1]-margin)
# bottom_right = (mid[0]+margin, mid[1]+margin)
# cv2.rectangle(canny, top_left, bottom_right, (255, 0, 0), 1)

# plt.subplot(2,2,1)
# plt.imshow(image)
# plt.subplot(2,2,2)
# plt.imshow(canny)
# plt.subplot(2,2,3)
# plt.imshow(canny[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]])
# plt.show()

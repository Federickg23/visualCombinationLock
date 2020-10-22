"""
Created on Tue Feb 11 23:26:32 2020
@author: Federick Gonzalez
@uni: fag2113 
"""
import numpy as np
import math
import cv2
import argparse
import imutils
# Imports that follow are those I thought would be useful, but they turned out not to be.
import scipy.misc as misc
import imageio
import matplotlib.pyplot as plt
from skimage import exposure

cv2.__version__
'4.0.0-dev'

'''
This method takes 3 predetermined images as models which all other hand images will be compared against. Out of these 
models, the the bits that do not meet the threshold of set color are set to black, the rest to grey. From the remaining
image, a contour is formed, which gives the rough shape of the 3 chosen images. 
'''


def get_model_contours():
    model_contours = []
    location = './images/hand'
    ending = '.JPG'
    numbers = [20, 30, 24]
    for item in numbers:
        img1 = cv2.imread(location + str(item) + ending, 1)
        img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
        _, contours, hierarchy = cv2.findContours(thresh, 2, 1)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
        model_contours.append(contours[0])

    return model_contours


'''
This method takes in an image img, turns it gray, and from the grayed image a contour is made, giving a rough shape of 
the figure within. It returns the largest discovered contour. 
Much of the contour code is taken from the OpenCV Tutorial for Contour Features. 
'''


def get_contours(img):
    # contours = []
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
    _, contours, hierarchy = cv2.findContours(thresh, 2, 1)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    # modelContours.append(contours[0])
    return contours


'''
This method draws contours as a line to an image in order for them to be easily viewable by a user. 
'''


def draw_contours(img, contours):
    for item in contours:
        cnt = item
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)
        # print(defects)
        if type(defects) != type(None):
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                cv2.line(img, start, end, [0, 255, 0], 5)

    draw_lines(img)
    return img


'''
This method takes in a contour and calculates the center of the contour, also defined as a moment.
This code was taken from the Contour Features tutorial in OpenCV.  
'''


def find_moment(contour):
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    print("cx: ", cx)
    print("cy: ", cy)
    # cv2.circle(img, (cx, cy), 20, [255, 0, 0], 5)
    return cx, cy


'''
The distance formula, in method form. 
'''


def get_dist(x, x1, y, y1):
    return math.sqrt(((x - x1) ** 2) + ((y - y1) ** 2))


'''
This method divides an image into 9 sections (which are defined as sectors in this program) and based on the x and y 
coordinate of an object, returns which sector it is located in. It also returns the distance to the center of said 
sector, and the maximum possible distance from the center of each sector.
'''


def return_sector(img, x, y):
    max_dist = get_dist(0, img.shape[1] / 6, 0, img.shape[0] / 6)

    if x <= img.shape[1] / 3:
        if y <= img.shape[0] / 3:
            return 0, get_dist(x, img.shape[1] / 6, y, img.shape[0] / 6), max_dist
        elif y <= img.shape[0] / 3 * 2:
            return 3, get_dist(x, img.shape[1] / 6, y, img.shape[0] / 2), max_dist
        else:
            return 6, get_dist(x, img.shape[1] / 6, y, 5 * img.shape[0] / 6), max_dist
    elif x <= img.shape[1] / 3 * 2:
        if y <= img.shape[0] / 3:
            return 1, get_dist(x, img.shape[1] / 2, y, img.shape[0] / 6), max_dist
        elif y <= img.shape[0] / 3 * 2:
            return 4, get_dist(x, img.shape[1] / 2, y, img.shape[0] / 2), max_dist
        else:
            return 7, get_dist(x, img.shape[1] / 2, y, 5 * img.shape[0] / 6), max_dist
    else:
        if y <= img.shape[0] / 3:
            return 2, get_dist(x, 5 * img.shape[1] / 6, y, img.shape[0] / 6), max_dist
        elif y <= img.shape[0] / 3 * 2:
            return 5, get_dist(x, 5 * img.shape[1] / 6, y, img.shape[0] / 2), max_dist
        else:
            return 8, get_dist(x, 5 * img.shape[1] / 6, y, 5 * img.shape[0] / 6), max_dist
    pass


'''
This function draws 4 gridlines onto an image, dividing the image into 9 distinct sectors. 
'''


def draw_lines(img):
    img = cv2.line(img, (int(img.shape[1] / 3), 0), (int(img.shape[1] / 3), img.shape[0]), (255, 0, 0), 5, 1)
    img = cv2.line(img, (int(2 * img.shape[1] / 3), 0), (int(2 * img.shape[1] / 3), img.shape[0]), (255, 0, 0), 5, 1)
    img = cv2.line(img, (0, int(img.shape[0] / 3)), (img.shape[1], int(img.shape[0] / 3)), (255, 0, 0), 5, 1)
    img = cv2.line(img, (0, int(2 * img.shape[0] / 3)), (img.shape[1], int(2 * img.shape[0] / 3)), (255, 0, 0), 5, 1)

    return img


'''
This was a proof of concept, intended to determine whether or not simply using a threshold would discover all necessary
contours. It was determined that this would not work in all cases. Tis method can be considered deprecated. 
Much of the contour code is taken from the OpenCV Tutorial for Contour Features. 
'''


def analyze_hands_test():
    models = get_model_contours()
    location = './images/hand'
    ending = '.JPG'
    images = []
    for i in range(20, 26):
        images.append(location + str(i) + ending)

    for path in images:
        print(path)
        img = cv2.imread(path, 1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
        _, contours, hierarchy = cv2.findContours(thresh, 2, 1)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
        cx = 0
        cy = 0
        for item in contours:
            cnt = item
            hull = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull)
            # print(defects)

            if type(defects) != type(None):
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    cv2.line(img, start, end, [0, 255, 0], 5)
                    cv2.circle(img, far, 5, [0, 0, 255], -1)
                M = cv2.moments(item)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # print("cx: ", cx)
                # print("cy: ", cy)
                cv2.circle(img, (cx, cy), 20, [255, 0, 0], 5)
        img = draw_lines(img)
        # print(img.shape)
        location_info = return_sector(img, cx, cy)
        print('quadrant: ', location_info[0])
        print('distance to center: ', location_info[1])
        print('max distance to center', location_info[2])
        similarity = cv2.matchShapes(item, models[0], 3, 0.0)
        print('similarity to hand1: ', similarity)
        img = cv2.resize(img, None, fx=.15, fy=.15, interpolation=cv2.INTER_AREA)
        cv2.imshow('img', img)
        k = cv2.waitKey(0)
        if k == 27:
            break
        cv2.destroyAllWindows()


'''
This image takes a YCrCb lower bound and upper bound in order to try and determine what in the image is skin colored. It
then takes the image, and creates a mask out of the skin colored pixels, then smoothes out and blurs this mask, and then
takes the original image that fits under this mask and returns it over a blank black background.  
Most of this was taken from Adrian Rosebrock's tutorial titled Skin Detection: A Step-by-Step Example using Python and 
OpenCV. 
'''


def get_skin_mask(img):
    # lower = np.array([0, 48, 80], dtype="uint8")
    # upper = np.array([255, 255, 255], dtype="uint8")

    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([235, 173, 127], np.uint8)

    # img = imutils.resize(img, width=900)
    # converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # skinMask = cv2.inRange(converted, lower, upper)
    imageYCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    skinMask = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)
    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(img, img, mask=skinMask)
    return skin


'''
This was a test to see whether or not video detection would work. It was difficult to implement because it was hard to 
determine differences between hand and face. Due to time constraints, I opted for uploaded pictures instead.  
'''


def vid_skin_detect():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    args = vars(ap.parse_args())
    # define the upper and lower boundaries of the HSV pixel
    # intensities to be considered 'skin'
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    # if a video path was not supplied, grab the reference
    # to the gray
    if not args.get("video", False):
        camera = cv2.VideoCapture(0)
    # otherwise, load the video
    else:
        camera = cv2.VideoCapture(args["video"])
    # keep looping over the frames in the video
    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()
        # if we are viewing a video and we did not grab a
        # frame, then we have reached the end of the video
        if args.get("video") and not grabbed:
            break
        # resize the frame, convert it to the HSV color space,
        # and determine the HSV pixel intensities that fall into
        # the speicifed upper and lower boundaries
        # print(type(frame))
        frame = imutils.resize(frame, width=900)
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(converted, lower, upper)
        # apply a series of erosions and dilations to the mask
        # using an elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.erode(skinMask, kernel, iterations=2)
        skinMask = cv2.dilate(skinMask, kernel, iterations=2)
        # blur the mask to help remove noise, then apply the
        # mask to the frame
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        skin = cv2.bitwise_and(frame, frame, mask=skinMask)
        # show the skin in the image along with the mask

        contours = get_contours(skin)
        skin = draw_contours(skin, contours)
        skin = draw_lines(skin)

        # cv2.imshow("images", np.hstack([frame, skin]))
        cv2.imshow("skin", skin)
        # if the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()


'''
This was a proof of concept to determine if get_skin_mask would actually assist in determining whether or not skin was
visibily in the image, and taking that skin. Since it worked, the project continued. 
'''


def analysis_by_color_test():
    # print(type(cv2.imread('./images/hand1.CR2', 1)))
    # vidSkinDetect()
    location = './images/hand'
    ending = '.JPG'
    model = get_model_contours()
    images = []
    for i in range(20, 49):
        if i != 43 and i!= 44:
             images.append(location + str(i) + ending)

    for item in images:
        print(item)
        img = cv2.imread(item, 1)
        skin = get_skin_mask(img)
        # skin = cv2.bitwise_and(img, img, mask=skinMask)
        contours = get_contours(skin)
        skin = draw_contours(skin, contours)

        # skin = imutils.resize(skin, 600)
        moment = find_moment(contours[0])
        skin = cv2.circle(skin, (moment[0], moment[1]), 20, [255, 0, 0], 30)
        locationInfo = return_sector(skin, moment[0], moment[1])
        print(locationInfo)
        similarity = []
        for item in model:
            similarity.append(cv2.matchShapes(contours[0], item, 3, 0.0))
        print('similarity values: ', similarity)
        skin = imutils.resize(skin, height=800)
        cv2.imshow("skin", skin)
        k = cv2.waitKey(0)

        if k == 27:
            break
        cv2.destroyAllWindows()


'''
For each of the 12 test cases, test_images is called, and the results are compiled into an array and displayed. 
'''


def main():
    analysis_by_color_test()
    validity = []

    for i in range(1, 13):
        validity.append(str(i) + " " + str(test_images('./images/' + str(i) + '/')))
    for item in validity:
        print(item)


'''
This method tests 3 images in a folder determined by root. It compares them to the predetermined sequence of model images
and sectors. If both the quandrant and model match to a certain degree, a variable valid remains true. Else, it is changed
to false.
The shape matching was taken from Learn OpenCV's article Shape Matching using Hu Moments (C++/Python) 
'''


def test_images(root):
    valid = True
    end = ['hand1.jpg', 'hand2.jpg', 'hand3.jpg']
    pos = ['splayed', 'chop', 'fist']
    paths = []
    sectors = [4, 6, 8]
    for item in end:
        paths.append(root + item)
    models = get_model_contours()
    i = 0
    for item in paths:
        print(item)
        img = cv2.imread(item, 1)
        skin = get_skin_mask(img)
        # skin = cv2.bitwise_and(img, img, mask=skinMask)
        contours = get_contours(skin)
        skin = draw_contours(skin, contours)
        # skin = imutils.resize(skin, 600)
        moment = find_moment(contours[0])
        skin = cv2.circle(skin, (moment[0], moment[1]), 20, [255, 0, 0], 30)
        locationInfo = return_sector(skin, moment[0], moment[1])
        print(locationInfo)
        similarity = cv2.matchShapes(contours[0], models[i], 3, 0.0)
        print('similarity: ', similarity)
        print('sector: ', locationInfo[0])
        if sectors[i] != locationInfo[0] or similarity > 1:
            index = 0
            printed = False
            for model in models:
                similarity = cv2.matchShapes(contours[0], model, 3, 0.0)
                if similarity < 1 and not printed:
                    print(pos[index], ', ', locationInfo[0])
                index +=1
            if not printed:
                print('unknown, ', locationInfo[0])
            valid = False
        else:
            if locationInfo[1] > 2*locationInfo[2]/3:
                valid = False
                print('Too far from sector center')
            print(pos[i], ', ', locationInfo[0])
        # skin = imutils.resize(skin, height=800)
        # cv2.imshow("skin", skin)
        # k = cv2.waitKey(0)
        # if k == 27:
        #    break
        # cv2.destroyAllWindows()
        i += 1
    return valid


if __name__ == '__main__':
    main()

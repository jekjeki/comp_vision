import enum
import cv2 as cv  
from matplotlib import pyplot as plt 
import numpy as np  
import os
import math

from scipy.misc import face  

def menu():
    print('welcome to photo app')
    print('1. image processing')
    print('2. edge detection')
    print('3. shape detection')
    print('4. pattern detection')
    print('5. detect face')
    print('choose: ')

menu()
x=int(input())

if (x==1): 
    image = cv.imread('male.jpg')
    height, width = image.shape[:2]

    def menuImageProcessing():
        print('1. gray image processing')
        print('2. thresh image processing')

    menuImageProcessing()
    input = int(input())

    if input==1:

        def showResult(nrow=0, ncol=0, res_stack=None):
            plt.figure(figsize=(12, 12))
            for idx, (lbl, img) in enumerate(res_stack):
                plt.subplot(nrow, ncol, idx + 1)
                plt.imshow(img, cmap='gray')
                plt.title(lbl)
                plt.axis('off')
            plt.show()

        gray_ocv = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray_avg = np.dot(image, [0.33, 0.33, 0.33])

        b, g, r = image[ : , : , 0], image[ : , : , 1], image[ : , : , 2]
        max_cha = max(np.amax(b), np.amax(g), np.amax(r))
        min_cha = min(np.amin(b), np.amin(g), np.amin(r))
        gray_lig = np.dot(image, [(max_cha + min_cha)/2, (max_cha + min_cha)/2, (max_cha + min_cha)/2])

        gray_lum = np.dot(image, [0.07, 0.71, 0.21])
        gray_wag = np.dot(image, [0.114, 0.587, 0.299])

        gray_labels = ['gray opencv', 'gray average', 'gray lightness', 'gray luminosity', 'gray weighted average']
        gray_images = [gray_ocv, gray_avg, gray_lig, gray_lum, gray_wag]
        showResult(3, 2, zip(gray_labels, gray_images))

    elif input==2:

        def showResult(nrow=0, ncol=0, res_stack=None):
                plt.figure(figsize=(12, 12))
                for idx, (lbl, img) in enumerate(res_stack):
                    plt.subplot(nrow, ncol, idx + 1)
                    plt.imshow(img, cmap='gray')
                    plt.title(lbl)
                    plt.axis('off')
                plt.show()
        
        gray_ocv = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray_avg = np.dot(image, [0.33, 0.33, 0.33])

        #thresh   
        thresh = 100
        thresh_image = gray_ocv.copy()

        for i in range(height):
            for j in range(width):
                if thresh_image[i, j] > thresh:
                    thresh_image[i, j] = 255
                else:
                    thresh_image[i, j] = 0

        _, bin_thresh = cv.threshold(gray_ocv, 100, 255, cv.THRESH_BINARY)
        _, inv_bin_thresh = cv.threshold(gray_ocv, 100, 255, cv.THRESH_BINARY_INV)
        _, trunc_thresh = cv.threshold(gray_ocv, 100, 255, cv.THRESH_TRUNC)
        _, tozero_thresh = cv.threshold(gray_ocv, 100, 255, cv.THRESH_TOZERO)
        _, inv_tozero_thresh = cv.threshold(gray_ocv, 100, 255, cv.THRESH_TOZERO_INV)
        _, otsu_thresh = cv.threshold(gray_ocv, 100, 255, cv.THRESH_OTSU)

        thresh_labels = ['manual thresh', 'binary thresh', 'inverse binary thresh', 'trunc thresh', 'tozero thresh', 'inverse tozero thresh', 'otsu thresh']
        thresh_images = [thresh_image, bin_thresh, inv_bin_thresh, trunc_thresh, tozero_thresh, inv_tozero_thresh, otsu_thresh]
        showResult(3, 3, zip(thresh_labels, thresh_images))

        def manual_mean_filter(source, ksize):
            np_source = np.array(source) # np.array vs np.asarray | mutable vs immutable
            for i in range(height - ksize - 1):
                for j in range(width - ksize - 1):
                    matrix = np.array(np_source[i : (i + ksize), j : (j + ksize)]).flatten()
                    mean = np.mean(matrix)
                    np_source[i + ksize // 2, j + ksize // 2] = mean
            return np_source

        def manual_median_filter(source, ksize):
            np_source = np.array(source) # np.array vs np.asarray | mutable vs immutable
            for i in range(height - ksize - 1):
                for j in range(width - ksize - 1):
                    matrix = np.array(np_source[i : (i + ksize), j : (j + ksize)]).flatten()
                    median = np.median(matrix)
                    np_source[i + ksize // 2, j + ksize // 2] = median
            return np_source

        b, g, r = cv.split(image)
        ksize = 5

        b_mean = manual_mean_filter(b, ksize)
        g_mean = manual_mean_filter(g, ksize)
        r_mean = manual_mean_filter(r, ksize)

        b_median = manual_median_filter(b, ksize)
        g_median = manual_median_filter(g, ksize)
        r_median = manual_median_filter(r, ksize)

        mmean_filter = cv.merge( (r_mean, b_mean, g_mean) )
        mmedian_filter = cv.merge( (r_median, b_median, g_median) )

        filter_image = gray_ocv.copy()

        mean_blur = cv.blur(filter_image, (5, 5))
        median_blur = cv.medianBlur(filter_image, 5)
        gaussian_blur = cv.GaussianBlur(filter_image, (5, 5), 2.0)
        bilateral_blur = cv.bilateralFilter(filter_image, 5, 150, 150)

        filter_labels = ['manual mean filter', 'manual median blur', 'mean filter', 'median filter', 'gaussian filter', 'bilateral filter']
        filter_images = [mmean_filter, mmedian_filter, mean_blur, median_blur, gaussian_blur, bilateral_blur]
        showResult(3, 2, zip(filter_labels, filter_images))

# elif x==2:
#     print('test')

elif (x==2):
    #menu 2

    image = cv.imread('fruit.jpg')
    igray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    height, width = image.shape[:2]

    def showResult(nrow=0, ncol=0, res_stack=None):
        plt.figure(figsize=(12, 12))
        for idx, (lbl, img) in enumerate(res_stack):
            plt.subplot(nrow, ncol, idx + 1)
            plt.imshow(img, cmap='gray')
            plt.title(lbl)
            plt.axis('off')
        plt.show()

    laplace_uintu08 = cv.Laplacian(igray, cv.CV_8U)
    laplace_uintu16 = cv.Laplacian(igray, cv.CV_16S)
    laplace_uintu32 = cv.Laplacian(igray, cv.CV_32F)
    laplace_uintu64 = cv.Laplacian(igray, cv.CV_64F)

    laplace_labels = ['laplace 8-bit', 'laplace 16-bit', 'laplace 32-bit', 'laplace 64-bit']
    laplace_images = [laplace_uintu08, laplace_uintu16, laplace_uintu32, laplace_uintu64]
    showResult(2, 2, zip(laplace_labels, laplace_images))

    def calculateSobel(source, kernel, ksize):
        res_matrix = np.array(source)
        for i in range(height - ksize - 1):
            for j in range(width - ksize - 1):
                patch = source[i : (i + ksize) , j : (j + ksize)].flatten()
                result = np.convolve(patch, kernel, 'valid')
                res_matrix[i + ksize//2, j + ksize//2] = result[0]
        return res_matrix

    kernel_x = np.array([
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    ])

    kernel_y = np.array([
        -1, -2, -1,
        0,  0,  0,
        1,  2,  1
    ])
    ksize = 3
    manual_sobel_x = igray.copy()
    manual_sobel_y = igray.copy()

    manual_sobel_x = calculateSobel(manual_sobel_x, kernel_x, ksize)
    manual_sobel_y = calculateSobel(manual_sobel_y, kernel_y, ksize)

    sobel_x = cv.Sobel(igray, cv.CV_32F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(igray, cv.CV_32F, 0, 1, ksize=3)

    sobel_labels = ['manual sobel-x', 'manual sobel-y', 'sobel-x', 'sobel-y']
    sobel_images = [manual_sobel_x, manual_sobel_y, sobel_x, sobel_y]
    showResult(2, 2, zip(sobel_labels, sobel_images))

    merged_sobel = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    merged_sobel *= 255.0 / merged_sobel.max()

    manual_merged_sobel = cv.bitwise_or(manual_sobel_x, manual_sobel_y)
    manual_merged_sobel = np.uint16(np.absolute(manual_merged_sobel))

    merged_sobel_labels = ['merged sobel', 'manual merge sobel']
    merged_sobel_images = [merged_sobel, manual_merged_sobel]
    showResult(1, 2, zip(merged_sobel_labels, merged_sobel_images))

    canny_050100 = cv.Canny(igray, 50, 100)
    canny_050150 = cv.Canny(igray, 50, 150)
    canny_075150 = cv.Canny(igray, 75, 150)
    canny_075225 = cv.Canny(igray, 75, 225)

    canny_labels = ['canny 50 100', 'canny 50 150', 'canny 75 150', 'canny 75 225']
    canny_images = [canny_050100, canny_050150, canny_075150, canny_075225]
    showResult(2, 2, zip(canny_labels, canny_images))

elif (x==3):
    #menu 3  


    image = cv.imread('chessboard.jpg')
    igray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    igray = np.float32(igray)

    def showResult(source, cmap=None):
        plt.imshow(source, cmap=cmap)
        plt.show()

    harris_corner = cv.cornerHarris(igray, 2, 5, 0.04)
    showResult(harris_corner, 'gray')

    without_subpix = image.copy()
    without_subpix[harris_corner > 0.01 * harris_corner.max()] = [0, 0, 255]
    showResult(without_subpix, 'gray')

    _, thresh = cv.threshold(harris_corner, 0.01 * harris_corner.max(), 255, 0)
    thresh = np.uint8(thresh)
    _, _, _, centroids = cv.connectedComponentsWithStats(thresh)
    centroids = np.float32(centroids)

    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 0.0001)
    enhanced_corner = cv.cornerSubPix(igray, centroids, (2, 2), (-1, -1), criteria)

    with_subpix = image.copy()
    enhanced_corner = np.uint16(enhanced_corner)
    for corner in enhanced_corner:
        x, y = corner[:2]
        with_subpix[y, x] = [0, 255, 0]
    showResult(with_subpix)

elif (x==4):
    classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    #train

    train_path = 'train'
    tdir = os.listdir(train_path)

    face_list = []
    class_list = []

    for index, train_dir in enumerate(tdir):
        for image_path in os.listdir(f'{train_path}'):
            path = f'{train_path}/{image_path}'
            if path.split('.')[1] != 'db':
                gray = cv.imread(path, 0)
                faces = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2)
                if len(faces) < 1:
                    continue
                for face_rect in faces:
                    x, y, w, h = face_rect 
                    face_image = gray[y: y+w, x : x+h]
                    face_list.append(face_image)
                    class_list.append(index)

    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(face_list, np.array(class_list))

    # TEST
    test_path = 'test'
    for path in os.listdir(test_path):
        full_path = f'{test_path}/{path}'
        image = cv.imread(full_path)
        igray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        faces = classifier.detectMultiScale(igray, scaleFactor=1.2, minNeighbors=5)
        if len(faces) < 1:
            continue
        for face_rect in faces:
            x, y, w, h = face_rect
            face_image = gray[y: y + w, x : x + h]
            res, conf = face_recognizer.predict(face_image)
            conf = math.floor(conf * 100) / 100
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            text = f'{tdir[res]} {str(conf)}%'
            cv.putText(image, text, (x, y - 10), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)
            cv.imshow('result', image)
            cv.waitKey(0)
    cv.destroyAllWindows()

elif (x==5):
    image = cv.imread('faces.jpg')
    face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.0485258, 6)

    if faces is ():
        print('no faces found')
    
    for (x,y,w,h) in faces:
        cv.rectangle(image, (x,y), (x+w, y+h), (127, 0, 255), 2)
        cv.imshow('face detect', image)
        cv.waitKey(0)

    cv.destroyAllWindows()

    def showResult(nrow=0, ncol=0, res_stack=None):
            plt.figure(figsize=(12, 12))
            for idx, (lbl, img) in enumerate(res_stack):
                plt.subplot(nrow, ncol, idx + 1)
                plt.imshow(img, cmap='gray')
                plt.title(lbl)
                plt.axis('off')
            plt.show()   

    gray_ocv = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow('blur image',gray_ocv)
    cv.waitKey(0)
    cv.destroyAllWindows()

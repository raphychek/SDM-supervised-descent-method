import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio
import glob, os
import argparse

from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge


def get_bboxes(bbox_path):
    """
    input:
        bbox_path: the path to the bbox .mat files
    output:
        ret: an easily usable dict
    """
    bboxes = sio.loadmat(bbox_path)['bounding_boxes'] 
    ret = {}
    for bb in bboxes[0]:
        ret[bb[0][0][0][0]] = list(bb[0][0][1][0])
    return ret


def load_lm(file_path):
    """
    input:
        file_path: the path to the landmarks .mat files
    output:
        lms: an array containing landmarks, in the same order than images
    """
    with open(file_path) as f: 
        rows = [rows.strip() for rows in f]
    coords_set = [point.split() for point in rows[rows.index('{') + 1:rows.index('}')]]
    lms = np.array([list([round(float(point),2) for point in coords]) for coords in coords_set])
    return lms


def big_bbox(img, bbox, ratio = 10):
    """
    Not used in the vanilla code from the article, but might give interesting results

    input:
        img: the input image
        bbox: the given bbox
    output:
        [x1, y1, x2, y2]: enlarged by 30% bbox, but cannot go outside the image
    """
    h, w = img.shape[:2]
    [x1, y1, x2, y2] = bbox
    l = abs(x2 - x1) * ratio/100
    x1, x2 = max(x1 - l, 0), min(x2 + l, w - 1)
    l = abs(y2 - y1) * ratio/100
    y1, y2 = max(y1 - l, 0), min(y2 + l, h - 1)
    return [x1, y1, x2, y2]


def load_images_lms(images_path, lms_path, dim = 400):
    """
    input:
        images: the path of images
        lms: the path of landmarks
        dim: the size in which to resize the cropped image
        size: the size of the training set - cannot be too high or my computer will explode
    output:
        images_list: array containing all cropped, resized B&W images
        lms_list: array containing all resized and displaced landmarks
        lm/len(images): the average landmark position
    """
    images = sorted(glob.glob(images_path))
    lms = sorted(glob.glob(lms_path))

    images_list = np.zeros((len(images), 400, 400), dtype = np.uint8)
    lms_list = np.zeros((len(images), 68, 2))
    lms_tot = np.zeros((68,2))
    for i in range(len(images)):
        image = cv2.imread(images[i])

        [x1, y1, x2, y2] = bboxes[images[i].split("/")[-1]]
        [x1, y1, x2, y2] = big_bbox(image, [x1, y1, x2, y2], ratio = 0) # the ratio being 0 here it sticks to the article 
        
        image_crop = image[int(y1) : int(y2), int(x1) : int(x2)]
        hh, ww = image_crop.shape[:2]
        image_crop = cv2.resize(image_crop, (dim, dim))
        
        image_copy = np.copy(image_crop)
        
        lm = load_lm(lms[i])
        lm = np.array(lm) - [x1, y1]
        lm = lm * [dim, dim] / [ww, hh]
        
        images_list[i] = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        lms_list[i] = lm

        lms_tot += lm
    return images_list, lms_list, lms_tot/len(images)


def compute_descriptor(image_gray, lms):
    """
    input:
        image_gray: the croped, resized, gray image
    output:
        descriptors: the HOG descriptor
    """
    to_tuple = lambda L : tuple(tuple(x) for x in L)
    winSize = (32, 32)
    blockSize = (8, 8)
    blockStride = (8, 8)
    cellSize = (2, 2)
    nbins = 4
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    descriptors = hog.compute(image_gray, locations = to_tuple(lms) )
    return descriptors


def matricial_hog(images, lms):
    """
    input:
        images: array of croped, resized, gray images
        lms: array of resized and displaced landamrks
    output: 
        hogsx: array containing hog descriptors for each image
    """
    hog0 = compute_descriptor(images[0], lms[0]).T
    hogsx = np.zeros((images.shape[0], len(hog0[0])))
    hogsx[0] = hog0
    for i in range(1, images.shape[0]):
        hogx = compute_descriptor(images[i], lms[i]).T
        hogsx[i] = hogx
    return hogsx


def train(epochs, images, lms):
    """
    input:
        epochs: the number of training epochs
        images: array of croped, resized, gray images
        lms: array of resized and displaced landmarks
    output:
        Rs: list containing R_i matrixes
        bs: list containing b_i vectors
        x0: the mean landmark on the whole dataset
    """
    x0 = np.mean(lms, axis = 0)
    xstar = lms
    x0s = np.array(x0.ravel().tolist() * len(lms)).reshape(lms.shape[0],68,2)
    Rs, bs = [], []

    print("epoch 0")
    print("Loss:", np.linalg.norm(xstar - x0))
    for i in range(epochs):
        print("epoch", i+1)
        delta = xstar - x0s
        hogx = matricial_hog(images, x0s)
        #solver = LinearRegression()
        #solver = Lasso(alpha = 0.001)
        solver = Ridge(alpha = 0.01)
        solver.fit(hogx, delta.reshape(lms.shape[0],136)) 
        Rs.append(solver.coef_.T)
        bs.append(solver.intercept_.T)
        x0s = (x0s.reshape(lms.shape[0],136) + np.matmul(hogx, Rs[-1]) + bs[-1]).reshape(lms.shape[0],68,2)
        print("Loss:", np.linalg.norm(xstar - x0s))


    return Rs, bs, x0


def infer_one_image(Rs, bs, x0, image):
    """
    input:
        Rs: list containing R_i matrixes
        bs: list containing b_i vectors
        x0: the mean landmark on the whole dataset
        images: array of croped, resized, gray images
    output:
        xc: predicted landmarks
    """
    xc = np.copy(x0)
    for i in range(len(Rs)):
        hogx = compute_descriptor(image, xc).T
        x_old = np.copy(xc)
        xc = (xc.reshape(1, 136) + np.matmul(hogx, Rs[i]) + bs[i]).reshape(68,2)
    return xc.reshape(68,2)


def test_folder(Rs, bs, x0, images, image_path, bboxes, output_dir, lms):
    """
    input:
        Rs: list containing R_i matrixes
        bs: list containing b_i vectors
        x0: the mean landmark on the whole dataset
        images: array of croped, resized, gray images
        image_path: path containing the original images
        bboxes: a dict containing bounding boxes around people's face
        output_dir: path were to save images with landmarks plot on them
    output:
        pred_lms: list of predicted landmarks
    """
    full_images = sorted(glob.glob(image_path))
    pred_lms, MAE = [], []
    for i in range(len(images)):
        image = np.copy(images[i])
        x_pred = infer_one_image(Rs, bs, x0, image)
        MAE.append(mean_absolute_error(x_pred, lms[i]))
        full_image = cv2.imread(full_images[i])
        hh, ww = full_image.shape[:2]
        [x1, y1, x2, y2] = bboxes[full_images[i].split("/")[-1]]
        [x1, y1, x2, y2] = big_bbox(full_image, [x1, y1, x2, y2], ratio = 0) # the ratio being 0 here it sticks to the article 
        x_pred = x_pred * [x2 - x1, y2 - y1] / (400, 400)
        x_pred = x_pred + [x1, y1]
        for xy in x_pred:
            x, y = int(xy[0]), int(xy[1])
            cv2.circle(full_image, (int(x), int(y)), 5, (0, 0, 255), max(full_image.shape[:2])//100 )
        cv2.imwrite(output_dir + full_images[i].split("/")[-1], full_image)
        pred_lms.append(x_pred)
    return pred_lms, sum(MAE)/len(MAE)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Arguments for SDM')
    parser.add_argument('--mode', type=str, required = True, help = 'test or train mode')

    parser.add_argument('--weights_path', type=str, default = "./weights.mat", help='Weights containing R, b and initial vector')

    parser.add_argument('--train_bbox', type=str, default = "./bboxes/bounding_boxes_helen_trainset.mat", help = 'path of .mat file containing training bbox')
    parser.add_argument('--train_images', type = str, default = "./helen/trainset/*.jpg", help = "paths of train image files")
    parser.add_argument('--train_lms', type = str, default = "./helen/trainset/*.pts", help = "paths of train landmarks files")

    parser.add_argument('--test_bbox', type=str, default = "./bboxes/bounding_boxes_helen_testset.mat", help = 'path of .mat file containing testing bbox')
    parser.add_argument('--test_images', type = str, default = "./helen/testset/*.jpg", help = "paths of test image files")
    parser.add_argument('--test_lms', type = str, default = "./helen/testset/*.pts", help = "paths of test landmarks files")

    parser.add_argument('--output_dir', type = str, default = "./predictions/", help = "path where files save")
    parser.add_argument('--train_epochs', type = int, default = 4)
    

    args = parser.parse_args()

    if args.mode == 'train':
        print("Loading bounding boxes...")
        bboxes = get_bboxes(args.train_bbox)
        print("Loading images and landmarks...")
        images, lms, x0 = load_images_lms(args.train_images, args.train_lms)
        print("Start training...")
        Rs, bs, _ = train(args.train_epochs, images, lms)
        print("Saving weights")
        sio.savemat('weights.mat',{'R': Rs,'b': bs,'x0': x0})

    elif args.mode == 'test':
        print("Loading bounding boxes...")
        bboxes = get_bboxes(args.test_bbox)
        print("Loading images and landmarks...")
        images, lms, _ = load_images_lms(args.test_images, args.test_lms)
        print("Loading weights")
        try:
            weights = sio.loadmat(args.weights_path)
        except:
            print("train the model or give a correct weight path")
        Rs, bs, x0 = weights['R'], weights['b'], weights['x0']
        print("Infering on test set...")
        _, error = test_folder(Rs, bs, x0, images, args.test_images, bboxes, args.output_dir, lms)
        print("MAE is", round(error, 1))


    else:
        print("mode argument should be test or train")
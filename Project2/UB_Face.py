'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import cv2
import numpy as np
import os
import sys
import math

import face_recognition

from typing import Dict, List
from utils import show_image
import requests
'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''
np.random.seed(13)
def detect_faces(img: np.ndarray) -> List[List[float]]:
    """
    Args:
        img : input image is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    detection_results: List[List[float]] = [] # Please make sure your output follows this data format.

    # Add your code here. Do not modify the return and input arguments.
    URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    response = requests.get(URL)
    with open('feed.xml', 'wb') as file:
        file.write(response.content)
    face_cascade = cv2.CascadeClassifier('feed.xml')
    detection_results = [list(ele) for ele in face_cascade.detectMultiScale(img, 1.1, 4)]
    detection_results = [list(map(float, sublist)) for sublist in detection_results]
    return detection_results


def cluster_faces(imgs: Dict[str, np.ndarray], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    cluster_results: List[List[str]] = [[]] * K # Please make sure your output follows this data format.

    # Add your code here. Do not modify the return and input arguments.
    face_encodings = {}
    for key, value in imgs.items():
        for i in detect_faces(value):
            face_encodings[key] = face_recognition.face_encodings(value)
    face_encodingsf = []
    for i, k in zip(face_encodings.values(), range(len(face_encodings))):
        for j in i:
            face_encodingsf.append(j)
            break
    cluster_idx, centers, loss = KMeans()(np.array(face_encodingsf), K)
    clusterresults = []
    for i in range(K):
        list = []
        for j, k in zip(cluster_idx, face_encodings):
            if i == j:
                list.append(k)
        clusterresults.append(list)
    cluster_results = clusterresults
    print(cluster_results)
    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# Your functions. (if needed)


class KMeans(object):

    def pairwise_dist(self, x, y):
        return np.sqrt(abs(np.sum(np.square(x), axis=1)[:, np.newaxis] + np.sum(np.square(y), axis=1) - 2 * np.dot(x, y.T)))

    def _init_centers(self, points, K):
        row, col = points.shape
        retArr = np.empty([K, col])
        for number in range(K):
            retArr[number] = points[np.random.randint(row)]
        return retArr

    def _update_assignment(self, centers, points):
        row, col = points.shape
        cluster_idx = np.empty([row])
        return np.argmin(self.pairwise_dist(points, centers), axis=1)

    def _update_centers(self, old_centers, cluster_idx, points):
        K, D = old_centers.shape
        new_centers = np.empty(old_centers.shape)
        for i in range(K):
            new_centers[i] = np.mean(points[cluster_idx == i], axis=0)
        return new_centers

    def _get_loss(self, centers, cluster_idx, points):
        loss = 0.0
        N, D = points.shape
        for i in range(N):
            loss = loss + np.square(self.pairwise_dist(points, centers)[i][cluster_idx[i]])
        return loss

    def __call__(self, points, K, max_iters=500, abs_tol=1e-16, rel_tol=1e-16):
        centers = self._init_centers(points, K)
        for it in range(max_iters):
            cluster_idx = self._update_assignment(centers, points)
            centers = self._update_centers(centers, cluster_idx, points)
            loss = self._get_loss(centers, cluster_idx, points)
            K = centers.shape[0]
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
        return cluster_idx, centers, loss
import sys
import cv2
import numpy as np
from sklearn.cluster import KMeans


class DenseDetector(object):
    def __init__(self, step_size=20, feature_scale=20, img_bound=20):
        # Create a dense feature detector
        self.initXyStep = step_size
        self.initFeatureScale = feature_scale
        self.initImgBound = img_bound

    def detect(self, img):
        keypoints = []
        rows, cols = img.shape[:2]
        for x in range(self.initImgBound, rows, self.initFeatureScale):
            for y in range(self.initImgBound, cols, self.initFeatureScale):
                keypoints.append(cv2.KeyPoint(float(x), float(y), self.initXyStep))
        return keypoints


class SIFTDetector(object):
    def __init__(self):
        self.detector = cv2.xfeatures2d.SIFT_create()

    def detect(self, img):
        # Convert to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect keypoints using SIFT
        return self.detector.detect(gray_image, None)


class SIFTExtractor(object):
    def __init__(self):
        self.extractor = cv2.xfeatures2d.SIFT_create()

    def compute(self, image, kps):
        if image is None:
            print("Not a valid image")
            raise TypeError

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kps, des = self.extractor.detectAndCompute(gray_image, None)
        return kps, des


class Quantizer(object):
    def __init__(self, num_cluster=32):
        self.num_dims = 128
        self.extractor = SIFTExtractor()
        self.num_clusters = num_cluster
        self.num_retries = 10

    def quantize(self, datapoints):
        # Create KMeans object
        kmeans = KMeans(self.num_clusters,
                        n_init=max(self.num_retries, 1),
                        max_iter=10, tol=1.0)
        # Run KMeans on the datapoints
        res = kmeans.fit(datapoints)

        # Extract the centroids of those clusters
        centroids = res.cluster_centers_

        return kmeans, centroids

    def normalize(self,input_data):
        sum_input = np.sum(input_data)
        if sum_input > 0:
            return input_data / sum_input
        else:
            return input_data

    # Extract feature vector from the image
    def get_feature_vector(self, img, kmeans, centroids):
        kps = DenseDetector().detect(img)
        kps, fvs = self.extractor.compute(img, kps)
        labels = kmeans.predict(fvs)
        fv = np.zeros(self.num_clusters)

        for i, item in enumerate(fvs):
            fv[labels[i]] += 1

        fv_image = np.reshape(fv, ((1, fv.shape[0])))
        return self.normalize(fv_image)

class FeatureExtractor(object):
    def extract_image_features(self, img):
        # Dense feature detector
        kps = DenseDetector().detect(img)

        # SIFT feature extractor
        kps, fvs = SIFTExtractor().compute(img, kps)

        return fvs

    # Extract the centroids from the feature points
    def get_centroids(self, input_map, num_samples_to_fit=10):
        kps_all = []

        count = 0
        cur_lable = ""
        for item in input_map:
            if count >= num_samples_to_fit:
                if cur_lable != item['label']:
                    count = 0
                else:
                    continue
            count += 1

            if count == num_samples_to_fit:
                print("Built centroids for", item['label'])

            cur_lable = item['label']
            img = cv2.imread(item['image'])
            img = resize_to_size(img, 150)

            num_dims = 128
            fvs = self.extract_image_features(img)
            kps_all.extend(fvs)

        kmeans, centroids = Quantizer().quantize(kps_all)
        return kmeans, centroids

    def get_feature_vector(self, img, kmeans, centroids):
        return Quantizer().get_feature_vector(img, kmeans, centroids)

if __name__ == '__main__':
    input_image = cv2.imread('cpu_01.jpg')
    input_image_dense = np.copy(input_image)
    input_image_sift = np.copy(input_image)


    keypoints = DenseDetector(20, 40, 1).detect(input_image)
    # Draw keypoints o top of the input image
    input_image_dense = cv2.drawKeypoints(input_image_dense, keypoints, None,
                                          flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    # Display the output image
    cv2.imshow('Dense feature detector', input_image_dense)

    keypoints = SIFTDetector().detect(input_image)
    # Draw SIFT keypoints on the input image
    input_image_sift = cv2.drawKeypoints(input_image_sift, keypoints, None,
                                         flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    # Display the output image
    cv2.imshow('SIFT detector', input_image_sift)
    # Wait until user presses a key
    cv2.waitKey()
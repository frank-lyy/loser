import numpy as np
from numpy.linalg import inv
import scipy.signal
from dataloader import *

class Algorithm:
    def __init__(self, image):
        self.image = self.process_image(image)
        self.shape = self.image.shape
        print(self.shape)
        
        self.illumination = self.image
        self.prev_reflectance = np.zeros(self.shape)
        self.reflectance = np.ones(self.shape)
        self.noise = np.zeros(self.shape)
        self.z = np.zeros(self.shape) # represents Lagrange multiplier
        self.t = np.zeros(self.shape) # gradient of illumination
        self.g = self.calculate_guidance()
        self.d = self.calculate_gradient_operator()
        
        self.beta = 0.05
        self.omega = 0.01
        self.delta = 1
        self.mu = 1
        self.rho = 1.5
        self.epsilon = 0.001

        print("Initialize Done!")

    def process_image(self, image): # extract the v-channel from hsv image
        h, s, v = cv2.split(image)
        print("Processed Image!")
        return v
        
    def calculate_guidance(self):
        factor = 1 + 10 * np.exp(-self.gradient(self.image)/10) # see formula!
        g = factor * self.gradient(self.image)[0]
        print("Calculated Guidance!")
        return g
    
    def calculate_gradient_operator(self):
        sobel_operator = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        d = scipy.signal.convolve2d(sobel_operator, self.image)
        print("Calculated Gradient Operator!")
        return d

    def vectorize(self, matrix):
        return matrix.flatten()
    
    def unvectorize(self, vector):
        return vector.reshape(self.shape)
    
    def diagonalize(self, vector):
        return np.diag(vector)

    def matrix_product(self, matrix):
        return matrix.T @ matrix

    def gradient(self, matrix):
        gradient = np.gradient(matrix)
        return gradient[0] # first dimension
    
    def shrinkage(self):
        x = self.gradient(self.illumination) + self.z/self.mu
        return np.sign(x) * np.max(np.abs(x)-self.epsilon, 0)

    def optimize(self):
        for iteration in range(1): # while np.linalg.norm(self.reflectance) - np.linalg.norm(self.prev_reflectance) > self.threshold
            self.prev_reflectance = self.reflectance

            # Update reflectance
            l = self.vectorize(self.illumination)
            l_tilda = self.diagonalize(l)
            i = self.vectorize(self.image)
            n = self.vectorize(self.noise)
            g = self.vectorize(self.g)
            r = (self.matrix_product(l_tilda) + self.omega*self.matrix_product(self.d)).inverse() @ (l_tilda @ (i-n) + self.omega*self.d.T @ g)
            self.reflectance = self.unvectorize(r)
            print("Updated Reflectance!")

            # Update illumination
            r_tilda = self.diagonalize(r)
            t = self.vectorize(self.t)
            z = self.vectorize(self.z)
            l = (2*self.matrix_product(r_tilda) + self.mu*self.matrix_product(self.d)).inverse() \
                @ (2*r_tilda @ (i-n) + self.mu*self.d.T @ (t - z/self.mu))
            self.illumination = self.unvectorize(l)
            print("Updated Illumination!")

            # Update noise
            self.noise = (self.image - self.reflectance * self.illumination) / (1 + self.delta)

            # Update auxiliary T
            self.t = self.shrinkage()

            # Update mu
            self.mu = self.mu * self.rho

            print("HI")

        return self.reflectance, self.illumination, self.noise

if __name__ == "__main__":
    directory_path = 'LOLdataset/train/low/'
    image_filename = '27.png'
    image = load_png_image(directory_path+image_filename)
    hsv_image = convert_to_hsv(image)

    optimizer = Algorithm(hsv_image)
    weights = optimizer.optimize()
    print("Optimal weights:", weights)

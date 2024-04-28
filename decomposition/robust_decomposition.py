import numpy as np
from numpy.linalg import inv
import scipy.signal
from dataloader import *

import numpy as np
from numpy.linalg import inv
import torch
import torch.nn.functional as F

class OriginalAlgorithm:
    def __init__(self, image):
        self.image = image
        self.device = image.device
        self.shape = self.image.shape

        self.illumination = self.image
        self.prev_reflectance = torch.zeros(self.shape).to(self.device)
        self.reflectance = torch.ones(self.shape).to(self.device)
        self.noise = torch.zeros(self.shape).to(self.device)
        self.z = torch.zeros(self.shape).to(self.device) # represents Lagrange multiplier
        self.t = torch.zeros(self.shape).to(self.device) # gradient of illumination
        self.g = self.calculate_guidance().to(self.device)
        self.d = self.calculate_gradient_operator().to(self.device)

        self.beta = 0.05
        self.omega = 0.01
        self.delta = 1
        self.mu = 1
        self.rho = 1.5

        self.threshold = 0.001

        print("Initialize Done! Image shape:", self.shape, self.image.dtype)

    def calculate_guidance(self):
        factor = 1+10*torch.exp(-torch.abs(self.gradient(self.image))/10) # Refer to formula
        g = factor * self.gradient(self.image)
        # print(g)
        print("Calculated Guidance!", g.shape, g.dtype)
        return torch.Tensor(g)

    def calculate_gradient_operator(self):
        gradient_matrix_size = self.shape[0] * self.shape[1]
        gradient_matrix = np.zeros((gradient_matrix_size, gradient_matrix_size))
        for i in range(gradient_matrix_size):
            if i % self.shape[1] == 0:
                gradient_matrix[i, i] = -1
                gradient_matrix[i, i+1] = 1
            elif (i % self.shape[1]) == self.shape[1]-1:
                gradient_matrix[i, i-1] = -1
                gradient_matrix[i, i] = 1
            else:
                gradient_matrix[i, i-1] = -.5
                gradient_matrix[i, i+1] = .5
        return torch.Tensor(gradient_matrix).float()
    
    def gradient_operator_test(self):
        test_gradient_operator = torch.randn(self.shape)
        using_gradient_operator = self.d @ test_gradient_operator.flatten()
        using_gradient = self.gradient(test_gradient_operator).flatten()
        print(test_gradient_operator)
        print(using_gradient_operator)
        print(using_gradient)
        print(using_gradient_operator - using_gradient)
        print(self.d)

    def vectorize(self, matrix):
        vector = matrix.flatten()
        return vector

    def unvectorize(self, vector):
        return vector.reshape(self.shape)

    def diagonalize(self, vector):
        return torch.diag(vector)

    def matrix_product(self, matrix):
        return matrix.T @ matrix

    def gradient(self, matrix):
        gradient = torch.gradient(matrix)
        return gradient[1] # first dimension

    def shrinkage(self):
        epsilon = self.beta/self.mu
        x = self.gradient(self.illumination) + self.z/self.mu
        shrinkage_left = torch.sign(x).to(self.device)
        shrinkage_right = torch.maximum(torch.abs(x)-epsilon, torch.zeros(x.shape).to(self.device)).to(self.device)

        assert shrinkage_left.shape == self.shape
        assert shrinkage_right.shape == self.shape
        return shrinkage_left * shrinkage_right

    def optimize(self):
        iteration_num = 0
        while not torch.allclose(self.reflectance, self.prev_reflectance, atol=self.threshold):
            self.prev_reflectance = self.reflectance

            # Update reflectance
            l = self.vectorize(self.illumination)
            l_tilda = self.diagonalize(l)
            l_tilda_product = l_tilda.T @ l_tilda
            d_product = self.d.T @ self.d
            i = self.vectorize(self.image)
            n = self.vectorize(self.noise)
            g = self.vectorize(self.g)
            r = (l_tilda_product + 2*self.omega*d_product).inverse() @ (l_tilda@(i-n) + 2*self.omega*(self.d.T@g))
            self.reflectance = self.unvectorize(r)
            print("Updated Reflectance!", torch.min(self.reflectance), torch.max(self.reflectance))

            # Update illumination
            r_tilda = self.diagonalize(r)
            r_tilda_product = r_tilda.T @ r_tilda
            t = self.vectorize(self.t)
            z = self.vectorize(self.z)
            l = (2*r_tilda_product + self.mu*d_product).inverse() @ (2*r_tilda@(i-n) + self.mu*(self.d.T@(t-z/self.mu)))
            self.illumination = self.unvectorize(l)
            # print("Updated Illumination!")

            # Update noise
            self.noise = (self.image - self.reflectance*self.illumination) / (1+self.delta)

            # Update auxiliary T
            self.t = self.shrinkage()
            
            # Update z
            self.z = self.z + self.mu*(self.gradient(self.illumination) - self.t)

            # Update mu
            self.mu = self.mu * self.rho

            iteration_num += 1
            print("Iteration: ", iteration_num)

        print("Total Iterations:", iteration_num)
        return self.reflectance, self.illumination, self.noise

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    directory_path = 'LOLdataset/train/low/'
    image_filename = '27.png'
    image = load_png_image(directory_path+image_filename)
    hsv_image = convert_to_hsv(image)
    _, _, v_channel = cv2.split(hsv_image)
    v_image = torch.from_numpy(v_channel)
    v_image = v_image.to(device).float()

    optimizer = OriginalAlgorithm(v_image)
    weights = optimizer.optimize()
    print("Optimal weights:", weights)

import numpy as np
from dataloader import *
import torch
import matplotlib.pyplot as plt
import time

class OriginalAlgorithm:
    def __init__(self, image):
        self.image = image
        self.device = image.device
        self.shape = self.image.shape

        self.illumination = image.to(self.device)
        self.prev_reflectance = torch.zeros(self.shape).to(self.device)
        self.reflectance = torch.ones(self.shape).to(self.device)
        self.noise = torch.zeros(self.shape).to(self.device)
        self.z = torch.zeros(self.shape).to(self.device) # represents Lagrange multiplier
        self.t = torch.zeros(self.shape).to(self.device) # gradient of illumination
        self.g = self.calculate_guidance().to(self.device)
        self.d = self.calculate_gradient_operator().to(self.device)
        self.gradient_operator_test()

        self.beta = 0.05
        self.omega = 0.01
        self.delta = 1
        self.mu = 1
        self.rho = 1.5

        self.threshold = 0.1

        # print("Initialize Done! Image shape:", self.shape, self.image.dtype)

    def calculate_guidance(self):
        factor = 1+10*torch.exp(-torch.abs(self.gradient(self.image))/10) # Refer to formula
        assert factor.shape == self.shape
        # print("Calculated Factor", torch.min(factor), torch.max(factor))
        # print("Calculated Gradient", torch.min(self.gradient(self.image)), torch.max(self.gradient(self.image)))
        gradient = self.gradient(self.image)
        mask = torch.abs(gradient) < 0.2
        gradient[mask] = 0
        g = factor * gradient
        g = torch.Tensor(g)
        # g = torch.nn.functional.relu(g)
        # print("Calculated Guidance!", g.shape, g.dtype, torch.min(g), torch.max(g))
        return g

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
        test_gradient_operator = torch.randn(self.shape).to(self.device)
        using_gradient_operator = self.d @ test_gradient_operator.flatten()
        using_gradient = self.gradient(test_gradient_operator).flatten()
        assert using_gradient_operator.shape == using_gradient.shape
        assert torch.allclose(using_gradient_operator, using_gradient)

    def normalize(self, matrix_like, max=1):
        out = matrix_like-torch.min(matrix_like)
        out = out/torch.max(out)
        return out * max

    def plot(self):
        pass
    #     renormalized_r = self.normalize(self.reflectance, 255)
    #     renormalized_r = renormalized_r.to(torch.uint8).cpu().numpy()
    #     renormalized_l = self.normalize(self.illumination, 255)
    #     renormalized_l = renormalized_l.to(torch.uint8).cpu().numpy()
    #     renormalized_n = self.normalize(self.noise, 255)
    #     renormalized_n = renormalized_n.to(torch.uint8).cpu().numpy()

    #     reflectance = hsv_to_color(small_h, small_s, renormalized_r)
    #     illumination = hsv_to_color(small_h, small_s, renormalized_l)
    #     noise = hsv_to_color(small_h, small_s, renormalized_n)

    #     renormalized_r = self.normalize(self.reflectance) # self.normalize(self.reflectance)
    #     print("Renormalized R", torch.min(renormalized_r), torch.max(renormalized_r))
    #     renormalized_l = self.normalize(self.illumination) ** (1/2.2)
    #     # renormalized_l = self.normalize(self.illumination, 255)
    #     print("Renormalized L", torch.min(renormalized_l), torch.max(renormalized_l))

    #     image_v = renormalized_r * renormalized_l
    #     image_v = self.normalize(image_v, 255)
    #     image_v = image_v.to(torch.uint8).cpu().numpy()
    #     print("Image_v", np.min(image_v), np.max(image_v))
    #     reconstructed_image = hsv_to_color(small_h, small_s, image_v)

    #     plt.figure(figsize=(10, 5))

    #     plt.subplot(1, 4, 1)
    #     plt.title('Reflectance Image')
    #     plt.imshow(reflectance)
    #     plt.axis('off')

    #     plt.subplot(1, 4, 2)
    #     plt.title('Illumination Image')
    #     plt.imshow(illumination)
    #     plt.axis('off')

    #     plt.subplot(1, 4, 3)
    #     plt.title('Noise Image')
    #     plt.imshow(noise)
    #     plt.axis('off')

    #     plt.subplot(1, 4, 4)
    #     plt.title('Reconstructed Image')
    #     plt.imshow(reconstructed_image)
    #     plt.axis('off')

    #     plt.show()

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
        return gradient[1]

    def shrinkage(self):
        epsilon = self.beta/self.mu
        x = self.gradient(self.illumination) + self.z/self.mu
        shrinkage_left = torch.sign(x).to(self.device)
        shrinkage_right = torch.maximum(torch.abs(x)-epsilon, torch.zeros(x.shape).to(self.device))

        assert shrinkage_left.shape == self.shape
        assert shrinkage_right.shape == self.shape
        return shrinkage_left * shrinkage_right

    def optimize(self):
        iteration_num = 0
        while iteration_num < 3: # not torch.allclose(self.reflectance, self.prev_reflectance, atol=self.threshold):
            # print("Algorithm has not converged:", torch.max(torch.abs(self.reflectance-self.prev_reflectance)))
            self.prev_reflectance = self.reflectance
            # self.plot()

            # Update reflectance
            l = self.vectorize(self.illumination)
            l_tilda = self.diagonalize(l)
            l_tilda_product = l_tilda.T @ l_tilda
            d_product = self.d.T @ self.d
            i = self.vectorize(self.image)
            n = self.vectorize(self.noise)
            g = self.vectorize(self.g)
            r = (l_tilda_product + 2*self.omega*d_product).inverse() @ (l_tilda@(i-n) + 2*self.omega*(self.d.T@g))
            # r = self.normalize(r)
            self.reflectance = self.unvectorize(r)
            # print("Updated Reflectance!", torch.min(self.reflectance), torch.max(self.reflectance))

            # Update illumination
            r_tilda = self.diagonalize(r)
            r_tilda_product = r_tilda.T @ r_tilda
            t = self.vectorize(self.t)
            z = self.vectorize(self.z)
            l = (2*r_tilda_product + self.mu*d_product).inverse() @ (2*r_tilda@(i-n) + self.mu*(self.d.T@(t-z/self.mu)))
            # l = self.normalize(l)
            self.illumination = self.unvectorize(l)
            # print("Updated Illumination!", torch.min(self.illumination), torch.max(self.illumination))

            # Update noise
            self.noise = (self.image - self.reflectance*self.illumination) / (1+self.delta)
            # print("Updated Noise!", torch.min(self.noise), torch.max(self.noise))

            # Update auxiliary T
            self.t = self.shrinkage()
            # print("After Shrinkage!", torch.min(self.t), torch.max(self.t))

            # Update z
            self.z = self.z + self.mu*(self.gradient(self.illumination) - self.t)
            # print("Updated Lagrange Multiplier!", torch.min(self.z), torch.max(self.z))

            # Update mu
            self.mu = self.mu * self.rho

            iteration_num += 1
            # print("Iteration: ", iteration_num)


        # print("Total Iterations:", iteration_num)
        return self.reflectance, self.illumination, self.noise

def normalize(matrix_like, max=1):
    out = matrix_like-torch.min(matrix_like)
    out = out/torch.max(out)
    return out * max

def get_noise(low_image):
    """
    Returns the noise in the low-light image argument by applying the Robust Retinex Algorithm

    Parameters:
    low_image (Tensor): V-channel of low-light image. Size is 400x600.

    Returns:
    np.ndarry: Noise of the loaded images as a numpy array
    """
    low_image = normalize(low_image)

    patch_height, patch_width = 40, 60
    num_patches_height = low_image.shape[0] // patch_height
    num_patches_width = low_image.shape[1] // patch_width

    start = time.time()
    noise = torch.zeros((400, 600))
    for i in range(num_patches_height):
        for j in range(num_patches_width):
            upper = i*patch_height
            left = j*patch_width

            patch = low_image[upper:upper+patch_height, left:left+patch_width]

            # Normalization
            patch = patch-torch.min(patch)
            patch = patch/torch.max(patch)

            optimizer = OriginalAlgorithm(patch)
            _, _, patch_noise = optimizer.optimize()
            noise[upper:upper+patch_height, left:left+patch_width] = patch_noise
    
    print(f"Obtaining noise took {time.time()-start} seconds.")
    return noise


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    directory_path = 'LOLdataset/train/'
    image_filename = '33.png'
    large_low = load_png_image(directory_path+'low/'+image_filename)
    large_high = load_png_image(directory_path+'high/'+image_filename)

    low_hsv = convert_to_hsv(large_low)
    high_hsv = convert_to_hsv(large_high)
    low_h, low_s, low_v = cv2.split(low_hsv)
    high_h, high_s, high_v = cv2.split(high_hsv)

    low_v = torch.from_numpy(low_v)
    low_image = low_v.to(device)
    low_image = low_image.float()

    low_noise = get_noise(low_image)

    high_v = torch.from_numpy(high_v)
    high_image = high_v.to(device)
    high_image = high_image.float()

    high_noise = get_noise(high_image)

    renormalized_low_n = normalize(low_noise, 255)
    renormalized_low_n = renormalized_low_n.to(torch.uint8).cpu().numpy()
    low_noise_image = hsv_to_color(low_h, low_s, renormalized_low_n)

    renormalized_high_n = normalize(high_noise, 255)
    renormalized_high_n = renormalized_high_n.to(torch.uint8).cpu().numpy()
    high_noise_image = hsv_to_color(high_h, high_s, renormalized_high_n)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.title('Low Noise Image')
    plt.imshow(low_noise_image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('High Noise Image')
    plt.imshow(high_noise_image)
    plt.axis('off')
    plt.show()

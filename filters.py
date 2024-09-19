import os

import numpy as np

import matplotlib.pyplot as plt

from common import read_img, save_img

def image_patches(image, patch_size=(16, 16)):
    """
    --- Zitima 1.a ---
    Given an input image and patch_size,
    return the corresponding image patches made
    by dividing up the image into patch_size sections.

    Input- image: H x W
           patch_size: a scalar tuple M, N
    Output- results: a list of images of size M x N
    """
    # TODO: Use slicing to complete the function
    output  = []
    M, N = patch_size  # Διαστάσεις patch
    H, W = image.shape # Διαστάσεις εικόνας
    
    for i in range(0, H, M):
        for j in range(0, W, N):
            out = image[i:i+M, j:j+N]
            if out.shape == (M, N):  # Βεβαιωση ότι το patch έχει το σωστό μέγεθος
                out = (out - np.mean(out)) / np.std(out)  # Κανονικοποίηση patch
                output.append(out)
    
    return output 


def convolve(image, kernel):
    """
    --- Zitima 2.b ---
    Return the convolution result: image * kernel.
    Reminder to implement convolution and not cross-correlation!
    Caution: Please use zero-padding.

    Input- image: H x W
           kernel: h x w
    Output- convolve: H x W
    """
    
    # Λήψη των διαστάσεων της εισόδου εικόνας και του πυρήνα
    Hi, Wi = image.shape  # Διαστάσεις εικόνας
    Hk, Wk = kernel.shape # Διαστάσεις πυρήνα
    
    # Υπολογισμός των διαστάσεων P και Q για την επένδυση
    P = Hi + Hk - 1
    Q = Wi + Wk - 1
    
    # Υπολογισμός της επένδυσης (padding) για κάθε πλευρά
    pad_top = (P - Hi) // 2
    pad_down = P - Hi - pad_top
    pad_left = (Q - Wi) // 2
    pad_right = Q - Wi - pad_left
    
    # Δημιουργία νέας εικόνας με μηδενικά που είναι μεγαλύτερη από την είσοδο κατά τις διαστάσεις της επένδυσης
    padded_height = Hi + pad_top + pad_down
    padded_width = Wi + pad_left + pad_right
    padded_image = np.zeros((padded_height, padded_width))
    
    # Υπολογισμός των ορίων αντιγραφής της αρχικής εικόνας στο κέντρο της νέας εικόνας με επένδυση
    start_i = pad_top
    end_i = start_i + Hi
    start_j = pad_left
    end_j = start_j + Wi
    
    # Αντιγραφή της αρχικής εικόνας στο κέντρο της νέας εικόνας με επένδυση
    padded_image[start_i:end_i, start_j:end_j] = image
    
    # Αντιστροφή του πυρήνα
    flipped_kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    
    # Αρχικοποίηση του αποτελέσματος με μηδενικά, με τις ίδιες διαστάσεις με την είσοδο εικόνας
    output = np.zeros((Hi, Wi))
    
    # Υλοποίηση της συνέλιξης
    for i in range(Hi):
        for j in range(Wi):
            # Υπολογισμός του εσωτερικού γινομένου και αποθήκευση του αποτελέσματος
            # Εξαγωγή της περιοχής ενδιαφέροντος
            image_patch = padded_image[i:i + Hk, j:j + Wk]
            # Υπολογισμός του εσωτερικού γινομένου και αποθήκευση του αποτελέσματος
            output[i, j] = np.sum(image_patch * flipped_kernel)
    
    return output


def edge_detection(image):
    """
    --- Zitima 2.f ---
    Return Ix, Iy and the gradient magnitude of the input image

    Input- image: H x W
    Output- Ix, Iy, grad_magnitude: H x W
    """
    # TODO: Fix kx, ky
    # Ορισμός των πυρήνων συνέλιξης για τις παραγώγους κατά x και y
    kx = np.array([[-1, 0, 1]])  # 1 x 3
    ky = np.array([[-1], [0], [1]])  # 3 x 1

    # Υπολογισμός των παραγώγων κατά x και y
    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    # TODO: Use Ix, Iy to calculate grad_magnitude
    grad_magnitude = np.sqrt(np.square(Ix) + np.square(Iy))  # gradient magnitude

    return Ix, Iy, grad_magnitude


def sobel_operator(image):
    """
    --- Zitima 3.b ---
    Return Gx, Gy, and the gradient magnitude.

    Input- image: H x W
    Output- Gx, Gy, grad_magnitude: H x W
    """
    # TODO: Use convolve() to complete the function
    # Ορισμός των πυρήνων Sobel
    sobel_x = np.array([[1, 0, -1], 
                        [2, 0, -2], 
                        [1, 0, -1]])  # 3 x 3
    sobel_y = np.array([[1, 2, 1], 
                        [0, 0, 0], 
                        [-1, -2, -1]])  # 3 x 3

    # Υπολογισμός των παραγώγων κατά x και y
    Gx = convolve(image, sobel_x)
    Gy = convolve(image, sobel_y)
    
    grad_magnitude = np.sqrt(np.square(Gx) + np.square(Gy))  # gradient magnitude
    
    return Gx, Gy, grad_magnitude


def main():
    # The main function
    img = read_img('./grace_hopper.png')
    """ Image Patches """
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")

    # -- TODO Zitima 1: Image Patches --
    # (a)
    # First complete image_patches()
    patches = image_patches(img)
    # Now choose any three patches and save them
    # chosen_patches should have those patches stacked vertically/horizontally
    chosen_patches = np.hstack(patches[:3])  # Stack patches horizontally
    
    """ THE PLOT FOR THE REPORT -- uncomment this if you want the plot
    # Plot the chosen patches
    plt.imshow(chosen_patches, cmap='gray')
    plt.title('Chosen Patches')
    plt.show()
    """
    
    save_img(chosen_patches, "./image_patches/q1_patch.png")

    # (b), (c): No code

    """ Convolution and Gaussian Filter """
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")

    # -- TODO Zitima 2: Convolution and Gaussian Filter --
    # (a): No code

    # (b): Complete convolve()

    # (c)
    # Calculate the Gaussian kernel described in the question 2.(c).
    # There is tolerance for the kernel.
    sigma = 0.572  # Ορισμός της τυπικής απόκλισης για το Gaussian φίλτρο
    kernel_size = 3  # Ορισμός του μεγέθους του πυρήνα

    # Δημιουργία του άξονα των συντεταγμένων για τον πυρήνα Gaussian
    axis_range = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    # Δημιουργία πλεγμάτων συντεταγμένων για τον υπολογισμό του 2Δ πυρήνα Gaussian
    x_grid, y_grid = np.meshgrid(axis_range, axis_range)

    # Υπολογισμός των τετραγώνων των συντεταγμένων
    x_squared = np.square(x_grid)  # Τετράγωνο των τιμών στον άξονα x
    y_squared = np.square(y_grid)  # Τετράγωνο των τιμών στον άξονα y

    # Άθροισμα των τετραγώνων των συντεταγμένων
    sum_squares = x_squared + y_squared  # Άθροισμα των τετραγώνων των συντεταγμένων x και y

    # Διαίρεση του αθροίσματος των τετραγώνων με το διπλάσιο του τετραγώνου της τυπικής απόκλισης
    sigma_squared = np.square(sigma)  # Τετράγωνο της τυπικής απόκλισης
    divided_squares = sum_squares / (2 * sigma_squared)  # Διαίρεση με 2 * σ^2

    # Υπολογισμός των εκθετικών τιμών για το Gaussian φίλτρο
    gaussian_exp = np.exp(-0.5 * divided_squares)  # Υπολογισμός του εκθετικού για κάθε στοιχείο

    # Κανονικοποίηση του πυρήνα ώστε το άθροισμα των τιμών του να είναι 1
    kernel_gaussian = gaussian_exp / np.sum(gaussian_exp)  # Κανονικοποίηση του πυρήνα
    
    filtered_gaussian = convolve(img, kernel_gaussian)
    save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")
    
    """ THE PLOT FOR THE REPORT -- uncomment this if you want the plot
    plt.imshow(filtered_gaussian, cmap='gray')
    plt.title('Filtered Gaussian Image')
    plt.show()
    """
    
    # (d), (e): No code

    # (f): Complete edge_detection()

    # (g)
    # Use edge_detection() to detect edges
    # for the orignal and gaussian filtered images.
    _, _, edge_detect = edge_detection(img)
    save_img(edge_detect, "./gaussian_filter/q3_edge.png")
    _, _, edge_with_gaussian = edge_detection(filtered_gaussian)
    save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")
    
    print("Gaussian Filter is done. ")
    
    """ THE PLOT FOR THE REPORT -- uncomment this if you want the plot
    # Σχεδίαση των αποτελεσμάτων
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(edge_detect, cmap='gray')
    plt.title('Gradient Magnitude - Original Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(edge_with_gaussian, cmap='gray')
    plt.title('Gradient Magnitude - Gaussian Filtered Image')
    
    plt.show()
    """
    
    # -- TODO Zitima 3: Sobel Operator --
    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")

    # (a): No code

    # (b): Complete sobel_operator()

    # (c)
    Gx, Gy, edge_sobel = sobel_operator(img)
    save_img(Gx, "./sobel_operator/q2_Gx.png")
    save_img(Gy, "./sobel_operator/q2_Gy.png")
    save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")
    
    """ THE PLOT FOR THE REPORT -- uncomment this if you want the plot
    # Σχεδίαση των αποτελεσμάτων
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(Gx, cmap='gray')
    plt.title('Gx - Sobel Operator (Horizontal Gradient)')

    plt.subplot(1, 3, 2)
    plt.imshow(Gy, cmap='gray')
    plt.title('Gy - Sobel Operator (Vertical Gradient)')

    plt.subplot(1, 3, 3)
    plt.imshow(edge_sobel, cmap='gray')
    plt.title('Gradient Magnitude - Sobel Operator')

    plt.show()
    """
    
    print("Sobel Operator is done. ")

    # -- TODO Zitima 4: LoG Filter --
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")

    # (a)
    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_LoG2 = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [2, 5, 0, -23, -40, -23, 0, 5, 2],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [0, 0, 3, 2, 2, 2, 3, 0, 0]])
                            
    filtered_LoG1 = convolve(img, kernel_LoG1)
    filtered_LoG2 = convolve(img, kernel_LoG2)
    
    # Use convolve() to convolve img with kernel_LOG1 and kernel_LOG2
    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")
    
    """ THE PLOT FOR THE REPORT -- uncomment this if you want the plot
    # Σχεδίαση των αποτελεσμάτων
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(filtered_LoG1, cmap='gray')
    plt.title('LoG Filter 1')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_LoG2, cmap='gray')
    plt.title('LoG Filter 2')

    plt.show()
    """
    
if __name__ == "__main__":
    main()

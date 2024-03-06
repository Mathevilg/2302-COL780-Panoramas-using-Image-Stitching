import cv2
import numpy as np

def gaussianBlur(img, size) :
    m, n = len(img), len(img[0])
    size = 5
    image = [[0 for _ in range (n+4)] for _ in range(m+4)]
    # Copy the original matrix into the center of the new matrix
    for i in range(2, m + 2):
        for j in range(2, n + 2):
            image[i][j] = img[i - 2][j - 2]
    
    gaussianBlurredImage = [[0 for _ in range (n)] for _ in range(m)]
    gaussianKernel = [[0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902],
 [0.01330621, 0.0596343,  0.09832033, 0.0596343,  0.01330621],
 [0.02193823, 0.09832033, 0.16210282, 0.09832033, 0.02193823],
 [0.01330621, 0.0596343,  0.09832033, 0.0596343,  0.01330621],
 [0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902]]
    for i in range (m) :
        for j in range (n) :
            pdt = 0
            for x in range (-2, 3, 1) :
                for y in range (-2, 3, 1) :
                    pdt += image[i+x+2][j+y+2]*gaussianKernel[x+2][y+2]
            gaussianBlurredImage[i][j] = pdt
    return np.array(gaussianBlurredImage)

def harris(img_path) :
    originalImg = cv2.imread(img_path)
    cv2.imshow("original", originalImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    print(img)
    gaussianBlurredImage = gaussianBlur(img, 5)

    m, n = len(gaussianBlurredImage), len(gaussianBlurredImage[0])
    image = [[0 for _ in range (n+2)] for _ in range(m + 2)]
    # Copy the original matrix into the center of the new matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            image[i][j] = gaussianBlurredImage[i - 1][j - 1]
    sobelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    Gx = [[0 for _ in range (n)] for _ in range(m)]
    Gy = [[0 for _ in range (n)] for _ in range(m)]
    edge_gradient = [[0 for _ in range (n)] for _ in range(m)]
    angles = [[0 for _ in range (n)] for _ in range(m)]
    for i in range (m) :
        for j in range (n) :
            pdtx = 0
            pdty = 0
            for x in range (3) :
                for y in range (3) :
                    pdtx+= sobelx[x][y]*image[i+x][j+y]
                    pdty+= sobely[x][y]*image[i+x][j+y]
            Gx[i][j] = pdtx
            Gy[i][j] = pdty
            edge_gradient[i][j] = np.sqrt(0.01*pdtx*pdtx + pdty*pdty)
            angles[i][j] = np.arctan(pdty/pdtx)
    cv2.imshow("Gx",np.array(Gx))
    cv2.imshow("GY",np.array(Gy))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    Rs = [[0 for _ in range (n-1)] for _ in range (m-1)]
    for i in range (m-1) :
        for j in range(n-1) :
            m11 = Gx[i][j]**2 + Gx[i+1][j]**2 + Gx[i][j+1]**2 + Gx[i+1][j+1]**2
            m12 = Gx[i][j]*Gy[i][j] + Gx[i+1][j]*Gy[i+1][j] + Gx[i][j+1]*Gy[i][j+1] + Gx[i+1][j+1]*Gy[i+1][j+1]
            m22 = Gy[i][j]**2 + Gy[i+1][j]**2 + Gy[i][j+1]**2 + Gy[i+1][j+1]**2
            R = (m11*m22-m12**2) - 0.04*(m11+m22)*(m11+m22)
            Rs[i][j] = R
    print(Rs)
    for i in range (m-1) :
        for j in range(n-1) :
            if Rs[i][j]>10000000 :
                originalImg[i][j]=[0,0,255]
    cv2.imshow("final", originalImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 


# harris("./data/Images/Mountain/1.jpg")
# harris("./data/Images/Office/1.jpg")

def get_keypoints(image) :
    # Use a basic feature detector (e.g., Harris corners) for simplicity
    corners = cv2.cornerHarris(image, 2, 3, 0.04)
    keypoints = np.argwhere(corners > 0.01 * corners.max())  # Adjust the threshold as needed
    return keypoints

def compute_gradient(image):
    # Compute gradient magnitude and orientation
    # print(image)
    # cv2.imshow("img", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    orientation = np.arctan2(sobely, sobelx)
    return magnitude, orientation

def get_descriptor(keypoint, image):
    # Define descriptor parameters
    patch_size = 16
    num_bins = 8
    # print(image)
    # Extract a patch around the keypoint
    x, y = keypoint
    # Ensure the coordinates are within the valid range
    x_start = max(0, x - patch_size // 2)
    x_end = min(image.shape[0], x + patch_size // 2)
    y_start = max(0, y - patch_size // 2)
    y_end = min(image.shape[1], y + patch_size // 2)

    # Extract the patch
    patch = image[x_start:x_end, y_start:y_end]

    # print(x,y)
    # print(patch)
    # Compute gradient magnitude and orientation for the patch
    magnitude, orientation = compute_gradient(patch)
    # print("len orien", len(orientation))
    
    # Create histogram of orientations
    histogram = np.zeros(num_bins)
    bin_width = 2 * np.pi / num_bins
    for i in range(x_end-x_start):
        for j in range(y_end-y_start):
            angle = orientation[i, j]
            bin_index = int(angle // bin_width) % num_bins
            histogram[bin_index] += magnitude[i, j]
    
    # Normalize the histogram
    descriptor = histogram / np.linalg.norm(histogram)
    
    return descriptor

image = cv2.imread('./data/Images/Office/1.jpg', cv2.IMREAD_GRAYSCALE)
keypoints = get_keypoints(image)
print(keypoints)

# Compute descriptors for each keypoint
descriptors = []
for keypoint in keypoints:
    descriptor = get_descriptor(keypoint, image)
    descriptors.append(descriptor)
print(descriptors[2000])

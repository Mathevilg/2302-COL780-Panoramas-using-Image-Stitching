import os
import cv2
def task1(image_dir, output_path) :
    # for task 1, assume the images 1 to 6 are ordered
    images = []


    files = os.listdir(image_dir)
    files.sort()
    for f in files:
        images.append(cv2.imread(os.path.join(image_dir,f)))
        print(f)
    # print(images) # check if a list of cv2 images gets printed
    return 0


task1("./data/images/Field", "./")
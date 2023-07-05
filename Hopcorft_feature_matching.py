import os
import cv2
import numpy as np

def convert_to_grayscale(pic1, pic2):
    g1 = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(pic2, cv2.COLOR_BGR2GRAY)
    return (g1, g2)

def detector(pic1, pic2):
    detect = cv2.ORB_create()
    kp1, des1 = detect.detectAndCompute(pic1, None)
    kp2, des2 = detect.detectAndCompute(pic2, None)
    return (kp1, des1, kp2, des2)

def Hopcroft_feature_matching(des1, des2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(des1, des2)
    # finding the hamming distance of the matches and sorting them
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def calculate_accuracy(matches, im1_gray, im2_gray, pixel_diff_threshold):
    diff_pixels = 0
    for match in matches:
        query_point = match.queryIdx
        train_point = match.trainIdx
        pixel_diff = np.abs(int(im1_gray[0][query_point]) - int(im2_gray[0][train_point]))
        if pixel_diff > pixel_diff_threshold:
            diff_pixels += 1

    total_pixels = len(matches)
    accuracy = 1 - (diff_pixels / total_pixels) if total_pixels > 0 else 0

    return accuracy

def display_output(pic1, kpt1, pic2, kpt2, best_match):
    # Drawing the feature matches using drawMatches() function
    output_image = cv2.drawMatches(pic1, kpt1, pic2, kpt2, best_match, None, flags=2)

    # Display the output image
    cv2.imshow('Output image comparison', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_folder = "output_images"
violating_images_folder = "output_violating_images"

image_names = os.listdir(image_folder)
violating_image_names = os.listdir(violating_images_folder)

desired_width = 640  # Specify the desired width for resizing
pixel_diff_threshold = 10  # Threshold for pixel-wise difference

total_accuracy = 0

for violating_img_name in violating_image_names:
    violating_image_path = os.path.join(violating_images_folder, violating_img_name)
    img2 = cv2.imread(violating_image_path)

    if img2 is None:
        print(f"Failed to read violating image: {violating_image_path}")
        continue

    key1 = []
    key2 = []
    features = []
    max_val = 0
    max_img = None

    for img_name in image_names:
        image_path = os.path.join(image_folder, img_name)
        img1 = cv2.imread(image_path)

        if img1 is None:
            print(f"Failed to read image: {image_path}")
            continue

        # Resize the images to the desired width while maintaining the aspect ratio
        img1 = cv2.resize(img1, (desired_width, int(img1.shape[0] * desired_width / img1.shape[1])))
        img2 = cv2.resize(img2, (desired_width, int(img2.shape[0] * desired_width / img2.shape[1])))

        key_pnt1, descrip1, key_pnt2, descrip2 = detector(img1, img2)

        if descrip1 is None or descrip2 is None:
            print(f"Failed to compute descriptors for image: {image_path}")
            continue

        matches = Hopcroft_feature_matching(descrip1, descrip2)
        print(f'Total number of features matching found between image 1 and image 2: {len(matches)}')

        if len(matches) > max_val:
            max_val = len(matches)
            max_img = img1.copy()
            im1 = img1.copy()
            im2 = img2.copy()
            features.append(matches)
            key1.append(key_pnt1)
            key2.append(key_pnt2)

        #display_output(img1, key_pnt1, img2, key_pnt2, matches)

    print(f'The picture with the maximum feature matching has {max_val} matches')

    # Convert images to grayscale for pixel-wise comparison
    im1_gray, im2_gray = convert_to_grayscale(im1, im2)
    accuracy = calculate_accuracy(matches, im1_gray, im2_gray, pixel_diff_threshold) * 100
    total_accuracy += accuracy

    if max_img is not None:
        display_output(im1, key1[-1], im2, key2[-1], features[-1])

    print("\n")

average_accuracy = total_accuracy / len(violating_image_names)
print(f"Average accuracy based on pixel-wise difference: {average_accuracy:.2f}%")

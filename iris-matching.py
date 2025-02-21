import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to dataset (Downloaded from kagglehub!)
DATASET_PATH = r"C:\Users\bhavy\.cache\kagglehub\datasets\monareyhanii\casia-iris-syn\versions\1"

# Initialize ORB feature detector
orb = cv2.ORB_create(nfeatures=500)

# Initialize Brute Force Matcher with Hamming distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Threshold for authentication success
MATCH_THRESHOLD = 0.75  

def extract_orb_features(image_path):
    """Extract ORB keypoints and descriptors from an image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return image, keypoints, descriptors

def match_orb_features(desc1, desc2):
    """Match ORB descriptors and compute similarity score."""
    if desc1 is None or desc2 is None:
        return 0, []  

    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)  
    score = sum([m.distance for m in matches]) / len(matches)
    
    return 1 - (score / 100), matches  

def display_images(img1, img2, keypoints1, keypoints2, matches):
    """Display images with matched features."""
    img1_kp = cv2.drawKeypoints(img1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_kp = cv2.drawKeypoints(img2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    match_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img1_kp, cmap='gray')
    plt.title("Image 1 (Keypoints)")

    plt.subplot(1, 3, 2)
    plt.imshow(img2_kp, cmap='gray')
    plt.title("Image 2 (Keypoints)")

    plt.subplot(1, 3, 3)
    plt.imshow(match_img, cmap='gray')
    plt.title("Feature Matches")

    plt.show()

def authenticate_iris(image1_path, image2_path):
    """Compare two iris images to determine if they belong to the same person."""
    img1, keypoints1, desc1 = extract_orb_features(image1_path)
    img2, keypoints2, desc2 = extract_orb_features(image2_path)
    
    score, matches = match_orb_features(desc1, desc2)
    
    print(f"Match Score: {score:.4f}")
    
    if score >= MATCH_THRESHOLD:
        print("✅ Authentication Successful: Same Person")
    else:
        print("❌ Authentication Failed: Different Person")

    display_images(img1, img2, keypoints1, keypoints2, matches)
    return score

# Load identity folders from the dataset
identity_folders = [os.path.join(DATASET_PATH, folder) for folder in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, folder))]

# Test Case 1: Same Person
if len(identity_folders) > 0:
    same_person_folder = identity_folders[0]
    same_person_images = os.listdir(same_person_folder)
    if len(same_person_images) >= 2:
        image1_same = os.path.join(same_person_folder, same_person_images[0])
        image2_same = os.path.join(same_person_folder, same_person_images[1])
        print("\n🔹 Testing Same Person:")
        authenticate_iris(image1_same, image2_same)

# Test Case 2: Different Person
if len(identity_folders) > 1:
    different_person_folder = identity_folders[1]
    image1_diff = os.path.join(identity_folders[0], os.listdir(identity_folders[0])[0])
    image2_diff = os.path.join(different_person_folder, os.listdir(different_person_folder)[0])
    print("\n🔹 Testing Different Person:")
    authenticate_iris(image1_diff, image2_diff)

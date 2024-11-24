import cv2

def apply_gaussian_filter(image_path, save_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found!")
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    cv2.imwrite(save_path, blurred)

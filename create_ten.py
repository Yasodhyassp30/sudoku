import cv2
import numpy as np
import os


def combine_and_save_images(input_folder, output_folder):
    
    folder_path = os.path.join(input_folder, "1")
    first_bit = [img for img in os.listdir(folder_path)]
    second_bit = [img for img in os.listdir(os.path.join(input_folder, "zeros"))] 
    min_size = min(len(first_bit), len(second_bit))

    for i in range(min_size):
        img = cv2.imread(os.path.join(folder_path, first_bit[i]))
        output_path = os.path.join(output_folder, "10")
        second_img = cv2.imread(os.path.join(input_folder, "zeros", second_bit[i]),cv2.IMREAD_GRAYSCALE)
        second_img = cv2.resize(second_img, (28, 28))
        _threshold,second_img = cv2.threshold(second_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        second_img = cv2.cvtColor(second_img, cv2.COLOR_GRAY2BGR)
        second_img = cv2.bitwise_not(second_img)
        combined_image = cv2.hconcat([img, second_img])
        combined_image = cv2.resize(combined_image, (28, 28))
        combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2GRAY)
        os.makedirs(output_path, exist_ok=True)
        cv2.imwrite(os.path.join(output_path, f"{i}.jpg"), combined_image)


input_folder = "assets"
output_folder = "assets"



combine_and_save_images(input_folder, output_folder)

print("Combined images created and saved successfully.")

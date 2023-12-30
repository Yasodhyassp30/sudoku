import cv2
import numpy as np
import os


def combine_and_save_images(input_folder, output_folder):
    
    folder_path = os.path.join(input_folder, "1")
    first_bit = [img for img in os.listdir(folder_path)]
    second_bit = [img for img in os.listdir(os.path.join(input_folder, "1"))] 
    min_size = min(len(first_bit), len(second_bit))

    for i in range(min_size):
        img = cv2.imread(os.path.join(folder_path, first_bit[i]))
        output_path = os.path.join(output_folder, "11")
        second_img = cv2.imread(os.path.join(input_folder, "1", second_bit[i]))
        combined_image = cv2.hconcat([img, second_img])
        combined_image = cv2.resize(combined_image, (28, 28))
        combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2GRAY)
        os.makedirs(output_path, exist_ok=True)
        cv2.imwrite(os.path.join(output_path, f"{i}.jpg"), combined_image)


input_folder = "assets"
output_folder = "assets"



combine_and_save_images(input_folder, output_folder)

print("Combined images created and saved successfully.")

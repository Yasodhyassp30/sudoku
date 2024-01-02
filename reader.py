import copy
import cv2
import numpy as np
import pytesseract
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from solver import solve_16x16,solve_9x9
import tensorflow as tf




pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
modelx16 = tf.keras.models.load_model('models/model1_16.h5')
modelx9 = tf.keras.models.load_model('models/model_9.h5')

def preprocess_image(image):
    selected_mode = 9
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0) 
    thershed =cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    edges = cv2.Canny(thershed, 50, 150, apertureSize=3)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    topleft = None
    topright = None
    bottomleft = None
    bottomright = None
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    if len(approx) >= 4:
        sorted_points = sorted(approx, key=lambda x: x[0][1])
        if (sorted_points[0][0][0] < sorted_points[1][0][0]):
            topleft = sorted_points[0].tolist()[0]
            topright = sorted_points[1].tolist()[0]
        else:
            topleft = sorted_points[1].tolist()[0]
            topright = sorted_points[0].tolist()[0]

        if (sorted_points[-1][0][0] < sorted_points[-2][0][0]):
            bottomleft = sorted_points[-1].tolist()[0]
            bottomright = sorted_points[-2].tolist()[0]
        else:
            bottomleft = sorted_points[-2].tolist()[0]
            bottomright = sorted_points[-1].tolist()[0]

    homography_matrix, _ = cv2.findHomography(np.array([topleft, topright, bottomleft, bottomright]),
                                            np.array([[0, 0], [image.shape[1], 0], [0, image.shape[0]],
                                                        [image.shape[1], image.shape[0]]]))
    

    result = cv2.warpPerspective(image, homography_matrix, (image.shape[1], image.shape[0]))
    result = cv2.resize(result, (400, 400))

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    otsued = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    eroded = cv2.erode(otsued, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.Canny(eroded, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    minArea = 400*400
    area = minArea


    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cellArea = cv2.contourArea(approx)
        if len(approx) == 4 and area//500 < cellArea< area:
            if cellArea < minArea and cellArea >0:
                minArea = cellArea

    print(area)
    print(minArea)
    print(area//minArea)
    if area//minArea in range(80,256):
        selected_mode = 9
    elif area//minArea > 256:
        selected_mode = 16
    elif area//minArea  in range(6,12):
        selected_mode = 9
    elif area//minArea  in range(12,20):
        selected_mode = 16


    print("puzzle detected:",selected_mode)


    matrix = [[0 for _ in range(selected_mode)] for _ in range(selected_mode)]
    return matrix, result, selected_mode,homography_matrix

@tf.function
def predict_function9x9(input_tensor):
    return modelx9(input_tensor)

@tf.function
def predict_function16x16(input_tensor):
    return modelx16(input_tensor)

def identify_number(image, selected_mode):
    image_resize = cv2.resize(image, (28, 28))    
    image_resize = image_resize.reshape(1, 1, 28, 28)  
    image_resize = image_resize.transpose(0, 2, 3, 1)
    if selected_mode == 9:
        loaded_model_pred = np.argmax(predict_function9x9(image_resize), axis=1)
    else:
        loaded_model_pred = np.argmax(predict_function16x16(image_resize), axis=1)
    return loaded_model_pred[0]

def read_cells(result,selected_mode,matrix,size = 400):
    def process_roi(i, j):
        roi = result[round((i * size / selected_mode)):round(((i + 1) * size/ selected_mode)),
            round((j *size / selected_mode)):round(((j + 1) *size/ selected_mode))]

        resized = cv2.resize(roi, (160, 160))
        gray_scaled = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_scaled, (3, 3), 0)
        _, roi = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        forTmodel  = roi[y:y + h, x:x + w]

        cropped_image = roi
        #cropped_image = cv2.morphologyEx(cropped_image, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        forTmodelInverse = cv2.bitwise_not(forTmodel)
        inverted_image = cv2.bitwise_not(cropped_image)
        text = identify_number(inverted_image, selected_mode)
        if selected_mode == 16 and text!=11 and text >9:
            text_tes = pytesseract.image_to_string(forTmodelInverse, config="--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789")
            if len(text_tes) != 0 and int(text_tes) == 10:
                text = int(text_tes)


        return i, j, int(text),inverted_image

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_roi, i, j) for i in range(selected_mode) for j in range(selected_mode)]
        concurrent.futures.wait(futures)


    for future in futures:
        i, j, value,digit= future.result()

        
        matrix[i][j] = value

    for i in range(selected_mode):
        print(matrix[i])

def create_grid(image, selected_mode):
    empty_white_image = np.ones_like(image) * 255

    border_thickness = 4
    empty_white_image[:border_thickness, :] = 0
    empty_white_image[-border_thickness:, :] = 0
    empty_white_image[:, :border_thickness] = 0
    empty_white_image[:, -border_thickness:] = 0

    for i in range(1, selected_mode):
        line_thickness = 4
        if i % int(np.sqrt(selected_mode)) == 0:
            cv2.line(empty_white_image,
                     (0, round(i * empty_white_image.shape[0] / selected_mode)),
                     (empty_white_image.shape[1], round(i * empty_white_image.shape[0] / selected_mode)),
                     (0, 0, 0), line_thickness, 1)
            cv2.line(empty_white_image,
                     (round(i * empty_white_image.shape[1] / selected_mode), 0),
                     (round(i * empty_white_image.shape[1] / selected_mode), empty_white_image.shape[0]),
                     (0, 0, 0), line_thickness, 1)
        else:
            cv2.line(empty_white_image, (0, round(i * empty_white_image.shape[0] / selected_mode)), (empty_white_image.shape[1], round(i * empty_white_image.shape[0] / selected_mode)), (0, 0, 0), 1, 1)
            cv2.line(empty_white_image, (round(i * empty_white_image.shape[1] / selected_mode), 0), (round(i * empty_white_image.shape[1] / selected_mode), empty_white_image.shape[0]), (0, 0, 0), 1, 1)

    return empty_white_image

def draw_on_image(image, matrix, selected_mode,matrix_copy):
    if selected_mode == 9:
        j_value = 0.35
    else:
        j_value = 0.2
    for i in range(selected_mode):
        for j in range(selected_mode):
            if matrix_copy[i][j] == 0:
                center_x = round(((j+j_value) * image.shape[1] / selected_mode))
                center_y = round(((i+0.75) * image.shape[0] / selected_mode))

                cv2.putText(image, str(matrix[i][j]), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)



def main ():

    image = cv2.imread('images/16x16.png')
    matrix, result, selected_mode,H_matrix = preprocess_image(image) 


    in_H_matrix = np.linalg.inv(H_matrix) 

    read_cells(result,selected_mode,matrix)
    matirx_copy = copy.deepcopy(matrix)
    solved =False
    if selected_mode == 16:
        solved = solve_16x16(matrix)
    else:
        solved = solve_9x9(matrix)

    if solved:
        print("solved")
        draw_on_image(result,matrix,selected_mode,matirx_copy)
    else:
        print("not solved")

    cv2.imshow('result',result)
    original_shape = (image.shape[1], image.shape[0])


    result = cv2.resize(result, (image.shape[1], image.shape[0]))
    result_back = cv2.warpPerspective(result, in_H_matrix, original_shape)
    result_back_gray = cv2.cvtColor(result_back, cv2.COLOR_BGR2GRAY)
    non_black_mask = (result_back_gray > 0).astype(np.uint8) * 255
    black_mask = cv2.bitwise_not(non_black_mask)
    other_image_black = cv2.bitwise_and(image, image, mask=black_mask)
    result_final = cv2.add(result_back, other_image_black)

    cv2.imshow('result_final',result_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
import cv2
import numpy as np
import pytesseract
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

import tensorflow as tf




pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def preprocess_image(image):
    selected_mode = 9
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0) 
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

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


    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    otsued = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    eroded = cv2.erode(otsued, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.Canny(eroded, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    minArea = image.shape[0] * image.shape[1]
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area < minArea:
                minArea = area

    area = (image.shape[0] * image.shape[1])
    

    if minArea*81 <area and minArea*256 >area:
        selected_mode = 9
    elif minArea*256 <area:
        selected_mode = 16


    print(selected_mode)

    matrix = [[0 for _ in range(selected_mode)] for _ in range(selected_mode)]

    return matrix, result, selected_mode,homography_matrix

def identify_number(image,model):
    image_resize = cv2.resize(image, (28,28))    
    image_resize = image_resize.reshape(1,1,28,28)  
    image_resize = image_resize.transpose(0, 2, 3, 1)
    loaded_model_pred = np.argmax(model.predict(image_resize),axis=1)
    return loaded_model_pred[0]

def read_cells(result,selected_mode,matrix,model):
    def process_roi(i, j):
        roi = result[round((i * result.shape[0] / selected_mode)):round(((i + 1) * result.shape[0] / selected_mode)),
            round((j * result.shape[1] / selected_mode)):round(((j + 1) * result.shape[1] / selected_mode))]

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
        text = identify_number(inverted_image, model)
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
        print(value)
        matrix[i][j] = value


    for i in range(selected_mode):
        print(matrix[i])


def create_grid(image, selected_mode):
    empty_white_image = np.ones_like(image) * 255
    for i in range(1, selected_mode):
        cv2.line(empty_white_image, (0, round(i * empty_white_image.shape[0] / selected_mode)), (empty_white_image.shape[1], round(i * empty_white_image.shape[0] / selected_mode)), (0, 0, 0), 2, 1)
        cv2.line(empty_white_image, (round(i * empty_white_image.shape[1] / selected_mode), 0), (round(i * empty_white_image.shape[1] / selected_mode), empty_white_image.shape[0]), (0, 0, 0), 2, 1)
    
    return empty_white_image


def draw_on_image(image, matrix, selected_mode):
    if selected_mode == 9:
        j_value = 0.35
    else:
        j_value = 0.2
    for i in range(selected_mode):
        for j in range(selected_mode):
            if matrix[i][j] != 0:
                center_x = round(((j+j_value) * image.shape[1] / selected_mode))
                center_y = round(((i+0.75) * image.shape[0] / selected_mode))

                cv2.putText(image, str(matrix[i][j]), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main ():
    modelx16 = tf.keras.models.load_model('model1_16.h5')
    modelx9 = tf.keras.models.load_model('model_9.h5')
    image = cv2.imread('images/p4.jpg')
    matrix, result, selected_mode,homography_matrix = preprocess_image(image)
    if selected_mode == 9:
        model = modelx9
    else:
        model = modelx16

    read_cells(result,selected_mode,matrix,model)
    grid = create_grid(result, selected_mode)
    draw_on_image(grid,matrix,selected_mode)


if __name__ == '__main__':
    main()
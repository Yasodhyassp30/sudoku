import copy
import cv2
import numpy as np
import pytesseract
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

from solver import solve_16x16,solve_9x9
import threading





pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def detect_lines(image, threshold=50, rho_resolution=1, theta_resolution=np.pi/180):
    

    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho_resolution, theta_resolution, threshold, minLineLength=100, maxLineGap=10)
    vertical_lines = 0
    horizontal_lines = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if abs(angle) < 30:
            horizontal_lines += 1
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        elif abs(angle - 90) < 30:
            vertical_lines += 1
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

    return horizontal_lines, vertical_lines


def display_frame(image, name='frame'):
    cv2.imshow(name, image)
    cv2.waitKey(1000)

def preprocess_image(image):
    selected_mode = 9
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0) 
    thershed =cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    edges = cv2.Canny(thershed, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ordered_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    topleft = None
    topright = None
    bottomleft = None
    bottomright = None
    for idx, contour in enumerate(ordered_contours):
        largest_contour = contour
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        mask_contour = np.zeros_like(image)
        cv2.drawContours(mask_contour, [largest_contour], -1, (0, 255, 255), thickness=cv2.FILLED)
        result_contour = cv2.bitwise_and(image, mask_contour)
        v_lines, h_lines = detect_lines(result_contour)
        if len(approx) == 4 and v_lines >= 8 and h_lines >= 8:
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
            break

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
    return matrix, result, selected_mode,homography_matrix,result_contour



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
        text = 0
        if selected_mode ==9 :
            text_Ts= pytesseract.image_to_string(forTmodelInverse, config="--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789")
            if len(text_Ts) != 0:
                text = int(text_Ts)
        

        if selected_mode == 16:
            text_tes = pytesseract.image_to_string(forTmodelInverse, config="--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789")
            if len(text_tes) > 1 :
                text = int(text_tes)
            elif len(text_tes) != 0:
                text = int(text_tes)



        return i, j, int(text),forTmodelInverse

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_roi, i, j) for i in range(selected_mode) for j in range(selected_mode)]
        concurrent.futures.wait(futures)

    combined_image = np.zeros((selected_mode*60, selected_mode*60), np.uint8)
    for future in futures:
        i, j, value,digit= future.result()
        digit = cv2.resize(digit, (60, 60))
        combined_image[round((i * 60)):round(((i + 1) * 60)),
        round((j *60)):round(((j + 1) *60))] = digit
        matrix[i][j] = value
    

    cv2.morphologyEx(combined_image, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cv2.dilate(combined_image, np.ones((3, 3), np.uint8), iterations=1)
    text=pytesseract.image_to_string(combined_image, config="--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789",lang='eng')

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
        thickness = 2
        scale = 1
    else:
        j_value = 0.2
        thickness = 1
        scale = 0.5
    for i in range(selected_mode):
        for j in range(selected_mode):
            if matrix_copy[i][j] == 0:
                center_x = round(((j+j_value) * image.shape[1] / selected_mode))
                center_y = round(((i+0.75) * image.shape[0] / selected_mode))

                cv2.putText(image, str(matrix[i][j]), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), thickness, cv2.LINE_AA)

def draw_on_image_initial(image, matrix, selected_mode,matrix_copy):
    if selected_mode == 9:
        j_value = 0.35
        thickness = 2
        scale = 1
    else:
        j_value = 0.2
        thickness = 1
        scale = 0.5
    for i in range(selected_mode):
        for j in range(selected_mode):
            if matrix_copy[i][j] != 0:
                center_x = round(((j+j_value) * image.shape[1] / selected_mode))
                center_y = round(((i+0.75) * image.shape[0] / selected_mode))

                cv2.putText(image, str(matrix[i][j]), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), thickness, cv2.LINE_AA)



def main ():
    
    cap = cv2.VideoCapture(0)
    counter = 1
    while True:
        ret, frame = cap.read() 
        image = frame
        
        try:
            matrix, result, selected_mode,H_matrix,contoured = preprocess_image(image) 
            display_thread = threading.Thread(target=display_frame, args=(contoured, 'frame'))
            display_thread.start()
            
            in_H_matrix = np.linalg.inv(H_matrix) 
            
            
            read_cells(result,selected_mode,matrix)
            incompleted = result.copy()
            matirx_copy = copy.deepcopy(matrix)
            initial_show = create_grid(result,selected_mode)
            draw_on_image_initial(initial_show,matrix,selected_mode,matirx_copy)

            print("detected puzzle:")
            
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

            
            original_shape = (image.shape[1], image.shape[0])


            result = cv2.resize(result, (image.shape[1], image.shape[0]))
            result_back = cv2.warpPerspective(result, in_H_matrix, original_shape)
            result_back_gray = cv2.cvtColor(result_back, cv2.COLOR_BGR2GRAY)
            non_black_mask = (result_back_gray > 0).astype(np.uint8) * 255
            black_mask = cv2.bitwise_not(non_black_mask)
            other_image_black = cv2.bitwise_and(image, image, mask=black_mask)
            result_final = cv2.add(result_back, other_image_black)

            target_width = min(initial_show.shape[1], incompleted.shape[1], result.shape[1], result_final.shape[1])
            target_height = min(initial_show.shape[0], incompleted.shape[0], result.shape[0], result_final.shape[0])

            initial_show_resized = cv2.resize(initial_show, (target_width, target_height))
            incompleted_resized = cv2.resize(incompleted, (target_width, target_height))
            result_resized = cv2.resize(result, (target_width, target_height))
            result_final_resized = cv2.resize(result_final, (target_width, target_height))


            row1 = np.concatenate((initial_show_resized, incompleted_resized), axis=1)


            row2 = np.concatenate((result_resized, result_final_resized), axis=1)


            combined_image = np.concatenate((row1, row2), axis=0)


            cv2.imshow('Combined Images', combined_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            display_thread.join()
        except:
            cv2.imshow('contoured', frame)
            cv2.waitKey(1)
    


if __name__ == '__main__':
    main()
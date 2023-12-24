import cv2
import numpy as nps
import tensorflow as tf




def process_image(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    topleft = None
    topright = None
    bottomleft= None
    bottomright=None
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    if len(approx) >= 4:
        sorted_points = sorted(approx, key=lambda x: x[0][1])
        if (sorted_points[0][0][0]<sorted_points[1][0][0]):
            topleft = sorted_points[0].tolist()[0]
            topright =sorted_points[1].tolist()[0]
        else:
            topleft = sorted_points[1].tolist()[0]
            topright =sorted_points[0].tolist()[0]

        if (sorted_points[-1][0][0]<sorted_points[-2][0][0]):
            bottomleft = sorted_points[-1].tolist()[0]
            bottomright =sorted_points[-2].tolist()[0]
        else:
            bottomleft = sorted_points[-2].tolist()[0]
            bottomright =sorted_points[-1].tolist()[0]
    
    homography_matrix, _ = cv2.findHomography(np.array([topleft,topright,bottomleft,bottomright]), np.array([[0,0],[img.shape[1],0],[0,img.shape[0]],[img.shape[1],img.shape[0]]]))

    result = cv2.warpPerspective(img, homography_matrix,(img.shape[1], img.shape[0]))
    return result

def identify_number(image,model):
    image_resize = cv2.resize(image, (28,28))    
    image_resize = image_resize.reshape(1,1,28,28)  
    image_resize = image_resize.transpose(0, 2, 3, 1)
    loaded_model_pred = np.argmax(model.predict(image_resize),axis=1)
    return loaded_model_pred[0]

def obtain_digits(img,model,matrix):
    for i in range(9):
        for j in range(9):
            roi = img[round((i*img.shape[0]/9)+3):round(((i+1)*img.shape[0]/9)-3), round((j*img.shape[1]/9)):round(((j+1)*img.shape[1]/9))]
            
            resized = cv2.resize(roi, (200, 200))
            gray_scaled = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray_scaled,(5,5),0)
            _,roi=  cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(roi, contours, -1, (0, 255, 0), 3)

            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            cropped_image = roi[y:y + h, x:x + w]
            result_image = np.ones_like(cropped_image) * 255

            inner_contours, h = cv2.findContours(cropped_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for index in range(len(inner_contours)):
                if h[0][index][3] !=-1:
                    cv2.drawContours(result_image, [inner_contours[index]], -1, 0, -1)
                    cropped_image = cv2.add(cropped_image,result_image)
                    cropped_image = cv2.GaussianBlur(cropped_image,(5,5),0)
                    break 
            inverted_image = cv2.bitwise_not(cropped_image)

            digit = identify_number(inverted_image,model)
            matrix[i][j] = int(digit)



matrix = [[0 for _ in range(9)] for _ in range(9)]
model = tf.keras.models.load_model('models/model.h5')

puzzle = cv2.imread("images/p4.jpg")
warped = process_image(puzzle)

obtain_digits(warped,model,matrix)

for i in range(9):
    print(matrix[i])

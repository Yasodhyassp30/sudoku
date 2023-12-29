import cv2
import numpy as np

image  = cv2.imread('images/16x16.png')
minarea9x9 = 1/81
minarea16x16 = 1/256
selected_mode = 9

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
topleft = None
topright = None
bottomleft= None
bottomright=None
largest_contour = max(contours, key=cv2.contourArea)
epsilon = 0.02 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)
if len(approx) == 4:
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

homography_matrix, _ = cv2.findHomography(np.array([topleft,topright,bottomleft,bottomright]), np.array([[0,0],[image.shape[1],0],[0,image.shape[0]],[image.shape[1],image.shape[0]]]))

result = cv2.warpPerspective(image, homography_matrix,(image.shape[1], image.shape[0]))
result = cv2.GaussianBlur(result,(5,5),0)

gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


minArea = image.shape[0]*image.shape[1]
for cnt in contours:
    epsilon = 0.009 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    if len(approx) == 4:
        cv2.drawContours(result, [approx], 0, (0, 0, 255), 3)
        area = cv2.contourArea(approx)
        if area < minArea:
            minArea = area

minArea_precentage = minArea/(image.shape[0]*image.shape[1])


if abs(minArea_precentage-minarea9x9) < abs(minArea_precentage-minarea16x16):
    selected_mode = 9
else:
    selected_mode = 16

for i in range(selected_mode):
    for j in range(selected_mode):
        roi = result[round((i*image.shape[0]/selected_mode)):round(((i+1)*image.shape[0]/selected_mode)), round((j*image.shape[1]/selected_mode)):round(((j+1)*image.shape[1]/selected_mode))]
        
        resized = cv2.resize(roi, (200, 200))
        gray_scaled = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_scaled,(5,5),0)
        _,roi=  cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        cropped_image = roi[y:y + h, x:x + w]

        inverted_image = cv2.bitwise_not(cropped_image)
        cv2.imshow('Inverted', inverted_image)
        cv2.waitKey(0)

    
cv2.destroyAllWindows() 
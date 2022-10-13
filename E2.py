import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

img = cv2.imread('laca4.PNG')
def Detector(img):
    start = time.time()
    img = img
    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # print(dim)
    # resize image
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_b =  cv2.blur(gray, (5,5))
    
    # cv2.imshow('Grises', gray)
    # cv2.imshow('Grises_b', gray_b)
    
    
    # hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # #print(hist.argmax())
    # plt.figure(figsize=(5,5))
    # plt.title("Grayscale Histogram")
    # plt.xlabel("Bins")
    # plt.ylabel("# of Pixels")
    # plt.plot(hist)
    # plt.xlim([0, 256])
    # plt.show()

    
    ret,th1 = cv2.threshold(gray_b,252,255,cv2.THRESH_BINARY)

    
    
    img_dil = th1.copy()
    kernel = np.ones((15,15),np.uint8)

    img_dil = cv2.erode(img_dil, (5,5),iterations = 2)
    for i in range(3):
        img_dil = cv2.dilate(img_dil, kernel,iterations = i)
    
    img_can = cv2.Canny(img_dil, 100, 200)
    contours, hierarchy = cv2.findContours(img_can, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    idx =0
    im_rec = img.copy()
    for cnt in contours:
        idx += 1
        x,y,w,h = cv2.boundingRect(cnt)
        U = np.sqrt((w**2 + h**2))
        print(U)
        # U = 60
        # if w >= U or h >= U:
        if U > 70:
            cv2.rectangle(im_rec,(x,y),(x+w,y+h),(0,0,255), 5)


    return im_rec



cap = cv2.VideoCapture('video7.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
print('Frames per second : ', fps,'FPS')


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('output_2.mp4', fourcc, 10, (768, 432))
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        img = Detector(frame)
        cv2.imshow('Inspector', img)
        writer.write(img)
        # out.write(img)
    else:
        break
    cv2.waitKey(int(1000/fps))


cv2.destroyAllWindows()
cap.release()





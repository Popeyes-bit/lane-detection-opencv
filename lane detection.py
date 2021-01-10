import numpy as np
from PIL import ImageGrab
import cv2
import time
#from directinputs import ReleaseKey,PressKey, W, A, S, D
#import pyautogui
from numpy import ones,vstack
from numpy.linalg import lstsq
from statistics import mean


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked




def process_img(image):
    original_image = image
    # convert to gray
    processed_img = cv2.cv

def drawlines(img,lines):
    try:
        for line in lines:
            cords=line[0]
            cv2.line(img,(cords[0],cords[1]),(cords[2],cords[3]),[255,255,255],5)
    except:
        pass
def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(processed_img, 127, 255, cv2.THRESH_TRUNC)
    processed_img=blackAndWhiteImage

    processed_img = cv2.Canny(processed_img, threshold1=50, threshold2=100)
    ret, processed_img = cv2.threshold(processed_img, 130, 145, cv2.THRESH_BINARY)
    processed_img = cv2.GaussianBlur(processed_img,(5,5),0)
    ret, processed_img = cv2.threshold(processed_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    vertices = np.array([[800,600],[581,629],[493,381],[325,385],[215,627],[189,623],[203,437],[7,379],[307,293],[533,291],[805,383]], np.int32)
    processed_img = roi(processed_img, [vertices])
    lines = cv2.HoughLinesP(processed_img   , 1, np.pi/180, 50,np.array([]),30,20)
    drawlines(processed_img,lines)
    vertical = np.copy(processed_img)
    #Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = rows // 30
    #Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    #Show extracted vertical lines
   	#show_wait_destroy("vertical", vertical)
    return processed_img


##for i in list(range(4))[::-1]:
##    print(i+1)
##    time.sleep(1)


def main():
    #initial time before loop starts
    last_time=time.time()
    while(True):
        
        screen =  np.array(ImageGrab.grab(bbox=(0,30,800,640)))
        new_screen= process_img(screen)
        cv2.namedWindow('window1', cv2.WINDOW_NORMAL)

       # cv2.imshow('window2',original_image)
        cv2.imshow('window1',new_screen)
        print("loop took time="+str(time.time()-last_time))
        last_time=time.time()
        # cv2.imshow('window2',cv2.cvtColor(screen,cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()

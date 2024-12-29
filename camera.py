import pathlib
import cv2
import HandModule as hm
import os
import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0
detector = hm.handDetector()

defaultPath = 'dataset/o/capture_{}'
while os.path.exists(defaultPath.format(img_counter)):
    img_counter+=1
os.makedirs(defaultPath.format(img_counter))
img_name = defaultPath.format(img_counter) + "/opencv_frame.png"

while True:
    ret, frame = cam.read()
    success, img = cam.read()
    img = detector.findHands(img, draw=True )
    lmList = detector.findPosition(img, draw=False)
                
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", img)


    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        cords_array = []
        if len(lmList) != 0:
            for j in range (len(lmList)):
                cords_array.append(lmList[j])
                print(cords_array)
        with open(defaultPath.format(img_counter) + "/text_data.txt","w", encoding='utf-8-sig') as text_file:
            text_file.write(str(cords_array))
            
        
        cv2.imwrite(img_name, img)
        print("{} written!".format(img_name))
        img_counter += 1
        break

cam.release()

cv2.destroyAllWindows()

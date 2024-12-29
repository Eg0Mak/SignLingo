from tkinter import *
from tkinter import ttk
import cv2
from threading import Thread
import numpy as np
import sys
import mediapipe as mp
import HandModule as hm
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math
import time
import keyboard
import tensorflow
from PIL import Image, ImageTk
import win10toast



root = Tk()
root.title("SignLingo")
root.geometry("1000x500")
root.resizable(width=False, height=False)
root['background']='#E6E6FA'
root.iconbitmap('images/icon/test.ico')


btn_clear = Button(text="Clear warning image",background='white', foreground='#242424', font="TimesNewRoman 10")

# def action(lmList,img) :
#      if len(lmList) != 0:
#             img_name = "opencv.png"
#             cv2.imwrite(img_name,img)
#             cv2.destroyAllWindows()
#             for j in range (len(lmList)):
#                 print(lmList[j])


# Функция кнопки + label + кнопка для того, чтобы сохранять инфу в переменную sentence из label
text_get = Label(text="Your text here", foreground="#5E9CFE", background='white', font="TimesNewRoman 10 bold")
i = -1
def Set_text():
    global i 
    i += 1
    text = enter.get()
    sentence = text.lower()
    def Split(sentence):
        return[char for char in sentence]
    alphabet = Split(sentence)
    if(i<=len(alphabet)-1):
        def ShowImage(text, i):
            def clear_label_image():
                    global i
                    i = -1
                    lol.config(image='')
                    text.clear()
                    btn.config(command=Set_text)
                    enter.delete(0, END)
            def lblImage():
                img_invalid = cv2.imread('images\char_invalid_syntax\symbol_invalid_syntax.png')
                blue,green,red = cv2.split(img_invalid)
                img = cv2.merge((red,green,blue))
                im = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=im)
                return imgtk
            # def run_once(f):
            #     def wrapper(*args, **kwargs):
            #         if not wrapper.has_run:
            #             wrapper.has_run = True
            #             return f(*args, **kwargs)
            #     wrapper.has_run = False
            #     return wrapper
            try:
                img_path = cv2.imdecode(np.fromfile('images/char_'+text[i]+'/symbol_'+text[i]+'.png',dtype=np.uint8),cv2.IMREAD_UNCHANGED)
                cv2.imshow(str(i), img_path)  
            except:
                if text[i] == " ":
                    img_space = cv2.imread('images\char_\symbol_.png')
                    cv2.imshow(str(i), img_space)
                else:
                    imgtk = lblImage()
                    lol = Label(root, image = imgtk, background='#E6E6FA')
                    lol.image = imgtk
                    lol.place(x = 450, y = 40)
                    #cv2.imshow(str(i), img_invalid)
                    btn_clear.config(command=clear_label_image)
                    btn.config(command='')
                           
        ShowImage(alphabet, i)
    else:
        cv2.destroyAllWindows()
        i = -1

text_get.place(x = 401, y = 200)
btn_clear.place(x = -1, y = 474)

# Кнопка закрытия окон
def Close_button():
    cv2.destroyAllWindows()

# Вывод слова через нейронку
lbl = Label(text="Reverse translate", foreground="#FFFFFF", background='#5E9CFE', font=("TimesNewRoman", 11))
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# Функция кнопки запуска нейросети
def neural():
    def function():
        detector = HandDetector(maxHands=1)

        offset = 20
        imgSize = 300

        classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
        labels = ["А","Б","В","Г","Д","Е","О","Ж","З","К","Л","М","Н","П","Р","С","Т","У","Ф","Х","Ц","Ч","Ш","Ъ","Ы","Ь","Э","Ю","Я"]

        sentence = ""

        while True:
            success, img = cap.read()
            imgOutput = img.copy()
            hands, img = detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
        
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        
                imgCropShape = imgCrop.shape
        
                aspectRatio = h / w
        
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    print(prediction, index)
        
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
        
                #cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                        # (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 165, 255), 2)
                #cv2.rectangle(imgOutput, (x-offset, y-offset),
                        #(x + w+offset, y + h+offset), (255, 0, 255), 4)
        

                char = str(labels[index])
                if keyboard.is_pressed('ENTER'):
                    sentence += char
                    print("Catched")
                    time.sleep(0.5)
                    lbl["text"] = sentence



            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
                break
                #cv2.imshow("ImageCrop", imgCrop)
                #cv2.imshow("ImageWhite", imgWhite)
        
            cv2.imshow("Image", imgOutput)
            cv2.waitKey(1)
    try:
        function()
    except cv2.error:
        toaster = win10toast.ToastNotifier()
        toaster.show_toast("Sign Lingo Warning", "Hand must be in capture", duration = 3)
        function()


def Reset():
    global i
    i = -1
    enter.delete(0, END)
    cv2.destroyAllWindows()
    lbl["text"] = "Reverse translate"

lbl.place(x = 402, y = 260)

# Иконка для нейронки
neural_photo = PhotoImage(file="images\Icon_button\set4.png")

# Иконка для кнопки
photo_test = PhotoImage(file="images\Icon_button\setv1.png")
photo = PhotoImage(file = "images\Icon_button\set.png")

#Кнопка для сохранение текста в Labe
btn = Button(command=Set_text, width=20, height=20, background='white', foreground='#5E9CFE', font=('TimesNewRoman', 8), image = photo_test, compound=TOP)
btn.place(x = 600, y = 220, width=28)
#btn_close = Button(text='Close all windows', command=Close_button, background='white', foreground='#242424', font="TimesNewRoman 10").place(x = -1, y = 474)
btn_neural = Button(text="Neural ", command=neural, foreground='#242424', background='white', font="TimesNewRoman 10", image=neural_photo, compound=RIGHT, activebackground = 'white').place(x = 556, y = 260, height=24)
btn_reset = Button(text="Reset", command=Reset, foreground='#242424', background='white', font="TimesNewRoman 10", activebackground = 'white').place(x = 956, y = 474)



# Окно для ввода текста
enter = Entry(font=('TimesNewRoman', 12))
enter.place(x=400, y = 220, width=200, height=25)
enter.focus()


root.mainloop()







 
#cv2.destroyAllWindows()

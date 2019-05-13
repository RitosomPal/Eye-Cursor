import numpy as np
import cv2
import mouse
import ctypes

detector = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)


def map(n, start1, stop1, start2, stop2):
    return int(((n-start1)/(stop1-start1))*(stop2-start2)+start2)


class screen():
    X = ctypes.windll.user32.GetSystemMetrics(0)
    Y = ctypes.windll.user32.GetSystemMetrics(1)


def DETECT_EYE(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye = detector.detectMultiScale(gray, 1.3, 5)
    singleEye = gray
    rows, cols = singleEye.shape
    frows, fcols = gray.shape

    crd = [(0, 0, 0, 0), (0, 0, 0, 0)]

    for (x, y, w, h) in eye:
        y += 16
        h -= 30
        crd[0] = (x, y, w, h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        singleEye = gray[y:y+h, x:x+w]
        break
    singleEye = cv2.GaussianBlur(singleEye, (9, 9), 0) 
    _, thre = cv2.threshold(singleEye, 75, 255, cv2.THRESH_BINARY_INV) # Chang Threshold as required
    _, cnt, _ = cv2.findContours(
        thre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(cnt, key=lambda x: cv2.contourArea(x), reverse=True)

    for cn in cnt:
        (x, y, w, h) = cv2.boundingRect(cn)
        crd[1] = (x, y, w, h)
        # cv2.drawContours(singleEye, [cn], -1, (0, 0, 255), 3)
        cv2.rectangle(singleEye, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.line(singleEye, (x + int(w/2), 0),
                 (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(singleEye, (0, y + int(h/2)),
                 (cols, y + int(h/2)), (0, 255, 0), 2)
        break

    (x1, y1, w1, h1) = crd[0]
    (x2, y2, w2, h2) = crd[1]

    x = x1 + x2
    y = y1 + y2

    cx = x + int(w2/2)
    cy = y + int(h2/2)

    cv2.line(frame, (cx, 0), (cx, frows), (0, 255, 0), 1)
    cv2.line(frame, (0, cy), (fcols, cy), (0, 255, 0), 1)
    cv2.rectangle(frame, (x, y), (x+w2, y+h2), (255, 200, 0), 2)

    if x1 != 0 and (x1+w1) != 0:
        mx = map(cx, x1, x1+w1, 0, screen.X)
        my = map(cy, y1, y1+h1, 0, screen.Y)
        mouse.move(mx, my)

    # cv2.imshow("SingleEye", singleEye)

    return crd


while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    DETECT_EYE(frame)

    cv2.imshow("Eye", frame)

    # Exit
    k = cv2.waitKey(5)
    if k == ord('q'):
        break


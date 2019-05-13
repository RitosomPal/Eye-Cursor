# Eye-Cursor
Move mouse cursor by tracking your retina.

## Main Eye Tracking Function
![Screenshot](https://github.com/RitosomPal/Eye-Cursor/blob/master/screenshot/EyeTrack.png "Screenshot")
```py
detector = cv2.CascadeClassifier('haarcascade_eye.xml')

def DETECT_EYE(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    eye = detector.detectMultiScale(gray, 1.3, 5) 
    singleEye = gray
    rows, cols = singleEye.shape
    frows, fcols = gray.shape
    
    for (x, y, w, h) in eye:
        y += 16
        h -= 30
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        singleEye = gray[y:y+h, x:x+w]
        break
        
    # singleEye = cv2.resize(singleEye, (300, 200))
    singleEye = cv2.GaussianBlur(singleEye, (9, 9), 0)
    _, thre = cv2.threshold(singleEye, 75, 255, cv2.THRESH_BINARY_INV) # Threshold value may varey due to environment light.
    _, cnt, _ = cv2.findContours(thre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(cnt, key=lambda x: cv2.contourArea(x), reverse=True)
    
    for cn in cnt:
        (x, y, w, h) = cv2.boundingRect(cn)
        cv2.drawContours(singleEye, [cn], -1, (0, 0, 255), 3)
        cv2.rectangle(singleEye, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.line(singleEye, (x + int(w/2), 0),(x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(singleEye, (0, y + int(h/2)),(cols, y + int(h/2)), (0, 255, 0), 2)
        break
    cv2.imshow("SingleEye", singleEye)
```


### Todos

 - Add slider to alter threshold value.
 - Map retina movement range with the screen resolution.

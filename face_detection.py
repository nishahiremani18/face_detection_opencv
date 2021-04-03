import cv2

# draw boundary around the detected object
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords



def detect(img, faceCascade, eyeCascade, noseCascade, mouthCascade):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")
    
    if len(coords)==4:
        # region of interset (roi)
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        coords = draw_boundary(roi_img, eyeCascade, 1.1, 12, color['red'], "Eye")
        coords = draw_boundary(roi_img, noseCascade, 1.1, 4, color['green'], "Nose")
        coords = draw_boundary(roi_img, mouthCascade, 1.1, 20, color['white'], "Mouth")
    return img


# Loading classifiers
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
noseCascade = cv2.CascadeClassifier('nose.xml')
mouthCascade = cv2.CascadeClassifier('Mouth.xml')

#loading web cam
video_capture = cv2.VideoCapture(0)

while True:
    _, img = video_capture.read()
    img = detect(img, faceCascade, eyesCascade, noseCascade, mouthCascade)
    cv2.imshow("face detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
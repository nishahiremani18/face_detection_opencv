import cv2

# Method to generate dataset to recognize a person
def generate_dataset(img, id, img_id):
    # write image in data dir
    cv2.imwrite("data/user."+str(id)+"."+str(img_id)+".jpg", img)

# Method to draw boundary around the detected feature
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    # Converting image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords

def detect(img, faceCascade, img_id):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")
    # If feature is detected, the draw_boundary method will return the x,y coordinates and width and height of rectangle else the length of coords will be 0
    if len(coords)==4:
        # Updating region of interest by cropping image
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        # Assign unique id to each user
        user_id = 1
        # img_id to make the name of each image unique
        generate_dataset(roi_img, user_id, img_id)

    return img

# Loading classifiers
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
noseCascade = cv2.CascadeClassifier('nose.xml')
mouthCascade = cv2.CascadeClassifier('Mouth.xml')

# video_capture= 0 if you are using desktop camera and 1 or -1 for external cam
video_capture = cv2.VideoCapture(0)

while True:
    _, img = video_capture.read()
    img = detect(img, faceCascade, eyesCascade, noseCascade, mouthCascade)
    cv2.imshow("face detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
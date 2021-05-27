from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",
    help="path to input video")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based)

detector = dlib.get_frontal_face_detector()

# initialize the video stream and sleep for a bit, allowing the
# camera sensor to warm up
print("[INFO] camera sensor warming up...")
frame = cv2.imread(args['input'])
# vs = cv2.VideoCapture('videoplayback.mp4')
# vs = VideoStream(usePiCamera=True).start() # Raspberry Pi


# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    # _, frame = vs.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


 
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
    gender_list = ['Male', 'Female']

    # check to see if a face was detected, and if so, draw the total
    # number of faces on the frame
    if len(rects) > 0:
        text = "{} face(s) found".format(len(rects))
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 255), 2)

    # loop over the face detections
    for rect in rects:
        # compute the bounding box of the face and draw it on the
        # frame
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),
            (0, 255, 0), 1)
        face_img = frame[bY:bY+bH, bX:bX+bW].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        age_net=cv2.dnn.readNetFromCaffe('model/deploy_age.prototxt', 'model/age_net.caffemodel')

        gender_net = cv2.dnn.readNetFromCaffe('model/deploy_gender.prototxt', 'model/gender_net.caffemodel')

        #Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        print(gender)
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        print(age)

        overlay_text = "%s %s" % (gender, age)
        cv2.putText(frame, overlay_text, (bX, bY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        cv2.imwrite('out4.jpg',frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break




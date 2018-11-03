# USAGE
# frameworkpython3 ./budz_shape_predictor2.py --shape-predictor ./budz3_predictor_landmarks.dat --video ../../face_recognition-master/examples/bbunny-clip3.mp4 --output ./budz_shape_predictor2-1.avi

#frameworkpython3 ./budz_shape_predictor2.py --shape-predictor ./bb0_predictor_landmarks.dat --video ../../face_recognition-master/examples/bbunny-clip0.mp4 --output ./bb0predictor.avi

#frameworkpython3 ./budz_shape_predictor2.py --shape-predictor ./budz_predictor_landmarks.dat --video ../../face_recognition-master/examples/bbunny-clip0.mp4 --output ./budz_shape_predictor2-1.avi
# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-v", "--video", required=True,
    help="path to input video")
ap.add_argument("-o", "--output", type=str,
    help="path to output video")
# ap.add_argument('-w', '--weights', default='./mmod_human_face_detector.dat',
#                 help='path to weights file')
# ap.add_argument('-w', '--weights', default='./dlib_face_recognition_resnet_model_v1.dat',
#                 help='path to weights file')
ap.add_argument("-d", "--display", type=int, default=1,
                help="whether or not to display output frame to screen")
args = vars(ap.parse_args())

# stream = cv2.VideoCapture("../../face_recognition-master/examples/bbunny-clip2-1.mp4")
stream = cv2.VideoCapture(args["video"])
frame_number = 0
length = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
writer = None
# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if int(major_ver)  < 3 :
    fps = stream.get(cv2.cv.CV_CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else :
    fps = stream.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
# detector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
# detector = dlib.face_recognition_model_v1(args["weights"])
detector = dlib.get_frontal_face_detector()
# detector = dlib.fhog_object_detector("bb0.svm")
# detector = dlib.simple_object_detector("b2detector.svm")
# predictor = dlib.shape_predictor("budz3_predictor_landmarks.dat")
predictor = dlib.shape_predictor(args["shape_predictor"])
# recognizer = dlib.face_recognition_model_v1("./dlib_face_recognition_resnet_model_v1.dat")
while True:
    (grabbed, frame) = stream.read()
    if frame_number == 0:
    	print("video dims: {} x {}".format(frame.shape[0], frame.shape[1]))
    	vid_h = frame.shape[0]
    	vid_w = frame.shape[1]
#     	ratio = 1/(500/vid_w)
#     	print("1:1 ratio: {}".format(ratio))
    frame_number += 1
    # if the frame was not grabbed, then we have reached the
    # end of the stream
    if not grabbed:
        break
# load the input image, resize it, and convert it to grayscale
# image = cv2.imread(args["image"])
#     frame = imutils.resize(frame, width=500)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# detect faces in the grayscale image
#     rects = detector(gray, 1)
    rects = detector(rgb_img, 1)
    rects = [rect.rect for rect in rects]

# loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
#         shape = predictor(gray, rect)
        shape = predictor(rgb_img, rect)
        
#         face_embedding = recognizer.compute_face_descriptor(rgb_img, shape)
#         face_embedding = [x for x in face_embedding]
#         face_embedding = np.array(face_embedding, dtype="float32")[np.newaxis, :]
        shape = face_embedding
        print("face_embedding {}".format(face_embedding))
        
#         shape = face_utils.shape_to_np(shape)

    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
#         x *= ratio
#         y *= ratio
#         w *= ratio
#         h *= ratio
#         x = int(x)
#         y = int(y)
#         w = int(w)
#         h = int(h)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the face number
#         cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
        for (x, y) in shape:
#         	x *= ratio
#         	y *= ratio
#         	x = int(x)
#         	y = int(y)
        	cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

# # save output image
# cv2.imwrite("cnn_face_detection.png", image)

    # if the video writer is None *AND* we are supposed to write
    # the output video to disk initialize the writer
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 24,
            (frame.shape[1], frame.shape[0]), True)
#             (open_cv_image.shape[1], open_cv_image.shape[0]), True)
    # if the writer is not None, write the frame with recognized
    # faces to disk
    if writer is not None:
        writer.write(frame)
    #-- Write the resulting image to the output video file
        print("Writing frame {} / {}".format(frame_number, length))
    # check to see if we are supposed to display the output frame to
    # the screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
#         cv2.imshow("Frame", open_cv_image)
        cv2.waitKey(int(1000/fps))
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# # close all windows
# cv2.destroyAllWindows()
stream.release();
# check to see if the video writer point needs to be released
if writer is not None:
    writer.release()
# USAGE
#frameworkpython3 ./budz_cnn_shape_predictor.py --shape-predictor ./budz_predictor_landmarks.dat --video ../../face_recognition-master/examples/bbunny-clip0.mp4 --output ./budz_shape_predictor0.avi
# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import math
from random import randint

def vidinit(args):
    stream = cv2.VideoCapture(args["video"])
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
    
    return stream, length, writer, fps

def frameincrement(frame_number):
    (grabbed, frame) = stream.read()
    frame_number += 1
    # if not grabbed, then no more stream
    return grabbed, frame, frame_number

def dispwrite(writer, args, vid_h, vid_w, frame, fps):
    # if the video writer is None *AND* we are supposed to write
    # the output video to disk initialize the writer
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 24, (vid_h, vid_w), True)

#             (frame.shape[1], frame.shape[0]), True)
#             (open_cv_image.shape[1], open_cv_image.shape[0]), True)
    # if the writer is not None, write the frame with recognized
    # faces to disk
    if writer is not None:
        writer.write(frame)
        print("frame write")
    # check to see if we are supposed to display the output frame to
    # the screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
#         cv2.imshow("Frame", open_cv_image)
        cv2.waitKey(int(1000/fps))
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
#         if key == ord("q"):
#             break
    return frame, key, writer

def calculateDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def drawffeatures(rgb_img, face, leyepos, frame, reyepos, nosepos):
    shape = predictor(rgb_img, face)
    embedding = recognizer.compute_face_descriptor(rgb_img, shape)
    embedding = [x for x in embedding]
    embedding = np.array(embedding, dtype="float32")[np.newaxis, :]
    shape = face_utils.shape_to_np(shape)
#         label = recognize_face(embedding, embeddings, labels)
    e1cx = int((shape[0][0] - shape[1][0])/ 2 + shape[1][0])
    e2cx = int((shape[2][0] - shape[3][0])/ 2 + shape[3][0])
    ncy = int((shape[0][1] - shape[4][1])/ 2 + shape[2][1])
    radius = (shape[0][0] - shape[1][0]) / 2
    r1 = int(radius * 2)
    r2 = int(radius * 1.5)
    nr = int(radius * 3)
    for e in range(60):
        
        e1x = randint(-35, 30) + e1cx
        e1y = randint(-30, 35) + int(shape[1][1])
        e1r = randint(100, 165)
        e1g = randint(50, 140)
        e1b = randint(0, 70)
        e1pos = (e1x, e1y)
        e1clr = (e1b, e1g, e1r)
        leyepos.append(e1pos)
        lcolor.append(e1clr)
        

        e2x = randint(-25, 20) + e2cx
        e2y = randint(-20, 20) + int(shape[3][1])
        e2r = randint(100, 165)
        e2g = randint(50, 140)
        e2b = randint(0, 70)
        e2pos = (e2x, e2y)
        e2clr = (e2b, e2g, e2r)
        reyepos.append(e2pos)
        rcolor.append(e2clr)

        nx = randint(-20, 20) + int(shape[4][0])
        ny = randint(-10, 20) + int(shape[4][1])
        nr = randint(100, 165)
        ng = randint(50, 140)
        nb = randint(0, 70)
        npos = (nx, ny)
        nclr = (nb, ng, nr)
        nosepos.append(npos)
        ncolor.append(nclr)
        
        
        if len(nosepos) > 2000:
            nosepos[:-1].copy()
            ncolor[:-1].copy()
            reyepos[:-1].copy()
            rcolor[:-1].copy()
            leyepos[:-1].copy()
            lcolor[:-1].copy()

        for i, pos in enumerate(leyepos):
            x1, y1 = list(pos)
#             print("x1: {}, y1: {}".format(x1, y1))
#             cv2.circle(frame, (x1, y1), 1, lcolor[i], -1)
            if ((x1 - e1cx) ** 2) / (r2 ** 2) + ((y1 - int(shape[1][1])) ** 2) / (r1 ** 2) <= 1 :
#                 cv2.circle(frame, pos, 1, lcolor[i], -1)
#                 cv2.circle(frame, pos, randint(1, 5), lcolor[randint(len(lcolor) - 1)], -1)
                cv2.circle(frame, pos, randint(1, 5), lcolor[0], -1)
            center = (e1cx, int(shape[1][1]))
            if r1 >= 50:
                r1 = int((shape[0][0] - shape[1][0])/1)
                r2 = int(r1 / 1.5)
            axes= (r1, r2)
            rotation_angle = randint(0, 10)
            start_angle = randint(0, 360)
            end_angle = randint(30, 360)
            color = (randint(30, 120), randint(50, 140), randint(100, 165))
            thickness = 1
            cv2.ellipse(frame, center, axes, rotation_angle, start_angle, end_angle, color, thickness)
            while r1 < 50:
                r1 += 1
                r2 += 1
            
        for i, pos in enumerate(reyepos):
            x1, y1 = list(pos)
#             print("x1: {}, y1: {}".format(x1, y1))
#             cv2.circle(frame, (x1, y1), 1, lcolor[i], -1)
            if ((x1 - e2cx) ** 2) / (r1 ** 2) + ((y1 - int(shape[1][1])) ** 2) / (r2 ** 2) <= 1 :
                cv2.circle(frame, pos, randint(1, 5), rcolor[i], -1)

            center = (e2cx, int(shape[1][1]))
            if r1 >= 50:
                r1 = int((shape[0][0] - shape[1][0])/1)
                r2 = int(r1 / 1.5)
            axes= (r1, r2)
            rotation_angle = randint(0, 10)
            start_angle = randint(0, 360)
            end_angle = randint(30, 360)
            color = (randint(30, 120), randint(50, 140), randint(100, 165))
            thickness = 1
            print("center {}, axes {}, rotation_angle {}, start_angle {}, end_angle {}, color {}, thickness {}" .format(center, axes, rotation_angle, start_angle, end_angle, color, thickness))
            
            cv2.ellipse(frame, center, axes, rotation_angle, start_angle, end_angle, color, thickness)
            while r1 < 50:
                r1 += 1
                r2 += 1
                        
        for i, pos in enumerate(nosepos):
            x1, y1 = list(pos)
#             print("x1: {}, y1: {}".format(x1, y1))
#             cv2.circle(frame, (x1, y1), 1, lcolor[i], -1)
            if ((x1 - nx) ** 2) / (r1 ** 2) + ((y1 - int(shape[1][1])) ** 2) / (r2 ** 2) <= 1 :
                cv2.circle(frame, pos, randint(1, 5), ncolor[i], -1)
                
            center = (int(shape[4][0]), int(shape[1][1]))
            if r1 >= 50:
                r1 = int((shape[0][0] - shape[1][0])/1)
                r2 = int(r1 / 1.5)
            axes= (r1, r2)
            rotation_angle = randint(0, 10)
            start_angle = randint(0, 360)
            end_angle = randint(30, 360)
            color = (randint(30, 120), randint(50, 140), randint(100, 165))
            thickness = 1
            cv2.ellipse(frame, center, axes, rotation_angle, start_angle, end_angle, color, thickness)
            while r1 < 50:
                r1 += 1
                r2 += 1


#         if ((xe1 - cxe1) ** 2) / (r1 ** 2) + ((ye1 - int(shape[1][1])) ** 2) / (r2 ** 2) <= 1 :
#             cv2.circle(frame, (xe1, ye1), 1, (e1b, e1g, e1r), -1)
#         if ((xe2 - cxe2) ** 2) / (r1 ** 2) + ((ye2 - int(shape[3][1])) ** 2) / (r2 ** 2) <= 1 :
#             cv2.circle(frame, (xe2, ye2), 1, (e2b, e2g, e2r), -1)
#         if ((xn - int(shape[4][0])) ** 2) / (r1 ** 2) + ((yn - int(shape[4][1])) ** 2) / (rn ** 2) <= 1 :
#             cv2.circle(frame, (xn, yn), 1, (nb, ng, nr), -1)

#     for (i, (x, y)) in enumerate(shape):
#         if i % 5 == 0:
#             for j in range(300):
#                 px = randint(-5, 5) + x
#                 py = randint(-5, 5) + y
#                 cv2.circle(frame, (px, py), 1, (255, 255, 255), -1)
#         if i % 5 == 1:
#             for j in range(300):
#                 px = randint(-5, 5) + x
#                 py = randint(-5, 5) + y
#                 cv2.circle(frame, (px, py), 1, (0, 0, 255), -1)
#
#         if i % 5 == 4:
#             for n in range(500):
#                 xn = randint(-100, 100) + x
#                 yn = randint(-100, 100) + y
#                 if (((xn - x) ** 2) / (r2 ** 2) + (yn - y) ** 2) / (r1 ** 2) <= 1 :
#                     cv2.circle(frame, (nosex, nosey), 1, (0, 0, 0), -1)
    return frame, leyepos, reyepos, nosepos

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-v", "--video", required=True,
    help="path to input video")
ap.add_argument("-o", "--output", type=str,
    help="path to output video")
ap.add_argument('-w', '--weights', default='./mmod_human_face_detector.dat',
                help='path to weights file')
ap.add_argument("-d", "--display", type=int, default=1,
                help="whether or not to display output frame to screen")
args = vars(ap.parse_args())
(stream, length, writer, fps) = vidinit(args)
# stream = cv2.VideoCapture(args["video"])
# length = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
# writer = None
# Find OpenCV version
# (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
# if int(major_ver)  < 3 :
#     fps = stream.get(cv2.cv.CV_CAP_PROP_FPS)
#     print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
# else :
#     fps = stream.get(cv2.CAP_PROP_FPS)
#     print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))vidinit(args)
points = []
leyepos = []
reyepos = []
nosepos = []
lcolor = []
rcolor = []
ncolor = []
#initialize detector, predictor, recognizer
#cnn
detector = dlib.cnn_face_detection_model_v1(args["weights"])
#hog
#detector = dlib.get_frontal_face_detector()
# detector = dlib.fhog_object_detector("bb0.svm")
# detector = dlib.simple_object_detector("b2detector.svm")
predictor = dlib.shape_predictor(args["shape_predictor"])
recognizer = dlib.face_recognition_model_v1("./dlib_face_recognition_resnet_model_v1.dat")
frame_number = 0
(grabbed, frame, frame_number) = frameincrement(frame_number)
print("video dims: {} x {}".format(frame.shape[0], frame.shape[1]))
vid_h = frame.shape[0]
vid_w = frame.shape[1]
# ratio = 1/(500/vid_w)
# print("1:1 ratio: {}".format(ratio))
while True:
# detect faces in the grayscale image
#     rects = detector(gray, 1)
    if not grabbed:
        break
    # load the input image, resize it, and convert it to grayscale
    # image = cv2.imread(args["image"])
    #     frame = imutils.resize(frame, width=500)
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_img, 1)
    faces = [face.rect for face in faces]
    if len(faces) == 0:
        print("no face detected, frame# {}".format(frame_number))
        (frame, key, writer) = dispwrite(writer, args, vid_h, vid_w, frame, fps)
        (grabbed, frame, frame_number) = frameincrement(frame_number)
    else:
        print("# of faces: {}" .format(len(faces)))
        for face in faces:
    #         x,y,w,h = face.left(), face.top(), face.right(), face.bottom()
    #         cv2.rectangle(frame, (x,y), (w,h), (255,200,150), 2, cv2.LINE_AA)
            (x1, y1, x2, y2) = face.left(), face.top(), face.right(), face.bottom()
            r = int((x2 - x1)/2)
            r1 = int(r * 1.2)
            r2 = int(r * 1)
            cx = int(x1 + r / .9)
            cy = int(y1 + r / 1.8)
    #         d = int(r * 1.414)
            center = (cx, cy)
            for spiral in range(10):
                r1 = r1 - spiral
                r2 = r2 - spiral
            axes= (r1, r2)
            rotation_angle = randint(-30, 30)
            start_angle = randint(0, 360)
            end_angle = randint(30, 360)
            color = (randint(20, 50), randint(50, 150), randint(150, 255))
            thickness = 1
            cv2.ellipse(frame, center, axes, rotation_angle, start_angle, end_angle, color, thickness)
            frame, leyepos, reyepos, nosepos = drawffeatures(rgb_img, face, leyepos, frame, reyepos, nosepos)
            
#             for p in range(100):
#                 shiftx = randint(-r1, r1) + cx
#                 shifty = randint(-r2, r2) + cy
#             #             dist = calculateDistance(cx, cy, shiftx, shifty)
#             #             if dist < r:
#             #                 cv2.circle(frame, (shiftx, shifty), 2, (0, 0, 255), -1)
#                 shiftxy = (shiftx, shifty)
#                 if len(points) > 2000:
#                     del points[(len(points) - 1)]
#                 points.append(shiftxy)
#             for point in points:
#                 print("point in face: {}" .format(point))
#                 if ((point[0] - cx) ** 2) / (r1 ** 2) + ((point[1] - cy) ** 2) / (r2 ** 2) <= 1 :
#                     thickness = randint(1, 3)
#                     red = randint(150, 255)
#                     green = randint(100, 150)
#                     blue = randint(50, 100)
#                     dots = cv2.circle(frame, (point[0], point[1]), thickness, (blue, green, red), -1)
#                     print("dots: {}".format(dots))
        (frame, key, writer) = dispwrite(writer, args, vid_h, vid_w, frame, fps)
        (grabbed, frame, frame_number) = frameincrement(frame_number)
        print("spray: {}, frame# {}".format(len(points), frame_number))
    # # save output image
    # cv2.imwrite("cnn_face_detection.png", image)
    
#         # if the video writer is None *AND* we are supposed to write
#         # the output video to disk initialize the writer
#         if writer is None and args["output"] is not None:
#             fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#             writer = cv2.VideoWriter(args["output"], fourcc, 24,
#                 (frame.shape[1], frame.shape[0]), True)
#     #             (open_cv_image.shape[1], open_cv_image.shape[0]), True)
#     # if the writer is not None, write the frame with recognized
#     # faces to disk
#         if writer is not None:
#             writer.write(frame)
#     # check to see if we are supposed to display the output frame to
#     # the screen
#         if args["display"] > 0:
#             cv2.imshow("Frame", frame)
#     #         cv2.imshow("Frame", open_cv_image)
#             cv2.waitKey(int(1000/fps))
#             key = cv2.waitKey(1) & 0xFF
#             # if the `q` key was pressed, break from the loop
#             if key == ord("q"):
#                 break

# # close all windows
cv2.destroyAllWindows()
stream.release();
# check to see if the video writer point needs to be released
if writer is not None:
    writer.release()
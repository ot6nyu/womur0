#Utilizing face_recognition example: facerec_from_video_file.py and find_faces_in_pictures among others.
########################################################################
#W-OUIB0 "withoutYou: i<=0 or Forever Dedicated to Rose(&Bud)
##################################################################
#--Train face_recognition to recognize bad bunnies in video clips.
#frameworkpython3 ./W-OUIB0-8.py --shape-predictor ../../dlib/python_examples/budz_predictor_landmarks.dat --video ./bbunny-clip0.mp4 --output ./W-OUIB0-8bb0.avi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from imutils import face_utils
from scipy.misc import imread
from PIL import Image, ImageDraw
import random
from random import randint
import face_recognition
import cv2
import sys
import dlib
import argparse

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

#####################################################################
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument('-w', '--weights', default='../../dlib/python_examples/mmod_human_face_detector.dat',
                help='path to weights file')
ap.add_argument("-v", "--video", required=True,
    help="path to input video")
ap.add_argument("-o", "--output", type=str,
    help="path to output video")
args = vars(ap.parse_args())
#--INITIALIZE
# detector = dlib.get_frontal_face_detector()
detector = dlib.cnn_face_detection_model_v1(args["weights"])
predictor = dlib.shape_predictor(args["shape_predictor"])
#####################################################################
stream, length, writer, fps = vidinit(args)

#FACE_RECOGNITION
front_image = face_recognition.load_image_file("Vincent_Gallo_0002.jpg")
# front_image = face_recognition.load_image_file(sys.argv[2]")
front_face_encoding = face_recognition.face_encodings(front_image)[0]
# side_image = face_recognition.load_image_file("vg-side.jpg")
# side_face_encoding = face_recognition.face_encodings(side_image)[0]
down_image = face_recognition.load_image_file("vg-side.jpg")
down_face_encoding = face_recognition.face_encodings(down_image)[0]

#--Create arrays of known face encodings and their names
# known_faces = [
known_face_encodings = [
    front_face_encoding,
#     side_face_encoding,
    down_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden"
]
#--Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0
process_this_frame = True
output_movie = None

while True:
#--Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1
#--Quit when the input video file ends
    if not ret:
        break
#--Resize frame of video to 1/4 size for faster face recognition processing
#     small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#--Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#     rgb_small_frame = small_frame[:, :, ::-1]
#     rgb_frame = frame[:, :, ::-1]

#################################################################
#     gray = cv2.cvtColor(rgb_small_frame, cv2.COLOR_BGR2GRAY)
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rects = detector(rgb_img, 1)
    rects = [rect.rect for rect in rects]
    for (i, rect) in enumerate(rects):
        
        shape = predictor(rgb_img, rect)
#         shape = face_utils.shape_to_np(shape)
#################################################################

#--Only process every other frame of video to save time
        if process_this_frame:
    #--Find all the faces and face encodings in the current frame of video
    #         face_locations = face_recognition.face_locations(rgb_frame)
    #         face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            face_locations = face_recognition.face_locations(rgb_img)
            face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
             
            face_names = []
            for face_encoding in face_encodings:
    #--See if the face is a match for the known face(s)
    #             match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                    
    ##-- If you had more than 2 faces, you could make this logic a lot prettier but I kept it simple for the demo
    #             name = None
    #             if match[0]:
    #                 name = "vg_frontal"
    #             elif match[1]:
    #                 name = "vg_down"
    #             face_names.append(name)
    
    #-- If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                face_names.append(name)
    
        process_this_frame = not process_this_frame
                    
    #-- create a temp image and a mask to work on
        tempFrame = frame.copy()
    #     maskShape = (frame.shape[0], frame.shape[1], 1)
    #     mask = np.full(maskShape, 0, dtype=np.uint8)
    #################################################################
    #-- convert dlib's rectangle to a OpenCV-style bounding box
        	# [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
    #     x *= 4
    #     y *= 4
    #     w *= 4
    #     h *= 4
    #
    #     cv2.rectangle(tempFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    #--show the face number
    #     	cv2.putText(tempFrame, "Face #{}".format(i + 1), (x - 10, y - 10),
    #     		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    #--loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
#         for (x, y) in shape:
# for face in faces:
# #         x,y,w,h = face.left(), face.top(), face.right(), face.bottom()
# #         cv2.rectangle(frame, (x,y), (w,h), (255,200,150), 2, cv2.LINE_AA)
#     (x1, y1, x2, y2) = face.left(), face.top(), face.right(), face.bottom()
#             x1 *= 4
#             y1 *= 4
#             x2 *= 4
#             y2 *= 4
#             cv2.circle(tempFrame, (x1, y1), 10, (0, 0, 255), -1)
#################################################################

#-- start the face loop
#-- Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue
#-- Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4.
        right *= 4.
        bottom *= 4.
        left *= 4.
        faceh = bottom - top
        facew = right - left
        xradius_flt = facew / 2.
        yradius_flt = faceh / 1.5
        xradius_int = int(xradius_flt)
        yradius_int = int(yradius_flt)
        center = int(left + (facew / 2)), int(top + (faceh / 2))
        
        eyf = (yradius_flt - (faceh/2))
        exf = (xradius_flt - (facew/2))
        etop = top - eyf
        ebottom = bottom + eyf
        eleft = left - exf
        eright = right + exf
        spray_size = 3000
        dot = 15
#-- Some test points
        x = np.random.uniform(eleft, eright, spray_size)
        y = np.random.uniform(etop, ebottom, spray_size)

#         x = np.random.rand(spray_size)*facew
#         y = np.random.rand(spray_size)*faceh
#--The ellipse
        angle = 0
        cos_angle = np.cos(np.radians(180.-angle))
        sin_angle = np.sin(np.radians(180.-angle))

        xc = x - center[0]
        yc = y - center[1]
        
        xct = xc * cos_angle - yc * sin_angle
        yct = xc * sin_angle + yc * cos_angle
        
#         rad_cc = (xct**2/(facew/2.)**2) + (yct**2/(faceh/2.)**2)
        rad_cc = (xct**2/(xradius_flt)**2) + (yct**2/(yradius_flt)**2)
        
        colors_array = []
        for r in rad_cc:
            if r <= 1.:
                # point in ellipse
                colors_array.append('green')
            else:
#--points not in ellipse
                colors_array.append('none')

#--find indices of dots within ellipse
        tf = np.isin(colors_array, 'green')
        ix = np.where(tf)
        
        for index in range(len(ix)):
            dx = (x[(ix[index])])
            dy = (y[(ix[index])])
            ddx = dx.astype(int)
            ddy = dy.astype(int)
            xy = list(zip(ddx, ddy))
#--blur first so that the circle is not blurred
#         tempFrame [top : top + faceh, left : left + facew] = cv2.blur(tempFrame [top : top + faceh, left : left + facew] ,(23,23)) #(7,7)
        for i in range(len(xy)):
    #             x = randint(left, right)
    #             y = randint(top, bottom)
            size = np.random.randint(dot)
    #             color = (randint(0, 255), randint(0, 255), randint(0,255))
            color = (randint(22, 64), randint(22, 180), randint(145, 165))
    #             print("color = {}".format(color))
#             cv2.circle(tempFrame, xy[i], int(size/2), color, -1)
            
#         cv2.ellipse(tempFrame,
#                     center=center,
#                     axes=(xradius_int, yradius_int),
#                     angle=0,
#                     startAngle=0,
#                     endAngle=360,
#                     color=(0, 255, 0),
#                     thickness=1)
#--alphablend
        opacity = random.uniform(0.4, 0.7)
        print("opacity = {}".format(opacity))
        cv2.addWeighted(tempFrame, opacity, frame, 1 - opacity, 0, frame)

#         cv2.ellipse(mask,
#                     center=center,
#                     axes=(xradius_int, yradius_int),
#                     angle=0,
#                     startAngle=0,
#                     endAngle=360,
#                     color=(255),
#                     thickness=-1)
#-- outide of the loop, apply the mask and save
#     mask_inv = cv2.bitwise_not(mask)
#     img1_bg = cv2.bitwise_and(frame, frame, mask = mask_inv)
#     img2_fg = cv2.bitwise_and(tempFrame, tempFrame, mask = mask)
#     dst = cv2.add(img1_bg, img2_fg)
#--Create an output movie file (make sure resolution/frame rate matches input video!)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output_movie = cv2.VideoWriter('W-OUIB08_.avi', fourcc, 25, (592,320))
    if output_movie == None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        output_movie = cv2.VideoWriter(args["output"], fourcc, fps, (frame.shape[1], frame.shape[0]), True)
#-- Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
#     output_movie.write(dst)
    output_movie.write(frame)
input_movie.release()
cv2.destroyAllWindows()
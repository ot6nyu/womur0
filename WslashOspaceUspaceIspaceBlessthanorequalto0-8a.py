# rose<3bud
#!rose=>bud
# python recognize_faces_video_file.py --encodings encodings.pickle --input videos/lunch_scene.mp4
# python recognize_faces_video_file.py --encodings encodings.pickle --input videos/lunch_scene.mp4 --output output/lunch_scene_output.avi --display 0
#frameworkpython3 WslashOspaceUspaceIspaceBlessthanorequalto0.py --encodings wouib0_encodings.pickle --input ../examples/bbunny-clip2.mp4 --output output/pckl1wouib0-18.mov --display 0

#frameworkpython3 WslashOspaceUspaceIspaceBlessthanorequalto0-5a.py --encodings wouib0_enc2.pickle --input ../examples/bbunny-clip2-1.mp4 --output output/pckl1wouib0-18.mov --display

# import the necessary packages
from PIL import Image, ImageDraw
from random import randint
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import numpy as np
import concurrent.futures

def prep_frame(loc_frame):
    # convert the input frame from BGR to RGB then resize it to have
    # a width of 750px (to speedup processing)
    loc_rgb = cv2.cvtColor(loc_frame, cv2.COLOR_BGR2RGB)
#     loc_rgb = imutils.resize(loc_frame, width=750)
    loc_r = loc_frame.shape[1] / float(loc_rgb.shape[1])
    return loc_rgb, loc_r

def face_loc_enc_marks(loc_rgb):
    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input frame, then compute
    # the facial embeddings for each face
    loc_boxes = face_recognition.face_locations(loc_rgb,
        model=args["detection_method"])
    loc_encodings = face_recognition.face_encodings(loc_rgb, loc_boxes)
    loc_landmarks = face_recognition.face_landmarks(loc_rgb)
    return loc_boxes, loc_encodings, loc_landmarks

def face_match(loc_encodings):
    # loop over the facial embeddings
    loc_names = []
    for encoding in loc_encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"],
            encoding)
        name = "Unknown"

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)
        
        # update the list of names
        loc_names.append(name)
    return loc_names

def face_graffiti(loc_boxes, loc_landmarks, loc_names, loc_frame, loc_r, loc_open_cv_image, loc_id_status):
#     loc_id_status = id_status
    # loop over the recognized faces
    for ((top, right, bottom, left), face_landmarks, name) in zip(loc_boxes, loc_landmarks, loc_names):
        if name != "buds":
            if loc_id_status >=1 and loc_id_status <= 110:
                pil_image = Image.fromarray(loc_frame)
                d = ImageDraw.Draw(pil_image)
                d.line(xy, fill=(45, 65, 125), width= w)
                print("d.line xy: ({}, {})" .format(x, y))
                loc_id_status += 1
                loc_open_cv_image = np.array(pil_image)
            else:
                loc_id_status = 0
                loc_open_cv_image = loc_frame
        elif name == "buds":
            # rescale the face coordinates
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)
            # Extract the region of the image that contains the face
#             face_image = loc_frame[top:bottom, left:right]
#
#             # Blur the face image
#             face_image = cv2.GaussianBlur(face_image, (99, 99), 30)
#
#             # Put the blurred face region back into the frame image
#             loc_frame[top:bottom, left:right] = face_image
#             # draw the predicted face name on the image
#             cv2.rectangle(frame, (left, top), (right, bottom),
#                 (0, 255, 0), 2)
#             y = top - 15 if top - 15 > 15 else top + 15
#             cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.75, (0, 255, 0), 2)
            
            # Create a PIL imagedraw object so we can draw on the picture
#             np.array(open_cv_image)
            pil_image = Image.fromarray(loc_frame)
            d = ImageDraw.Draw(pil_image)
            
            for facial_feature in face_landmarks.keys():
                print("facial_feature: {}" .format(facial_feature))
                xy=[]
                # Let's trace out each facial feature in the image with a line!
                for x, y in face_landmarks[facial_feature]:
#                     x *= r
#                     y *= r
                    x = (x + randint(-20, 20))*r
                    y = (y + randint(-20, 20))*r
                    w = randint(5, 30)
                    xy.append((x, y))
                print("xy: {}" .format(xy))
                d.line(xy, fill=(45, 65, 125), width= w)
#                 d.ellipse([x1, y1, x1, y2], fill=(45, 65, 125), outline=(125, 65, 20))
#                 print("d.line xy: ({}, {})" .format(x, y))
#                 print("d.ellipse xy: ({}, {}) , ({}, {})" .format(x1, y1, x2, y2))
            loc_id_status = 1
            loc_open_cv_image = np.array(pil_image)
    return loc_open_cv_image, loc_id_status

           
                #print("face_landmarks: {}" .format(face_landmarks))            print("face_landmarks[facial_feature]: {}" .format(face_landmarks[facial_feature]))
####################################
#             for face_landmarks in landmarks:
#                 for facial_feature in face_landmarks.keys():
#                     xy=[]
#                     # Let's trace out each facial feature in the image with a line!
# #                     (x, y) = face_landmarks[facial_feature]
#                     for x, y in face_landmarks[facial_feature]:
#                         x *= r
#                         y *= r
#                         xy.append((x, y))
#                     d.line(xy, width=5)
#                     print("d.line xy: {}" .format(xy))#           print("face_landmarks: {}" .format(face_landmarks))
####################################


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
    help="path to serialized db of facial encodings")
ap.add_argument("-i", "--input", required=True,
    help="path to input video")
ap.add_argument("-o", "--output", type=str,
    help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
    help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
    help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the pointer to the video file and the video writer
print("[INFO] processing video...")
stream = cv2.VideoCapture(args["input"])
length = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
writer = None
frame_number = 0
id_status = -1
# loop over frames from the video file stream

###################       MAIN LOOP      #######################

while True:
    # grab the next frame
    (grabbed, frame) = stream.read()
    frame_number += 1
    # if the frame was not grabbed, then we have reached the
    # end of the stream
    if not grabbed:
        break
    rgb, r = prep_frame(frame)
    boxes, encodings, landmarks = face_loc_enc_marks(rgb)
    names = face_match(encodings)
    open_cv_image = frame
    open_cv_image, id_status = face_graffiti(boxes, landmarks, names, frame, r, open_cv_image, id_status)

    # if the video writer is None *AND* we are supposed to write
    # the output video to disk initialize the writer
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 24,
#             (frame.shape[1], frame.shape[0]), True)
            (open_cv_image.shape[1], open_cv_image.shape[0]), True)
    # if the writer is not None, write the frame with recognized
    # faces to disk
    if writer is not None:
        writer.write(open_cv_image)
#         writer.write(frame)

    #-- Write the resulting image to the output video file
        print("Writing frame {} / {}".format(frame_number, length))

    # check to see if we are supposed to display the output frame to
    # the screen
    if args["display"] > 0:
#         cv2.imshow("Frame", frame)
        cv2.imshow("Frame", open_cv_image)

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# close the video file pointers
stream.release()

# check to see if the video writer point needs to be released
if writer is not None:
    writer.release()
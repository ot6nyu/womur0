# USAGE
# python recognize_faces_video_file.py --encodings encodings.pickle --input videos/lunch_scene.mp4
# python recognize_faces_video_file.py --encodings encodings.pickle --input videos/lunch_scene.mp4 --output output/lunch_scene_output.avi --display 0
#frameworkpython3 WslashOspaceUspaceIspaceBlessthanorequalto0.py --encodings wouib0_encodings.pickle --input ../examples/bbunny-clip2.mp4 --output output/pickle1wouib0- .avi --display 0

#frameworkpython3 WslashOspaceUspaceIspaceBlessthanorequalto0-1.py --encodings wouib0_encodings.pickle --input ../examples/bbunny-blip.mp4 --output output/pickle1wouib0-3.avi --display

# import the necessary packages
from PIL import Image, ImageDraw
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import numpy as np
import concurrent.futures

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

# loop over frames from the video file stream
while True:
    # grab the next frame
    (grabbed, frame) = stream.read()
    open_cv_image = frame
    frame_number += 1
    # if the frame was not grabbed, then we have reached the
    # end of the stream
    if not grabbed:
        break
    
    # convert the input frame from BGR to RGB then resize it to have
    # a width of 750px (to speedup processing)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input frame, then compute
    # the facial embeddings for each face
    boxes = face_recognition.face_locations(rgb,
        model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)
    landmarks = face_recognition.face_landmarks(rgb)
    names = []

    # loop over the facial embeddings
    for encoding in encodings:
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
        names.append(name)

    # loop over the recognized faces
#     for ((top, right, bottom, left), name) in zip(boxes, names):
    for ((top, right, bottom, left), face_landmarks, name) in zip(boxes, landmarks, names):
#         print("top: {}, right: {}, bottom: {}, left: {} is {}".format(top, right, bottom, left, name))
        for facial_feature in face_landmarks.keys():
#           print("face_landmarks: {}" .format(face_landmarks))
            print("face_landmarks[facial_feature]: {}" .format(face_landmarks[facial_feature]))
        
        if name == "vinnie_gal":

            # rescale the face coordinates
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)

#             # draw the predicted face name on the image
#             cv2.rectangle(frame, (left, top), (right, bottom),
#                 (0, 255, 0), 2)
#             y = top - 15 if top - 15 > 15 else top + 15
#             cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.75, (0, 255, 0), 2)
            # Create a PIL imagedraw object so we can draw on the picture
            pil_image = Image.fromarray(frame)
            d = ImageDraw.Draw(pil_image)

            for face_landmarks in landmarks:
                for facial_feature in face_landmarks.keys():
                    xy=[]
                    # Let's trace out each facial feature in the image with a line!
#                     (x, y) = face_landmarks[facial_feature]
                    for x, y in face_landmarks[facial_feature]:
                        x *= r
                        y *= r
                        xy.append((x, y))
                    d.line(xy, width=5)
                    print("d.line xy: {}" .format(xy))
            open_cv_image = np.array(pil_image)
        else:
            
            open_cv_image = frame
        
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
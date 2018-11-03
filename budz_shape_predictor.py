import cv2
import dlib

# if len(sys.argv) != 2:
#     print(
#         "Give the path to the examples/faces directory as the argument to this "
#         "program. For example, if you are in the python_examples folder then "
#         "execute this program by running:\n"
#         "    ./train_shape_predictor.py ../examples/faces")
#     exit()
# faces_folder = sys.argv[1]

predictor = dlib.shape_predictor("budz3_predictor_landmarks.dat")
# detector = dlib.get_frontal_face_detector()
detector = dlib.fhog_object_detector("b3adetector.svm")
# Now let's run the detector and shape_predictor over the images in the faces
frame_number = 0
stream = cv2.VideoCapture("../../face_recognition-master/examples/bbunny-clip2-1.mp4")

while True:
    (grabbed, frame) = stream.read()
    frame_number += 1
    # if the frame was not grabbed, then we have reached the
    # end of the stream
    if not grabbed:
        break
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(rgb_image, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        rect_to_bb(d)
        
        # Get the landmarks/parts for the face in box d.
        shape = predictor(rgb_image, d)
        for i, part in shape.part:
            print("part: {}, xy: {}".format(i, shape.part))
            
            
            
# close the video file pointers
stream.release()
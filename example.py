import dnn

a = dnn.DNN("mmod_human_face_detector.dat")
b = a.detect("./dlib/examples/faces/2009_004587.jpg")
print(b)


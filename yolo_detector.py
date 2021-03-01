import cv2
import time
import datetime

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("./models/coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

vc = cv2.VideoCapture("./inputs/rallying_ph_2_16_clip_0.mp4")

net = cv2.dnn.readNet("./models/yolov4-tiny.weights",
                      "./models/yolov4-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

fullVidWriter = cv2.VideoWriter(
    './out/rallying_2_16_clip_0_yolo.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    30,
    (round(vc.get(cv2.CAP_PROP_FRAME_WIDTH)),
     round(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
)
t1 = datetime.datetime.now()
print("starting inference {}".format(t1))

while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        break

    start = time.time()
    classes, scores, boxes = model.detect(
        frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()

    start_drawing = time.time()
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid[0]], score)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    end_drawing = time.time()

    fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (
        1 / (end - start), (end_drawing - start_drawing) * 1000)
    cv2.putText(frame, fps_label, (0, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    # cv2.imshow("detections", frame)
    fullVidWriter.write(frame)

t2 = datetime.datetime.now()
print("end inference {}".format(t1))
diff = t2-t1
print('diff {} seconds'.format(diff.seconds))
fullVidWriter.release()

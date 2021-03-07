import cv2
import time
import datetime

# rallying_2_16_clip_0

CONFIDENCE_THRESHOLD = 0.1
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("./models/coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

vc = cv2.VideoCapture("./inputs/rallying_2_16_clip_0.mp4")

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

while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        break

    classes, scores, boxes = model.detect(
        frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid[0]], score)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    fullVidWriter.write(frame)

diff = datetime.datetime.now()-t1
print('processing time {} seconds'.format(diff.seconds))
fullVidWriter.release()

# USAGE
# python client.pyx --server-ip SERVER_IP
from imutils import build_montages
import cv2
import zmq
import json
import numpy as np
import base64
import time
from utils.torch_utils import time_synchronized


context = zmq.Context()
receiver = context.socket(zmq.REP)
receiver.bind("tcp://*:5563")
print('client connected')
ms_count = 0
mean_count = 0
i = 0
frameDict = {}
while True:
    ms_count += 1
    t1 = time_synchronized()
    message = receiver.recv_string()
    t2 = time_synchronized()
    infer_time = t2 - t1
    mean_count = infer_time + mean_count
    avg_ms = str(mean_count / ms_count)
    # print(str(infer_time)+'\r')
    bbox = json.loads(message)
    rpiName = str(bbox['t_id'])
    fps = str(bbox['fps'])
    lat = str(bbox['lat'])

    # print(bbox['bbox'])
    if bbox['bbox'] is None:
        rect = []
    else:
        rect = list(bbox['bbox'])

    nparr = np.frombuffer(base64.decodebytes(bbox['img'].encode()), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # print(img.shape)

    frame = cv2.resize(frame, (300, 300))
    if frame is None:
        break
    cv2.putText(frame, "FPS: " + fps, (15, 20), cv2.FONT_HERSHEY_COMPLEX,
                0.4, (255, 0, 0), 1)
    cv2.putText(frame, 'infer latency: ' + lat, (15, 40), cv2.FONT_HERSHEY_COMPLEX,
                0.4, (255, 0, 0), 1)
    cv2.putText(frame, 'network latency: ' + str(infer_time), (15, 60), cv2.FONT_HERSHEY_COMPLEX,
                0.4, (255, 0, 0), 1)
    cv2.putText(frame, 'video channel id: ' + rpiName, (15, 80), cv2.FONT_HERSHEY_COMPLEX,
                0.4, (255, 0, 0), 1)

    frameDict[rpiName] = frame

    # build a montage using images in the frame dictionary
    montages = build_montages(
        frameDict.values(), (frame.shape[1], frame.shape[0]), (2, 2))

    # display the montage(s) on the screen
    for (i, montage) in enumerate(montages):
        cv2.imshow("pre-trained model",
                   montage)
    key = cv2.waitKey(1)
    if key == ord('q'):
        receiver.send_string(json.dumps({'state': 'br'}))

        cv2.destroyAllWindows()
        break
    else:
        receiver.send_string(json.dumps({'state': 'ok'}))

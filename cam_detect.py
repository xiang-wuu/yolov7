import argparse
from models.experimental import attempt_load
from utils.datasets import letterbox
import torch.multiprocessing as mp
import zmq
import base64
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import *
from torch.backends import cudnn
import json
from os import walk
import cv2
import numpy as np
import random


def update_fps(prev_time, fps, actual_fps):
    curr_time = time.time()
    if abs(curr_time - prev_time) <= 1:
        fps = fps + 1
    else:
        actual_fps = fps
        prev_time = curr_time
        fps = 0
    return prev_time, fps, actual_fps


def main_fun(idx, gpu_id):
    f_path = '/home/swap/Videos/'
    ls = ['Kendrick_Lamar_HUMBLE_hd720.mp4',
          '4K-camera.mp4',
          'day_time.mkv', 'day_time.mkv',
          'person1.mp4', '103.mp4']
          
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5563")

    print('gpu_id: ', gpu_id)
    device = select_device(device=str(gpu_id))
    model = attempt_load('/home/swap/Downloads/yolov7.pt', map_location=device)
    cudnn.enabled = True
    cudnn.fastest = True
    cudnn.deterministic = False
    # cudnn.benchmark = True
    model.share_memory()
    print(is_parallel(model))

    torch.no_grad()
    model.to(device).eval()
    half = False  # half precision only supported on CUDA
    if half:
        model.half()

    img_size = 608

    cam = cv2.VideoCapture(f_path + ls[2])

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(names))]
    ms_count = 0
    mean_count = 0
    avg_ms = 0
    # Run inference
    t0 = time.time()
    prev_time = time.time()
    j_count = 0
    fps = 0
    actual_fps = 0
    while True:
        prev_time, fps, actual_fps = update_fps(prev_time, fps, actual_fps)
        _, img = cam.read()
        if img is None:
            break
        im0 = img.copy()
        img = letterbox(img, img_size, stride=32, auto=False and True)[0]
        img = np.stack(img, 0)

        img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3x416x416

        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)

        if half:
            img = img.half()
        else:
            img = img.float()  # if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False, profile=False)[0]

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(
            pred, 0.1, 0.5, classes=None, agnostic=False)
        t2 = time_synchronized()
        # Process detections
        ms_count += 1
        ls_box = []

        for i, det in enumerate(pred):  # detections per image
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    # and int(c) != 2 and int(c) != 5 and int(c) != 7:
                    if int(c) != 0:
                        continue
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if int(cls) != 0 and int(cls) != 2 and int(cls) != 24 and int(cls) != 7 and int(cls) != 28:
                        continue
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label,
                                 color=colors[int(cls)])
                    ls_box.append(
                        (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
            if len(ls_box) > 0:
                j_count += 1

            # Print time (inference + NMS)
            infer_time = t2 - t1
            mean_count = infer_time + mean_count
            avg_ms = mean_count / ms_count
            print('p_id: %s %sDone. (%.3fs) FPS: %s' %
                  (idx, s, avg_ms, str(actual_fps)))

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
        retval, buffer = cv2.imencode('.jpg', im0, encode_param)
        image = base64.b64encode(buffer).decode('utf-8')
        # print(image)
        data = {'t_id': str(idx), "img": image, "lat": avg_ms, "fps": actual_fps,
                "bbox": ls_box}  # , 'fname': str(f_name)}
        data = json.dumps(data)
        socket.send_string(data)

        # Get the reply.
        _ = socket.recv()

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, default=1, help='*.cfg path')
    parser.add_argument('--gid', type=int, default=0, help='*.cfg path')
    parser.add_argument('--local_rank', type=int, default=0, help='*.cfg path')

    opt = parser.parse_args()
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    print(opt)

    ts = []
    with torch.no_grad():
        for i in range(opt.pid):
            # t = Test(i)
            t = mp.Process(target=main_fun, args=(i, opt.gid))
            t.start()
            ts.append(t)
        for tt in ts:
            tt.join()
        # with torch.no_grad():
        # detect()

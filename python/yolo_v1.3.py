''' 采用高斯混合模型版本'''
import numpy as np
import cv2
from rknn.api import RKNN
import torchvision.models as models
import torch
from PIL import Image
import time
import copy
import sys
import pdb

_CLASS = ["person", "bicycle", "car", "motorbike", "bus", "truck"]

_GRAY = (218, 227, 218)
_GREEN = (0, 255, 0)
_BLACK = (0, 0, 0)
_RED = (0, 0, 255)
SIZE_W, SIZE_H = [416, 256]  ## correct_this
video_w = 416
video_h = 256
OBJ_THRESH = 0.55
NMS_THRESH = 0.2
id_reset_thres = 10000  ## correct_this
frame_per_second = 5  # 每秒帧数 ## correct_this
debugger = True

TAG = "pr_3"
# MODEL = './checkpoints/onnx/yolo_' + TAG + '.onnx'
MODEL = 'checkpoints/yolov3.weights'
RKNN_MODEL_PATH = 'checkpoints/rknn/yolov3.rknn'
# VIDEO_PATH = "/home/labasus/DataDisk/luqiao/yolo/videos/XLT-45杏林大桥85#灯杆出岛停车.mp4"
SAVE_PATH = '/home/labasus/DataDisk/luqiao/yolo/output_video/'
PIC_PATH = "./video/1.png"
NEED_BUILD_MODEL = False
NEED_RUN_MODEL = True
NPU = True

NEED_RUN_PIC = False
NEED_RUN_VIDEO = True


#############
# Traffic
#############
## correct_this
def return_event(EventType=0):
    VidiconNumber = ''
    EventDate = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    EventAddress = ''
    EventImg = ''
    EventDetector = ''

    event = {'VidiconNumber': VidiconNumber,
             'EventDate': EventDate,
             'EventAddress': EventAddress,
             'EventType': EventType,
             'EventImg': EventImg,
             'EventDetector': EventDetector}
    return event


class Detector(object):
    def __init__(self):
        ## correct_this
        self.event_track_id = []
        self.event_type_id = []

        self.car_conf_thres = 0.55
        self.frame_info_list = []  # key: track_id, value: (center_x, center_y, x1, y1)
        self.frame_info_list_maxsize = frame_per_second * 10  # 保存帧数量, 250  ## correct_this
        self.video_name = VIDEO_PATH[VIDEO_PATH.rfind('/') + 1:].split('.')[0]

        # person
        self.person_switch = True
        self.person_conf_thres = 0.8  ## correct_this

        # jam && park
        self.park_box_area_min = 6000
        self.jam_box_area_min = 4000

        # park
        self.park_time_thres = frame_per_second * 2
        self.park_switch = True
        self.show_park = True
        self.park_pos_thres = 0.1  # 位移距离因子
        # self.parking_time_thres = frame_per_second * 2  # 50 ## correct_this
        # self.parking_pos_dif_max = 18
        # self.parking_pos_dif_min = 0.001  # 0.5
        # self.parking_pos_dif_mid = 8
        # self.parking_area_max = 10000
        self.park_flag = {}
        self.park_count = {}
        self.park_count_init = frame_per_second * 2
        # jam
        self.jam_time_thres = frame_per_second * 2
        self.jam_switch = True
        self.jam_flag = False
        self.pre_jam_flag = False
        self.jam_frame_count_init = frame_per_second * 8
        self.jam_pos_thres = 1.3  # 车身距离
        self.jam_frame_count = 0
        self.jam_thres_init = 5

        # cross
        self.cross_switch = True
        self.crossing_pos_dif_thres = 16
        self.crossing_pos_x_thres = 16
        # retrograde
        self.retrograde_switch = True
        self.retrograde_thres = 2
        self.id_frame_info = {}

        self.video_dict = {
            'XAT-36翔安隧道出岛150#灭火器违章变道': [[(656, 1080), (1038, 0)], [(1335, 1080), (1038, 0)]],
            'XAT-107翔安隧道进岛120#违章变道': [[(588, 1080), (951, 62)], [(951, 62), (1050, 0)], [(1270, 1080), (1005, 116)]],
            'XAT-108翔安隧道进岛115#违章变道': [[(554, 1080), (855, 143)], [(855, 143), (885, 90)], [(885, 90), (921, 52)],
                                      [(1235, 1080), (1096, 637)]],
            'XAT-113翔安隧道进岛90#灭火器违章变道': [[(600, 1080), (893, 208)], [(893, 208), (974, 75)], [(974, 75), (975, 36)],
                                        [(1285, 1080), (1020, 150)], [(1020, 150), (1025, 66)],
                                        [(1025, 66), (1008, 35)]],
            'XAT-99翔安隧道进岛160违章变道': [[(715, 1080), (1120, 0)], [(1400, 1080), (1122, 0)]],
            'XAQ-51翔安隧道190#事故37秒': [[(1420, 1080), (910, 0)]],
        }
        self.retrograde_dict = {
            'JMT-32集美大桥2#逆向': {1: {'left': [[(0, 794), (617, 277)], [(617, 277), (759, 0)]],
                                   'right': [[(1373, 1080), (1351, 566)], [(1351, 566), (1185, 240)],
                                             [(1185, 240), (1014, 0)]]}, -1: {}},
            '集美大桥4_JMQ-18集美大桥进岛方向隧道内LK2213_20200721051600_20200721051922_': {
                1: {'left': [[(0, 917), (1192, 563)]], 'right': [[(1115, 1080), (1288, 574)]]}, -1: {}},
            '集美大桥4_JMQ-19集美大桥进岛方向隧道内LK2113_20200721051310_20200721051440_': {
                1: {'left': [[(0, 520), (1490, 171)]],
                    'right': [[(1160, 1080), (1540, 250)], [(1185, 240), (1014, 0)]]},
                -1: {}},
            '集美大桥4_JMQ-20集美大桥进岛方向隧道内LK1813_20200721050830_20200721050955_': {
                1: {'left': [[(0, 773), (1405, 387)], [(1405, 387), (1761, 320)]],
                    'right': [[(1424, 1012), (1759, 396)]]},
                -1: {}},
        }
        self.region = [0, 0, 0, 0]
        self.exist_lane = True if self.video_name in self.video_dict.keys() else False
        self.exist_retrograde = True if self.video_name in self.retrograde_dict.keys() else False
        if self.exist_lane:
            self.k_b_list = []
            for lane in self.video_dict[self.video_name]:
                (p1_x, p1_y), (p2_x, p2_y) = lane
                k = (p2_y - p1_y) / (p2_x - p1_x)
                b = p1_y - p1_x * k
                self.k_b_list.append((k, b))

        if self.exist_retrograde:
            self.retrograde_k_b_list = {1: {'left': [], 'right': []}, -1: {'left': [], 'right': []}}
            for (derection, dict) in self.retrograde_dict[self.video_name].items():
                for (d, v) in dict.items():
                    for lane in v:
                        (p1_x, p1_y), (p2_x, p2_y) = lane
                        k = (p2_y - p1_y) / (p2_x - p1_x)
                        b = p1_y - p1_x * k
                        self.retrograde_k_b_list[derection][d].append((k, b))
        self.cross_ids = []
        self.retrograde_ids = []
        self.cross_ids_maxsize = 10
        self.retrograde_ids_maxsize = 10

    # def check_jam(self, frame_info_p1, frame_info_p2):
    #     bbox_area = (frame_info_p1[2] - frame_info_p1[0]) * (frame_info_p1[3] - frame_info_p1[1])
    #
    #     jam_pos_thres = self.pos_thres * max(frame_info_p1[2]-frame_info_p1[0],frame_info_p1[3]-frame_info_p1[1])
    #     return ((frame_info_p1[0] - frame_info_p2[0]) ** 2 + (frame_info_p1[1] - frame_info_p2[1]) ** 2) ** 0.5 <= jam_pos_thres

    def check_park_jam(self, frame_info_p1, frame_info_p2):
        bbox_area = (frame_info_p1[2] - frame_info_p1[0]) * (frame_info_p1[3] - frame_info_p1[1])
        park_jam = []
        if bbox_area < self.park_box_area_min:
            park_jam.append(False)
        else:
            park_dif_thres = self.park_pos_thres * min(frame_info_p1[2] - frame_info_p1[0],
                                                       frame_info_p1[3] - frame_info_p1[1])
            park_jam.append(((frame_info_p1[0] - frame_info_p2[0]) ** 2 +
                             (frame_info_p1[1] - frame_info_p2[1]) ** 2) ** 0.5 <= park_dif_thres)

        if bbox_area < self.jam_box_area_min:
            park_jam.append(False)
        else:
            jam_dif_thres = self.jam_pos_thres * min(frame_info_p1[2] - frame_info_p1[0],
                                                 frame_info_p1[3] - frame_info_p1[1])

            park_jam.append(((frame_info_p1[0] - frame_info_p2[0]) ** 2 +
                             (frame_info_p1[1] - frame_info_p2[1]) ** 2) ** 0.5 <= jam_dif_thres)
        return park_jam

    def check_retrograde(self, frame_info_p1, frame_info_p2):
        f = 0
        p1_x = (frame_info_p1[0] + frame_info_p1[2]) / 2
        p1_y = (frame_info_p1[1] + frame_info_p1[3]) / 2
        p2_x = (frame_info_p2[0] + frame_info_p2[2]) / 2
        p2_y = (frame_info_p2[1] + frame_info_p2[3]) / 2
        if abs(p2_y - p1_y) >= self.retrograde_thres:
            derector = -1 if p2_y - p1_y >= 0 else 1
            for i in range(len(self.retrograde_k_b_list[derector]['left'])):
                if p1_x >= ((p1_y - self.retrograde_k_b_list[derector]['left'][i][1]) /
                            self.retrograde_k_b_list[derector]['left'][i][0]):
                    f = 1
                    break
            if f == 1:
                for i in range(len(self.retrograde_k_b_list[derector]['right'])):
                    if p2_x <= ((p2_y - self.retrograde_k_b_list[derector]['right'][i][1]) /
                                self.retrograde_k_b_list[derector]['right'][i][0]):
                        return True
        return False

    def check_crossing(self, frame_info_p1):
        p_x = (frame_info_p1[0] + frame_info_p1[2]) / 2
        p_y = (frame_info_p1[1] + frame_info_p1[3]) / 2
        for i in range(len(self.k_b_list)):
            if p_y <= self.video_dict[self.video_name][i][0][1] and p_y >= self.video_dict[self.video_name][i][1][1] and \
                    abs((p_y - self.k_b_list[i][1]) / self.k_b_list[i][0] - p_x) <= self.crossing_pos_x_thres - (
                    video_h - p_y) * 0.01:
                return True
        return False

    def check_box_in_region(self, sp_box):
        p_x = (sp_box[0] + sp_box[2]) / 2
        p_y = (sp_box[1] + sp_box[3]) / 2
        if p_x < self.region[0]:
            return False
        if p_x > self.region[2]:
            return False
        if p_y < self.region[1]:
            return False
        if p_y > self.region[3]:
            return False
        return True

    def isRayIntersectsSegment(poi, s_poi, e_poi):  # [x,y] [lng,lat]
        # 输入：判断点，边起点，边终点，都是[lng,lat]格式数组
        if s_poi[1] == e_poi[1]:  # 排除与射线平行、重合，线段首尾端点重合的情况
            return False
        if s_poi[1] > poi[1] and e_poi[1] > poi[1]:  # 线段在射线上边
            return False
        if s_poi[1] < poi[1] and e_poi[1] < poi[1]:  # 线段在射线下边
            return False
        if s_poi[1] == poi[1] and e_poi[1] > poi[1]:  # 交点为下端点，对应spoint
            return False
        if e_poi[1] == poi[1] and s_poi[1] > poi[1]:  # 交点为下端点，对应epoint
            return False
        if s_poi[0] < poi[0] and e_poi[1] < poi[1]:  # 线段在射线左边
            return False

        xseg = e_poi[0] - (e_poi[0] - s_poi[0]) * (e_poi[1] - poi[1]) / (e_poi[1] - s_poi[1])  # 求交
        if xseg < poi[0]:  # 交点在射线起点的左侧
            return False
        return True  # 排除上述情况之后

    def check_box_in_region(self, poi):
        # 输入：点，多边形三维数组
        # poly=[[[x1,y1],[x2,y2],……,[xn,yn],[x1,y1]],[[w1,t1],……[wk,tk]]] 三维数组
        poly = [[[self.region[0], self.region[1]],
                 [self.region[0], self.region[3]],
                 [self.region[2], self.region[1]],
                 [self.region[2], self.region[3]]]]
        # 可以先判断点是否在外包矩形内
        # if not isPoiWithinBox(poi,mbr=[[0,0],[180,90]]): return False
        # 但算最小外包矩形本身需要循环边，会造成开销，本处略去
        sinsc = 0  # 交点个数
        for epoly in poly:  # 循环每条边的曲线->each polygon 是二维数组[[x1,y1],…[xn,yn]]
            for i in range(len(epoly) - 1):  # [0,len-1]
                s_poi = epoly[i]
                e_poi = epoly[i + 1]
                if isRayIntersectsSegment(poi, s_poi, e_poi):
                    sinsc += 1  # 有交点就加1

        return True if sinsc % 2 == 1 else False

    def show_results(self, image, res, spi_boxs):
        cur_frame_info = {}

        # jam
        jam_count = 0
        self.pre_jam_flag = self.jam_flag  ## correct_this

        for result in res:
            box, score, cls, tracking_id = result['bbox'], result['score'], result['class'], result['tracking_id']
            top, left, bottom, right = box
            bbox = [left, top, right, bottom]



            # person
            ##########
            if cls in [0, 1]:
                if score >= self.person_conf_thres:
                    add_accident_bbox(img=image, bbox=bbox, track_id=tracking_id, accident='person')
                    if not tracking_id in self.event_track_id and self.person_switch:
                        self.event_track_id.append(tracking_id)
                        self.event_type_id.append(3)
                continue
            ##########
            if cls in [2, 3, 5, 7] and score < self.car_conf_thres:
                continue

            cur_frame_info[tracking_id] = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

            flag = False


            # jam & park
            ###################
            park_no_jump = True
            jam_no_jump = True
            if not tracking_id in self.park_flag:
                self.park_flag[tracking_id] = False
            if len(self.frame_info_list) >= max(self.park_time_thres, self.jam_time_thres):
                for i in range(len(self.frame_info_list)):
                    if tracking_id in self.frame_info_list[i]:
                        park, jam = self.check_park_jam(cur_frame_info[tracking_id],
                                                        self.frame_info_list[i][tracking_id])
                        if i < len(self.frame_info_list) - self.park_time_thres and park_no_jump and park:
                            self.park_flag[tracking_id] = True
                            park_no_jump = False
                            if tracking_id not in self.park_count.keys():
                                self.park_count[tracking_id] = self.park_count_init
                        if i < len(self.frame_info_list) - self.jam_time_thres and jam_no_jump and jam:
                            jam_count += 1
                            jam_no_jump = False
                        if not park_no_jump and not jam_no_jump:
                            break

            if self.park_flag[tracking_id] and self.show_park:
                add_accident_bbox(img=image, bbox=bbox, track_id=tracking_id, accident='park')
                flag = True
                if tracking_id not in self.event_track_id and self.park_switch:
                    self.event_track_id.append(tracking_id)
                    self.event_type_id.append(1)

            ####################
            # retrograde
            ####################
            # print(self.retrograde_ids)
            if self.exist_retrograde and (tracking_id in self.retrograde_ids or (tracking_id in self.id_frame_info and \
                                                                                 self.check_retrograde(
                                                                                     cur_frame_info[tracking_id],
                                                                                     self.id_frame_info[tracking_id]))):
                add_accident_bbox(img=image, bbox=bbox, track_id=tracking_id, accident='retrograde')
                flag = True
                ## correct this
                if tracking_id not in self.event_track_id and self.retrograde_switch:
                    self.event_track__id.append(tracking_id)
                    self.event_type_id.append(2)

                if not tracking_id in self.retrograde_ids:
                    self.retrograde_ids.append(tracking_id)
                    # if len(self.retrograde_ids) > self.retrograde_ids_maxsize:
                    #   self.retrograde_ids.pop(0)
            #####################
            # crossing
            #####################
            if self.exist_lane and (self.check_crossing(bbox) or tracking_id in self.cross_ids):
                add_accident_bbox(img=image, bbox=bbox, track_id=tracking_id, accident='cross')
                flag = True
                ## correct this
                if tracking_id not in self.event_track_id and self.cross_switch:
                    self.event_track_id.append(tracking_id)
                    self.event_type_id.append(6)

                if not tracking_id in self.cross_ids:
                    self.cross_ids.append(tracking_id)
                    if len(self.cross_ids) > self.cross_ids_maxsize:
                        self.cross_ids.pop(0)
            #####################
            if not flag:
                add_accident_bbox(img=image, bbox=bbox, track_id=tracking_id)

            if tracking_id not in self.id_frame_info:
                self.id_frame_info[tracking_id] = cur_frame_info[tracking_id]

        # looooooooook
        if len(self.frame_info_list) >= self.frame_info_list_maxsize:
            self.frame_info_list.pop(0)
        self.frame_info_list.append(cur_frame_info)

        ## park
        trk_id = [k for k, v in self.park_count.items() if v > 0]
        for tr_id in trk_id:
            self.park_count[tr_id] -= 1
        trk_id = [k for k, v in self.park_count.items() if v == 0]
        for tr_id in trk_id:
            self.park_flag[tr_id] = False

        ## jam
        if debugger:  ## correct_this
            print('jam_count:{}'.format(jam_count))
        if jam_count >= self.jam_thres_init:
            add_accident_bbox(img=image, accident='jam')
            self.jam_flag = True  ## correct_this
            self.jam_frame_count = self.jam_frame_count_init
        if self.jam_frame_count > 0:
            self.show_park = False
            add_accident_bbox(img=image, accident='jam')
            self.jam_flag = True
            self.jam_frame_count = self.jam_frame_count - 1
        else:
            self.show_park = True
            self.jam_flag = False

        if not self.pre_jam_flag and self.jam_flag and self.jam_switch:
            self.event_type_id.append(5)

        # spill
        ######
        for s_box in spi_boxs:
            if self.check_box_in_region(s_box):
                add_accident_bbox(img, s_box, 0, 'spill')

    def reset(self):
        self.event_track_id = []
        self.event_type_id = []
        self.park_count = {key: value for key, value in self.park_count.items() if value != 0}
        self.park_flag = {key: value for key, value in self.park_flag.items() if value}


def add_accident_bbox(img, bbox=[0, 0, 0, 0], track_id=0, accident='no'):
    bbox = np.array(bbox, dtype=np.int32)
    show_txt = True
    thickness = 3
    fontsize = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    flag = True
    jam_flag = False
    if debugger:
        if accident == 'person':
            txt = '{}.person'.format(track_id)
        elif accident == 'park':
            txt = '{}.park'.format(track_id)
        elif accident == 'cross':
            txt = '{}.cross'.format(track_id)
        elif accident == 'retrograde':
            txt = '{}.retrograde'.format(track_id)
        elif accident == 'spill':
            txt = '{}.spill'.format(track_id)
        elif accident == 'jam':
            jam_flag = True
            txt = 'Jam'
        else:
            flag = False
            txt = '{}'.format(track_id)
    else:
        if accident == 'person':
            txt = 'person'
        elif accident == 'park':
            txt = 'park'
        elif accident == 'cross':
            txt = 'cross'
        elif accident == 'retrograde':
            txt = 'retrograde'
        elif accident == 'spill':
            txt = 'spill'
        elif accident == 'jam':
            jam_flag = True
            txt = 'Jam'
        else:
            show_txt = False

    if jam_flag:
        cv2.putText(img, txt, (30, 70),
                    font, 3, _RED, thickness=2, lineType=cv2.LINE_AA)
    elif show_txt:
        cat_size = cv2.getTextSize(txt, font, fontsize, thickness)[0]
        if flag:
            cv2.rectangle(
                img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                _RED, thickness)
            cv2.rectangle(img,
                          (bbox[0], bbox[1] - cat_size[1] - thickness),
                          (bbox[0] + cat_size[0], bbox[1]), _RED, -1)
        else:
            cv2.rectangle(
                img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                _GREEN, thickness)
            cv2.rectangle(img,
                          (bbox[0], bbox[1] - cat_size[1] - thickness),
                          (bbox[0] + cat_size[0], bbox[1]), _GREEN, -1)
        cv2.putText(img, txt, (bbox[0], bbox[1] - thickness - 1),
                    font, fontsize, _BLACK, thickness=1, lineType=cv2.LINE_AA)
    else:
        cv2.rectangle(
            img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
            _GREEN, thickness)


class Tracker(object):
    def __init__(self):
        self.id_count = 0
        self.tracks = []

    def step(self, results):
        N = len(results)
        M = len(self.tracks)
        # pdb.set_trace()
        dets = np.array([det['ct'] for det in results], np.float32)  # N x 2
        track_size = np.array([((track['bbox'][2] - track['bbox'][0]) * \
                                (track['bbox'][3] - track['bbox'][1])) for track in self.tracks], np.float32)  # M
        item_size = np.array([((item['bbox'][2] - item['bbox'][0]) * \
                               (item['bbox'][3] - item['bbox'][1])) for item in results], np.float32)  # N
        tracks = np.array([pre_det['ct'] for pre_det in self.tracks], np.float32)  # M x 2
        dist = (((tracks.reshape(1, -1, 2) - dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M

        invalid = ((dist > track_size.reshape(1, M)) + (dist > item_size.reshape(N, 1))) > 0
        dist = dist + invalid * 1e18

        matched_indices = self.greedy_assignment(copy.deepcopy(dist))
        unmatched_dets = [d for d in range(dets.shape[0]) if not (d in matched_indices[:, 0])]

        matches = matched_indices

        ret = []
        for m in matches:
            track = results[m[0]]
            track['tracking_id'] = self.tracks[m[1]]['tracking_id']
            ret.append(track)

        # Private detection: create tracks for all un-matched detections
        for i in unmatched_dets:
            track = results[i]
            if track['score'] > OBJ_THRESH:
                self.id_count += 1
                ## correct_this
                if self.id_count >= id_reset_thres:
                    self.id_count = 0
                    detector.reset()

                track['tracking_id'] = self.id_count
                ret.append(track)
        print('id_count: ', self.id_count)
        self.tracks = ret
        return ret

    def greedy_assignment(self, dist):
        matched_indices = []
        if dist.shape[1] == 0:
            return np.array(matched_indices, np.int32).reshape(-1, 2)
        for i in range(dist.shape[0]):
            j = dist[i].argmin()
            if dist[i][j] < 1e16:
                dist[:, j] = 1e18
                matched_indices.append([i, j])
        return np.array(matched_indices, np.int32).reshape(-1, 2)


def nms_boxes(boxes, scores, nms_thresh):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def output_post_process(input_data, obj_thresh, nms_thresh):
    masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119],
               [116, 90], [156, 198], [373, 326]]
    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s, obj_thresh)  # OBJ_THRESH
        # print(b)
        boxes.append(b)
        classes.append(c)
        scores.append(s)
    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        if c in [2, 4, 5]:
            inds = np.where((classes == 2) | (classes == 4) | (classes == 5))
        else:
            inds = np.where(classes == c)

        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s, nms_thresh)
        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])
    if not nclasses and not nscores:
        # return None, None, None
        return None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    ans = np.concatenate((boxes, classes.reshape(-1, 1), scores.reshape(-1, 1)), 1)
    ans = np.array(list(set([tuple(t) for t in ans])))

    return ans


def process(input, mask, anchors):
    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])
    box_wh = np.exp(input[..., 2:4])
    box_wh = box_wh * anchors

    col = np.tile(np.arange(0, grid_w), grid_h).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= (SIZE_W, SIZE_H)
    box_xy -= (box_wh / 2.)
    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def filter_boxes(boxes, box_confidences, box_class_probs, obj_thresh):
    """Filter boxes with object threshold.

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= obj_thresh)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores

def if_interact(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h
    if area > 0:
        return True
    return False

if __name__ == "__main__":

    VIDEO_PATH = sys.argv[1]
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    rknn.config(reorder_channel='0 1 2', channel_mean_value='0 0 0 255')
    print('done')

    if NEED_BUILD_MODEL:
        # Load pytorch model
        print('--> Loading model {}'.format(MODEL))
        ret = rknn.load_darknet(model='/home/labasus/DataDisk/luqiao/deep_sort_yolov3/yolov3.cfg', weight=MODEL)
        # ret = rknn.load_onnx(model=MODEL)
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        print('done')
        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
        if ret != 0:
            print('Build onnx failed!')
            exit(ret)
        print('done')

        # Export rknn model
        print('--> Export RKNN model {}'.format(RKNN_MODEL_PATH))
        ret = rknn.export_rknn(RKNN_MODEL_PATH)
        if ret != 0:
            print('Export RKNN model failed!')
            exit(ret)
        print('done')
    else:
        # Direct load rknn model
        print('Loading RKNN model {}'.format(RKNN_MODEL_PATH))
        ret = rknn.load_rknn(RKNN_MODEL_PATH)
        if ret != 0:
            print('load rknn model failed.')
            exit(ret)
        print('done')

    if not NEED_RUN_MODEL:
        rknn.release()
        exit(0)

    a = time.time()
    print('--> init runtime')
    if NPU:
        ret = rknn.init_runtime(target='rk3399pro', device_id='2YJLMNA67N')
    else:
        ret = rknn.init_runtime()
    b = time.time()
    if ret != 0:
        print('init runtime failed.')
        exit(ret)
    print('done %fs' % (b - a))

    if NEED_RUN_PIC:
        img = cv2.imread(PIC_PATH)  # BGR
        img_input = cv2.resize(img, (SIZE_W, SIZE_H))
        img_input = np.transpose(img_input, (2, 0, 1))
        print("Input shape: ", img_input.shape)
        # inference
        print('--> inference')
        a = time.time()
        outputs = rknn.inference(inputs=[img_input], data_format='nchw')
        b = time.time()
        print('done %f' % (b - a))

        input0_data = outputs[0]
        input1_data = outputs[1]
        input2_data = outputs[2]

        if (SIZE_W, SIZE_H) == (416, 256):
            input0_data = input0_data.reshape(3, 11, 8, 13)
            input1_data = input1_data.reshape(3, 11, 16, 26)
            input2_data = input2_data.reshape(3, 11, 32, 52)
        else:
            input0_data = input0_data.reshape(3, 11, 13, 13)
            input1_data = input1_data.reshape(3, 11, 26, 26)
            input2_data = input2_data.reshape(3, 11, 52, 52)

        input_data = []
        input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

        if isinstance(OBJ_THRESH, list) and isinstance(NMS_THRESH, list):
            for obj in OBJ_THRESH:
                for nms in NMS_THRESH:
                    boxes, classes, scores = output_post_process(
                        input_data, obj, nms)
                    if boxes is not None:
                        img_c = img.copy()
                        im = draw(image=img_c,
                                  boxes=boxes,
                                  scores=scores,
                                  classes=classes)
                        n = './rknn_out_pic/o{}_n{}_rknn.png'.format(obj, nms)
                        print('save ', n)
                        cv2.imwrite(n, im)
        else:
            boxes, classes, scores = output_post_process(
                input_data, OBJ_THRESH, NMS_THRESH)

            if boxes is not None:
                im = draw(image=img,
                          boxes=boxes,
                          scores=scores,
                          classes=classes)
                cv2.imwrite('./rknn_output.png', im)
        print("Run Pic done!")

    if NEED_RUN_VIDEO:
        capture = cv2.VideoCapture(VIDEO_PATH)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        print("==============================================================")
        print("File Information: {} \nframe_width:{} frame_height:{} fps:{} ".
              format(VIDEO_PATH, frame_width, frame_height, fps))
        print("==============================================================")
        if capture.isOpened() is False:
            print('Error openning the File')
        # save mp4
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_name = VIDEO_PATH[VIDEO_PATH.rfind('/') + 1:].split('.')[0]
        out_video_name = "out_{}.mp4".format(video_name)
        videoWriter = cv2.VideoWriter(SAVE_PATH + out_video_name, fourcc,
                                      int(fps), (frame_width, frame_height))
        c = 1.0
        t = 0
        frame_index = -1

        tracker = Tracker()
        detector = Detector()

        fgbg = cv2.createBackgroundSubtractorMOG2(1000, 800)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 形态学去噪

        while capture.isOpened():
            ret, frame = capture.read()
            if ret:
                frame_index += 1
                if frame_index % int(25 / frame_per_second) == 0:  ## correct_this
                    frame_index = 0
                else:
                    img = np.uint8(frame)
                    if detector.jam_frame_count > 0:
                        add_accident_bbox(img=img, accident='jam')
                    videoWriter.write(img)
                    continue
                ###############
                fgmask = fgbg.apply(frame)
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, element)  # 开运算去噪
                fgmask[fgmask == 127] = 0
                contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_no_sb = []
                # for contour in contours:
                #     if cv2.contourArea(contour) < 100:
                #         continue
                #     if cv2.contourArea(contour) > frame.shape[0] / 2 * frame.shape[1] / 2:
                #         continue
                #     x, y, w, h = cv2.boundingRect(contour)
                #     contours_no_sb.append([x, y, x + w, y + h])

                ###############
                org_h, org_w = frame.shape[:2]
                img = np.uint8(frame)
                img_input = cv2.resize(img, (SIZE_W, SIZE_H))
                img_input = np.transpose(img_input, (2, 0, 1))

                # inference

                a = time.time()
                outputs = rknn.inference(inputs=[img_input],
                                         data_format='nchw')

                input0_data = outputs[0]
                input1_data = outputs[1]
                input2_data = outputs[2]

                if (SIZE_W, SIZE_H) == (416, 256):
                    input0_data = input0_data.reshape(3, 11, 8, 13)
                    input1_data = input1_data.reshape(3, 11, 16, 26)
                    input2_data = input2_data.reshape(3, 11, 32, 52)
                else:
                    input0_data = input0_data.reshape(3, 85, 13, 13)
                    input1_data = input1_data.reshape(3, 85, 26, 26)
                    input2_data = input2_data.reshape(3, 85, 52, 52)

                input_data = []
                input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
                input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
                input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

                pred_infoes = output_post_process(
                    input_data, OBJ_THRESH, NMS_THRESH)

                if pred_infoes is not None:
                    res = []
                    for *box, classe, score in pred_infoes:
                        x, y, w, h = box
                        left = int(max(0, x) * org_w)
                        top = int(max(0, y) * org_h)
                        right = int(max(0, x + w) * org_w)
                        bottom = int(max(0, y + h) * org_h)
                        res.append({
                            'score': score,
                            'class': int(classe),
                            'ct': [(top + bottom) // 2, (left + right) // 2],
                            'bbox': np.array([top, left, bottom, right])
                        })

                    res = tracker.step(res)
                    ########
                    # spill
                    for contour in contours:
                        if cv2.contourArea(contour) < 400:
                            continue
                        if cv2.contourArea(contour) > frame.shape[0] / 2 * frame.shape[1] / 2:
                            continue
                        x, y, w, h = cv2.boundingRect(contour)
                        spill_flag = True
                        for r in res:
                            top, left, bottom, right = r['bbox']
                            box = [left, top, right, bottom]
                            if if_interact([x, y, x + w, y + h], box):
                                spill_flag = False
                                break
                        if spill_flag:
                            contours_no_sb.append([x, y, x + w, y + h])
                    ########
                    detector.show_results(img, res, contours_no_sb)
                videoWriter.write(img)

                b = time.time()
                t = t + (b - a)
                print('--> [{}]inference time:{}'.format(c, (b - a)))
                c += 1

            else:
                break
        capture.release()
        videoWriter.release()
        fps = 1.0 / (t / c)
        print('event:', detector.event_type_id)
        print(
            "Run Video done!\nAv. Time: {}, frame={}, time={}, FPS={}\n Save at {}"
                .format(t / c, c, t, fps, video_name))

    rknn.release()

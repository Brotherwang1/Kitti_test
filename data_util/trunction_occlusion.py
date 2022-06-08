
##########################################################################################################################################
####统计每一个object bounding box 中包含得点云数目, 并计算 trunction, occlusion 参数 
##########################################################################################################################################
import os
import numpy as np
from kitti import kitti_object,compute_box_3d
from matplotlib import pyplot as plt


def lidar_with_boxes(objects, calib):
    box3d_pts_2d, box3d_pts_3d = compute_box_3d(objects, calib.P) 
    box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
    return box3d_pts_3d_velo
# -----------------------------------------------------------------------------------------
if __name__ == '__main__': 
    root_dir = '/mnt/disk2/wangjunyong/wangjunyong/Roundabout_pedestrians/vehicle/training'
    dataset = kitti_object(root_dir) 
    filename_list = os.listdir('/mnt/disk2/wangjunyong/wangjunyong/Roundabout_pedestrians/label_no_trunction/')
    saveBasePath = '/mnt/disk2/wangjunyong/wangjunyong/Roundabout_pedestrians/label_2_new/'
    filename_list.sort()
    list_point_number=[]
    for i in filename_list:
        data_idx = int(i[1:6])
        filename = '%06d.txt'%(data_idx) 
        if not os.path.exists(saveBasePath):
            os.makedirs(saveBasePath)
        f = open(os.path.join(saveBasePath,filename), 'w')
        print(filename)
        lidar_data = dataset.get_lidar(data_idx)
        objects = dataset.get_label_objects(data_idx)
        calib = dataset.get_calibration(data_idx)
        for obj in objects:
            obj_number_point=0
            box3d_pts_3d_velo = lidar_with_boxes(obj, calib)
            min_x = min(box3d_pts_3d_velo[:,0])
            max_x = max(box3d_pts_3d_velo[:,0])
            min_y = min(box3d_pts_3d_velo[:,1])
            max_y = max(box3d_pts_3d_velo[:,1])
            min_z = min(box3d_pts_3d_velo[:,2])
            max_z = max(box3d_pts_3d_velo[:,2]) 
            left = max(obj.xmin, 0)
            right = min(obj.xmax, 1248)
            top = min(obj.ymax, 384)
            bottom = max(obj.ymin, 0)
            if left >= right or top <= bottom:
                obj.truncation = 1
            else:
                intersection = (right - left) * (top - bottom)
                area = (obj.xmax - obj.xmin) * (obj.ymax - obj.ymin)
                obj.truncation = 1 - intersection/area
                
            for i in range(lidar_data.shape[0]):
                temp = lidar_data[i,0:3]
                x = temp[0]
                y = temp[1]
                z = temp[2]
                if (x > min_x and x < max_x) and (y > min_y and y < max_y) and (z > min_z and z < max_z):
                    obj_number_point += 1
            if obj_number_point < 10:
                obj.occlusion = 2
            elif obj_number_point < 50:
                obj.occlusion = 1
            else:
                obj.occlusion = 0   
            s = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(obj.type,obj.truncation, obj.occlusion, obj.alpha, obj.xmin, obj.ymin, obj.xmax, obj.ymax, obj.h, obj.w, obj.l,obj.t[0],obj.t[1],obj.t[2],obj.ry)
            f.write(s +'\n')
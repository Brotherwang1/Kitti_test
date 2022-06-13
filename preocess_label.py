##########################################################################################################################################
#将采集到的carla 点云数据经行预处理，删除被墙体等遮挡的目标，去除采集错误的点云数据
####依次对五个文件夹进行处理
##########################################################################################################################################
import os
import numpy as np
from kitti import kitti_object
# -----------------------------------------------------------------------------------------
if __name__ == '__main__': 
    root_dir  = '/opt/carla-simulator/Kitti_test-main/Roundabout_pedestrians_V2'
    root_dir_vehicle = '/opt/carla-simulator/Kitti_test-main/Roundabout_pedestrians_V2/vehicle/training/'
    dataset = kitti_object(root_dir_vehicle) 
    filename_list = os.listdir(os.path.join(root_dir_vehicle, 'label_2/')) 
    for i in filename_list:
        a = 0
        data_idx = int(i[1:6])
        objects = dataset.get_label_objects(data_idx)
        filename = '%06d.txt'%(data_idx)
        if not os.path.exists(os.path.join(root_dir, 'label_2_new_2/')):
            os.makedirs(os.path.join(root_dir, 'label_2_new_2/'))
        f = open(os.path.join(os.path.join(root_dir, 'label_2_new_2/'),filename), 'w')
        for obj in objects:
            if obj.w < 0.5:
                obj.w = 0.6
                obj.l = 0.6
            s = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(obj.type, obj.truncation, obj.occlusion, obj.alpha, obj.xmin, obj.ymin, obj.xmax, obj.ymax, obj.h, obj.w, obj.l,obj.t[0],obj.t[1],obj.t[2],obj.ry)

            f.write(s +'\n')
   
        f.close()


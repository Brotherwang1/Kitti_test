##########################################################################################################################################
#将采集到的carla 点云数据经行预处理，删除被墙体等遮挡的目标，去除采集错误的点云数据
####依次对五个文件夹进行处理
##########################################################################################################################################
import os
import numpy as np
from kitti import Object3d,kitti_object,compute_box_3d



x_min = -40
x_max = 40
y_min = 0
y_max = 72


def lidar_with_boxes(objects, calib):
    box3d_pts_2d, box3d_pts_3d = compute_box_3d(objects, calib.P) 
    box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
    return box3d_pts_3d_velo
    

# -----------------------------------------------------------------------------------------
if __name__ == '__main__': 
    root_dir  = '/mnt/disk2/wangjunyong/T_junction_pedestrians/'
    root_dir_sensor1 = '/mnt/disk2/wangjunyong/T_junction_pedestrians/sensor1/training/'
    root_dir_sensor2 = '/mnt/disk2/wangjunyong/T_junction_pedestrians/sensor2/training/'
    #root_dir_sensor3 = '/mnt/disk2/wangjunyong/T_junction_pedestrians/sensor3/training/'
  

    root_dir_vehicle = '/mnt/disk2/wangjunyong/T_junction_pedestrians/vehicle/training/'

    root_dir_fusion_sensor_all = '/mnt/disk2/wangjunyong/wangjunyong/T_junction_pedestrians/vehicle_sensor_all/training/'


    dataset = kitti_object(root_dir_fusion_sensor_all) 
    filename_list = os.listdir(os.path.join(root_dir_fusion_sensor_all, 'label_2/')) 
  
     
    for i in filename_list:
        a = 0
        data_idx = int(i[1:6])
        lidar_data = dataset.get_lidar_reduced(data_idx)
        objects = dataset.get_label_objects(data_idx)
        calib = dataset.get_calibration(data_idx)
        filename = '%06d.txt'%(data_idx)
        if not os.path.exists(os.path.join(root_dir_fusion_sensor_all, 'label_2_new/')):
            os.makedirs(s.path.join(root_dir_fusion_sensor_all, 'label_2_new/'))
        f = open(os.path.join(os.path.join(root_dir_fusion_sensor_all, 'label_2_new/'),filename), 'w')
        for obj in objects:
            if obj == []:
                continue
            #if obj.type == 'Van':
                #obj.type = 'Truck'
            #if obj.type != 'Truck':
                #continue
            box3d_pts_3d_velo = lidar_with_boxes(obj, calib)
            min_x = min(box3d_pts_3d_velo[:,0])
            max_x = max(box3d_pts_3d_velo[:,0])
            min_y = min(box3d_pts_3d_velo[:,1])
            max_y = max(box3d_pts_3d_velo[:,1])
            min_z = min(box3d_pts_3d_velo[:,2])
            max_z = max(box3d_pts_3d_velo[:,2])
           
            if np.mean(box3d_pts_3d_velo[:,0]) < y_min or np.mean(box3d_pts_3d_velo[:,0]) > y_max or np.mean(box3d_pts_3d_velo[:,1]) > x_max or np.mean(box3d_pts_3d_velo[:,1]) < x_min :
                continue
            else:
                for i in range(lidar_data.shape[0]):
                    temp = lidar_data[i,0:3]
                    x = temp[0]
                    y = temp[1]
                    z = temp[2]
                    if (x > min_x and x < max_x) and (y > min_y and y < max_y) and (z > min_z and z < max_z):
                        s = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(obj.type, obj.truncation, obj.occlusion, obj.alpha,
                                                            obj.xmin, obj.ymin, obj.xmax, obj.ymax,
                                                            obj.h, obj.w, obj.l,obj.t[0],obj.t[1],obj.t[2],obj.ry )
                        f.write(s +'\n')
                        a += 1
                        break 
        f.close()


        size = os.path.getsize(os.path.join(os.path.join(os.path.join(root_dir_fusion_sensor_all, 'label_2_new/'),filename)))
        if size == 0:
            os.remove(os.path.join(os.path.join(os.path.join(root_dir_fusion_sensor_all, 'label_2_new/'),filename)))  # 删除这个文件

        if a == 0:
            print(data_idx)
            os.remove(os.path.join(root_dir + 'top_image/image_2/','%06d.png'%(data_idx)))  # 删除这个文件

            os.remove(os.path.join(root_dir_sensor1 + 'calib/','%06d.txt'%(data_idx)))  # 删除这个文件
            os.remove(os.path.join(root_dir_sensor1 + 'image_2/','%06d.png'%(data_idx)))  # 删除这个文件
            os.remove(os.path.join(root_dir_sensor1 + 'label_2/','%06d.txt'%(data_idx)))  # 删除这个文件
            os.remove(os.path.join(root_dir_sensor1 + 'velodyne/','%06d.bin'%(data_idx)))  # 删除这个文件
            os.remove(os.path.join(root_dir_sensor1 + 'velodyne_reduced/','%06d.bin'%(data_idx)))  # 删除这个文件


            os.remove(os.path.join(root_dir_sensor2 + 'calib/','%06d.txt'%(data_idx)))  # 删除这个文件
            os.remove(os.path.join(root_dir_sensor2 + 'image_2/','%06d.png'%(data_idx)))  # 删除这个文件
            os.remove(os.path.join(root_dir_sensor2 + 'label_2/','%06d.txt'%(data_idx)))  # 删除这个文件
            os.remove(os.path.join(root_dir_sensor2 + 'velodyne/','%06d.bin'%(data_idx)))  # 删除这个文件
            os.remove(os.path.join(root_dir_sensor2 + 'velodyne_reduced/','%06d.bin'%(data_idx)))  # 删除这个文件
            
            #os.remove(os.path.join(root_dir_sensor3 + 'calib/','%06d.txt'%(data_idx)))  # 删除这个文件
            #os.remove(os.path.join(root_dir_sensor3 + 'image_2/','%06d.png'%(data_idx)))  # 删除这个文件
            #os.remove(os.path.join(root_dir_sensor3 + 'label_2/','%06d.txt'%(data_idx)))  # 删除这个文件
            #os.remove(os.path.join(root_dir_sensor3 + 'velodyne/','%06d.bin'%(data_idx)))  # 删除这个文件
            #os.remove(os.path.join(root_dir_sensor3 + 'velodyne_reduced/','%06d.bin'%(data_idx)))  # 删除这个文件
            
        
            
            os.remove(os.path.join(root_dir_vehicle + 'calib/','%06d.txt'%(data_idx)))  # 删除这个文件
            os.remove(os.path.join(root_dir_vehicle + 'image_2/','%06d.png'%(data_idx)))  # 删除这个文件
            os.remove(os.path.join(root_dir_vehicle + 'label_2/','%06d.txt'%(data_idx)))  # 删除这个文件
            os.remove(os.path.join(root_dir_vehicle + 'velodyne/','%06d.bin'%(data_idx)))  # 删除这个文件
            os.remove(os.path.join(root_dir_vehicle + 'velodyne_reduced/','%06d.bin'%(data_idx)))  # 删除这个文件


            
            
   
            
 
   
            
       
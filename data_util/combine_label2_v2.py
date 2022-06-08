##########################################################################################################
##########################将多辆汽车的label给结合起来,并且将镜头外的目标label给去除##########################
##########################################################################################################
import os
from kitti import *
import numpy as np
from coordinate_global_all import raw_data_transform
from Roundabout_v6.location import location as Roundabout_v6_new_location
from pre_process_Carla_data_muily import x_min, x_max, y_min, y_max

'''
x_min = -40
x_max = 40
y_min = 0
y_max = 70.4
'''

def lidar_with_boxes(obj, calib):
    box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib.P) 
    box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
    return box3d_pts_3d_velo

def origin_label(objects_origin, list_location, f):
    for obj in objects_origin:
        if obj == []:
            continue
        #if obj.type == 'Van':
            #obj.type = 'Truck'
        #if obj.type != 'Truck':
            #continue
        location1, location2, location3 = obj.t[0], obj.t[1], obj.t[2]
        list_location.append([location1,location2,location3])
        s = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(obj.type,obj.truncation, obj.occlusion, obj.alpha,
                                                        obj.xmin, obj.ymin, obj.xmax, obj.ymax,
                                                        obj.h, obj.w, obj.l,obj.t[0],obj.t[1],obj.t[2],obj.ry )
        list_location.append([location1,location2,location3])
        f.write(s +'\n')



def label_2_origin(origin, objects_car_choose, calib_car_choose, X_choose, list_location, calib_choose,f):
    for obj in objects_car_choose:
        if obj == []:
            continue
        #if obj.type != 'Truck':
            #continue
        box3d_pts_3d_velo = lidar_with_boxes(obj, calib_car_choose)
        box3d_pts_3d_velo_transform = raw_data_transform(origin, X_choose, box3d_pts_3d_velo)
        if np.mean(box3d_pts_3d_velo_transform[:,0]) < y_min or np.mean(box3d_pts_3d_velo_transform[:,0]) > y_max or np.mean(box3d_pts_3d_velo_transform[:,1]) > x_max or np.mean(box3d_pts_3d_velo_transform[:,1]) < x_min :
            continue
        else:
            l = obj.l; w = obj.w; h = obj.h
            box3d_pts_3d_rect = calib_choose.project_velo_to_rect(box3d_pts_3d_velo_transform)
            x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
            y_corners = [0,0,0,0,-h,-h,-h,-h]
            z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
            temp = X_choose[3] + obj.ry
            temp =  temp - origin[3]
            R = roty(temp) 
            corners_3d = np.dot(np.linalg.inv(R), np.vstack([x_corners,y_corners,z_corners]))
            box3d_pts_3d_rect = np.transpose(box3d_pts_3d_rect)
            location1 = np.mean(box3d_pts_3d_rect[0,:] - corners_3d[0,:])
            location2 = np.mean(box3d_pts_3d_rect[1,:] - corners_3d[1,:])
            location3 = np.mean(box3d_pts_3d_rect[2,:] - corners_3d[2,:])
            corners_2d = project_to_image(np.transpose(box3d_pts_3d_rect), calib_choose.P)
            xmin = min(corners_2d[:,0]); xmax = max(corners_2d[:,0])
            ymin = min(corners_2d[:,1]); ymax = max(corners_2d[:,1])
            #if (xmax< 0 or xmin > 1248 or ymin < 0 or ymax > 500):
                #continue
            if list_location == []:
                list_location.append([location1,location2,location3])
            location_all = np.array(list_location)
            location = np.array([location1, location2, location3])
            error = np.sqrt((np.power(location_all - location,2)).sum(axis=1))
            if any(error < 1):
                continue
            else:
                s = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(obj.type,obj.truncation, obj.occlusion, obj.alpha,
                                                                xmin, ymin, xmax, ymax,
                                                                h, w, l, location1, location2, location3,temp)
                list_location.append([location1,location2,location3])
                f.write(s +'\n')
        

        

if __name__ == '__main__': 
    #import pdb;pdb.set_trace()
    X_vehicle = Roundabout_v6_new_location[0, :]
    X_sensor1 = Roundabout_v6_new_location[1, :]
    X_sensor2 = Roundabout_v6_new_location[2, :]
    X_sensor3 = Roundabout_v6_new_location[3, :]
    #X_sensor4 = Roundabout_v6_new_location[4, :]
    filename_list = os.listdir('/mnt/disk2/wangjunyong/Roundabout_pedestrians/vehicle/training/label_2/')

    root_dir  = '/mnt/disk2/wangjunyong/Roundabout_pedestrians/'
    
    root_dir_sensor1 = '/mnt/disk2/wangjunyong/Roundabout_pedestrians/sensor1/training/'
    root_dir_sensor2 = '/mnt/disk2/wangjunyong/Roundabout_pedestrians/sensor2/training/'
    root_dir_sensor3 = '/mnt/disk2/wangjunyong/Roundabout_pedestrians/sensor3/training/'
    #root_dir_sensor4 = '/mnt/disk2/wangjunyong/wangjunyong/Roundabout_pedestrians/sensor4/training/'

    root_dir_vehicle = '/mnt/disk2/wangjunyong/Roundabout_pedestrians/vehicle/training/'

    for i in filename_list:
        list_location = []
        data_idx = int(i[1:6])
        filename = '%06d.txt'%(data_idx) 
      
        dataset_vehicle = kitti_object(root_dir_vehicle)
        dataset_sensor1 = kitti_object(root_dir_sensor1)
        dataset_sensor2 = kitti_object(root_dir_sensor2)
        dataset_sensor3 = kitti_object(root_dir_sensor3)
     

        lidar_data_vehicle = dataset_vehicle.get_lidar(data_idx)
        objects_vehicle = dataset_vehicle.get_label_objects(data_idx)
        calib_vehicle = dataset_vehicle.get_calibration(data_idx)

        lidar_data_sensor1 = dataset_sensor1.get_lidar(data_idx)
        objects_sensor1= dataset_sensor1.get_label_objects(data_idx)
        calib_sensor1 = dataset_sensor1.get_calibration(data_idx)
    
        lidar_data_sensor2 = dataset_sensor2.get_lidar(data_idx)
        objects_sensor2= dataset_sensor2.get_label_objects(data_idx)
        calib_sensor2 = dataset_sensor2.get_calibration(data_idx)

        
        lidar_data_sensor3 = dataset_sensor3.get_lidar(data_idx)
        objects_sensor3= dataset_sensor3.get_label_objects(data_idx)
        calib_sensor3 = dataset_sensor3.get_calibration(data_idx)



        origin = X_vehicle # 选择原始坐标点
        saveBasePath = '/mnt/disk2/wangjunyong/Roundabout_pedestrians/All_label/label_2/'

        if not os.path.exists(saveBasePath):
            os.makedirs(saveBasePath)
        f = open(os.path.join(saveBasePath,filename), 'w')
        if objects_vehicle != []:
            origin_label(objects_vehicle, list_location, f)
        if objects_sensor1 != []:
            label_2_origin(X_vehicle, objects_sensor1, calib_sensor1, X_sensor1, list_location, calib_vehicle, f)
        if objects_sensor2 != []:
            label_2_origin(X_vehicle, objects_sensor2, calib_sensor2, X_sensor2, list_location, calib_vehicle, f)
        if objects_sensor3 != []:
            label_2_origin(X_vehicle, objects_sensor3, calib_sensor3, X_sensor3, list_location, calib_vehicle, f)
        #if objects_sensor4 != []:
            #label_2_origin(X_vehicle, objects_sensor4, calib_sensor4, X_sensor4, list_location, calib_vehicle, f)
        f.close() 
        size = os.path.getsize(os.path.join(saveBasePath,filename))
        if size == 0:
            print(data_idx)
            os.remove(os.path.join(saveBasePath,filename)) 

            os.remove(os.path.join(root_dir + 'top_image/image_2/','%06d.png'%(data_idx)))  # 删除这个文件
                
            os.remove(os.path.join(root_dir_vehicle + 'calib/','%06d.txt'%(data_idx)))  # 删除这个文件
            os.remove(os.path.join(root_dir_vehicle + 'image_2/','%06d.png'%(data_idx)))  # 删除这个文件
            os.remove(os.path.join(root_dir_vehicle + 'label_2/','%06d.txt'%(data_idx)))  # 删除这个文件
            os.remove(os.path.join(root_dir_vehicle + 'velodyne/','%06d.bin'%(data_idx)))  # 删除这个文件
            os.remove(os.path.join(root_dir_vehicle + 'velodyne_reduced/','%06d.bin'%(data_idx)))  # 删除这个文件

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
            
           
            os.remove(os.path.join(root_dir_sensor3 + 'calib/','%06d.txt'%(data_idx)))  # 删除这个文件
            os.remove(os.path.join(root_dir_sensor3 + 'image_2/','%06d.png'%(data_idx)))  # 删除这个文件
            os.remove(os.path.join(root_dir_sensor3 + 'label_2/','%06d.txt'%(data_idx)))  # 删除这个文件
            os.remove(os.path.join(root_dir_sensor3 + 'velodyne/','%06d.bin'%(data_idx)))  # 删除这个文件
            os.remove(os.path.join(root_dir_sensor3 + 'velodyne_reduced/','%06d.bin'%(data_idx)))  # 删除这个文件
            
            #os.remove(os.path.join(root_dir_sensor4 + 'calib/','%06d.txt'%(data_idx)))  # 删除这个文件
            #os.remove(os.path.join(root_dir_sensor4 + 'image_2/','%06d.png'%(data_idx)))  # 删除这个文件
            #os.remove(os.path.join(root_dir_sensor4 + 'label_2/','%06d.txt'%(data_idx)))  # 删除这个文件
            #os.remove(os.path.join(root_dir_sensor4 + 'velodyne/','%06d.bin'%(data_idx)))  # 删除这个文件
            

        
    


    

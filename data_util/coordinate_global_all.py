import numpy as np
from numpy import mat,pi
import os
from Roundabout_v6.location import location as Roundabout_pedestrians_location
from kitti import kitti_object
from math import sin as oldsin
from math import cos as oldcos


def sin(x):
    if x % pi == 0: 
        #if x is an integer mult of pi, like pi, 2pi, -7pi, etc.
        return 0
    else:
        return oldsin(x)

def cos(x):
    if x % pi != 0: 
        if x%(pi/2) == 0: 
            #if x is an integer mult of pi/2, like 3pi/2, 5pi/2, 7pi/2, etc.
            return 0
        else:
            return oldcos(x)
    else:
        return oldcos(x)


def calculate_rotation(pitch,yaw,roll):
   # pointcloud = np.fromfile(str("data1/car1/training/velodyne/000004.bin"), dtype=np.float32, count=-1).reshape([-1, 4])
    Rz = mat([[cos(yaw),sin(yaw),0],[-sin(yaw),cos(yaw),0],[0,0,1]])
    Ry = mat([[cos(pitch),0,-sin(pitch)],[0,1,0],[sin(pitch),0,cos(pitch)]])
    Rx = mat([[1,0,0],[0,cos(roll),sin(roll)],[0,-sin(roll),cos(roll)]])
    R = Rz*Ry*Rx
    return R



def raw_data_transform(vehicle_location,sensor1_location, pointcloud_sensor1):
    # 以点云1所在的坐标系作为全局坐标系，汽车1所在位置作为原点
    # 将2坐标转换为相对于1的坐标系中
    p2 = np.copy(pointcloud_sensor1)
    dx = sensor1_location[0] - vehicle_location[0]
    dy = sensor1_location[1] - vehicle_location[1]
    dz = sensor1_location[2] - vehicle_location[2]
    yaw = vehicle_location[3] - sensor1_location[3]

    distance = mat([[dx,-dy,dz]])  
    
    R1 = calculate_rotation(0, yaw, 0)    
    R2 = calculate_rotation(0, vehicle_location[3], 0)    
    distance = distance*R2
    p2[:,:3] = p2[:,:3]*R1
    p2[:,0] = distance[0,0]  + p2[:,0]
    p2[:,1] = distance[0,1]  + p2[:,1]
    p2[:,2] = distance[0,2]  + p2[:,2] 

    return p2


def reduced():

    
    if not os.path.exists(lidar_dir_transform_sensor1_reduced):
        os.makedirs(lidar_dir_transform_sensor1_reduced)
    
    if not os.path.exists(lidar_dir_transform_sensor2_reduced):
        os.makedirs(lidar_dir_transform_sensor2_reduced)
    
    if not os.path.exists(lidar_dir_transform_sensor3_reduced):
        os.makedirs(lidar_dir_transform_sensor3_reduced)
    
    #if not os.path.exists(lidar_dir_transform_sensor4_reduced):
        #os.makedirs(lidar_dir_transform_sensor4_reduced)

    if not os.path.exists(fusion_lidar_dir_all_reduced):
        os.makedirs(fusion_lidar_dir_all_reduced)



    for i in filename_list:
        idx = int(i[1:6])
        ##############reduced data################
        pointcloud_vehicle_reduced = dataset_vehicle.get_lidar_reduced(idx)
        pointcloud_sensor1_reduced = dataset_sensor1.get_lidar_reduced(idx)
        pointcloud_sensor2_reduced = dataset_sensor2.get_lidar_reduced(idx)
        pointcloud_sensor3_reduced = dataset_sensor3.get_lidar_reduced(idx)
        #pointcloud_sensor4_reduced = dataset_sensor4.get_lidar_reduced(idx)

        
        ###############################sensor1#########################
        pointcloud_sensor1_transform_reduced = raw_data_transform(vehicle_location, sensor1_location, pointcloud_sensor1_reduced)
        pointcloud_sensor1_transform_reduced.tofile(os.path.join(lidar_dir_transform_sensor1_reduced, '%06d.bin'%(idx)))
        
        ###############################sensor2#########################
        pointcloud_sensor2_transform_reduced = raw_data_transform(vehicle_location, sensor2_location, pointcloud_sensor2_reduced)
        pointcloud_sensor2_transform_reduced.tofile(os.path.join(lidar_dir_transform_sensor2_reduced, '%06d.bin'%(idx)))

        ###############################sensor3#########################
        pointcloud_sensor3_transform_reduced = raw_data_transform(vehicle_location, sensor3_location, pointcloud_sensor3_reduced)
        pointcloud_sensor3_transform_reduced.tofile(os.path.join(lidar_dir_transform_sensor3_reduced, '%06d.bin'%(idx)))

        ###############################sensor4#########################
        #pointcloud_sensor4_transform_reduced = raw_data_transform(vehicle_location, sensor4_location, pointcloud_sensor4_reduced)
        #pointcloud_sensor4_transform_reduced.tofile(os.path.join(lidar_dir_transform_sensor4_reduced, '%06d.bin'%(idx)))

        funsion_pointcloud_all_reduced = np.vstack((pointcloud_vehicle_reduced,pointcloud_sensor1_transform_reduced,pointcloud_sensor2_transform_reduced, pointcloud_sensor3_transform_reduced))
        funsion_pointcloud_all_reduced.tofile(os.path.join(fusion_lidar_dir_all_reduced, '%06d.bin'%(idx)))
        print('%06d'%(idx), "generate successfully")
    print("done")

def Raw_All():

    if not os.path.exists(fusion_lidar_dir1):
        os.makedirs(fusion_lidar_dir1)
    
    if not os.path.exists(fusion_lidar_dir2):
        os.makedirs(fusion_lidar_dir2)
    
    #if not os.path.exists(fusion_lidar_dir3):
        #os.makedirs(fusion_lidar_dir3)
    
    #if not os.path.exists(fusion_lidar_dir4):
        #os.makedirs(fusion_lidar_dir4)
    
    if not os.path.exists(fusion_lidar_dir_all):
        os.makedirs(fusion_lidar_dir_all)

    if not os.path.exists(lidar_dir_transform_sensor1):
        os.makedirs(lidar_dir_transform_sensor1)
    
    if not os.path.exists(lidar_dir_transform_sensor2):
        os.makedirs(lidar_dir_transform_sensor2)

    if not os.path.exists(lidar_dir_transform_sensor3):
        os.makedirs(lidar_dir_transform_sensor3)
    
    #if not os.path.exists(lidar_dir_transform_sensor4):
        #os.makedirs(lidar_dir_transform_sensor4)
    
    #############################################################################################################
    print("Now,creating the fusion lidar data")
    print("This will take several minutes")
    for i in filename_list:
        idx = int(i[1:6])
        pointcloud_vehicle = dataset_vehicle.get_lidar(idx)

        ##############raw data################
        pointcloud_sensor1 = dataset_sensor1.get_lidar(idx)
        pointcloud_sensor2 = dataset_sensor2.get_lidar(idx)
        pointcloud_sensor3 = dataset_sensor3.get_lidar(idx)
        #pointcloud_sensor4 = dataset_sensor4.get_lidar(idx)
      
        ###############################sensor1#########################
        pointcloud_sensor1_transform = raw_data_transform(vehicle_location, sensor1_location, pointcloud_sensor1)
        pointcloud_sensor1_transform.tofile(os.path.join(lidar_dir_transform_sensor1, '%06d.bin'%(idx)))
        

        #funsion_pointcloud_sensor1 = np.vstack((pointcloud_vehicle, pointcloud_sensor1_transform))
        #funsion_pointcloud_sensor1.tofile(os.path.join(fusion_lidar_dir1, '%06d.bin'%(idx)))

        ###############################sensor2#########################
        pointcloud_sensor2_transform = raw_data_transform(vehicle_location, sensor2_location, pointcloud_sensor2)
        pointcloud_sensor2_transform.tofile(os.path.join(lidar_dir_transform_sensor2, '%06d.bin'%(idx)))
        #funsion_pointcloud_sensor2 = np.vstack((pointcloud_vehicle, pointcloud_sensor2_transform))
        #funsion_pointcloud_sensor2.tofile(os.path.join(fusion_lidar_dir2, '%06d.bin'%(idx)))

        ###############################sensor3#########################
        pointcloud_sensor3_transform = raw_data_transform(vehicle_location, sensor3_location, pointcloud_sensor3)
        pointcloud_sensor3_transform.tofile(os.path.join(lidar_dir_transform_sensor3, '%06d.bin'%(idx)))
        #funsion_pointcloud_sensor3 = np.vstack((pointcloud_vehicle, pointcloud_sensor3_transform))
        #funsion_pointcloud_sensor3.tofile(os.path.join(fusion_lidar_dir3, '%06d.bin'%(idx)))

        ###############################sensor4#########################
        #pointcloud_sensor4_transform = raw_data_transform(vehicle_location, sensor4_location, pointcloud_sensor4)
        #pointcloud_sensor4_transform.tofile(os.path.join(lidar_dir_transform_sensor4, '%06d.bin'%(idx)))
        #funsion_pointcloud_sensor4 = np.vstack((pointcloud_vehicle, pointcloud_sensor4_transform))
        #funsion_pointcloud_sensor4.tofile(os.path.join(fusion_lidar_dir4, '%06d.bin'%(idx)))

        funsion_pointcloud_all = np.vstack((pointcloud_vehicle,pointcloud_sensor1_transform, pointcloud_sensor2_transform, pointcloud_sensor3_transform))
        funsion_pointcloud_all.tofile(os.path.join(fusion_lidar_dir_all, '%06d.bin'%(idx)))
        
        print('%06d'%(idx), "generate successfully")
    print("done")


if __name__ == '__main__':
    ################################ The location of every vechile ##############W################################
    vehicle_location = Roundabout_pedestrians_location[0, :] #vehicle
    sensor1_location = Roundabout_pedestrians_location[1, :] #sensor
    sensor2_location = Roundabout_pedestrians_location[2, :] #sensor
    sensor3_location = Roundabout_pedestrians_location[3, :] #sensor
    #sensor4_location = Roundabout_pedestrians_location[4, :] #sensor

    dataset_vehicle = kitti_object('/mnt/disk2/wangjunyong/Roundabout_pedestrians/vehicle/training/')
    dataset_sensor1 = kitti_object('/mnt/disk2/wangjunyong/Roundabout_pedestrians/sensor1/training/')
    dataset_sensor2 = kitti_object('/mnt/disk2/wangjunyong/Roundabout_pedestrians/sensor2/training/')
    dataset_sensor3 = kitti_object('/mnt/disk2/wangjunyong/Roundabout_pedestrians/sensor3/training/')
    #dataset_sensor4 = kitti_object('/mnt/disk2/wangjunyong/Roundabout_pedestrians/sensor4/training/')

    lidar_dir_transform_sensor1_reduced = os.path.join('/mnt/disk2/wangjunyong/wangjunyong/Roundabout_pedestrians/sensor1/training/velodyne_reduced/')
    lidar_dir_transform_sensor2_reduced = os.path.join('/mnt/disk2/wangjunyong/wangjunyong/Roundabout_pedestrians/sensor2/training/velodyne_reduced/')
    lidar_dir_transform_sensor3_reduced = os.path.join('/mnt/disk2/wangjunyong/wangjunyong/Roundabout_pedestrians/sensor3/training/velodyne_reduced/')
    #lidar_dir_transform_sensor4_reduced = os.path.join('/mnt/disk2/wangjunyong/wangjunyong/Roundabout_pedestrians/sensor4/training/velodyne_reduced/')
    fusion_lidar_dir_all_reduced = os.path.join('/mnt/disk2/wangjunyong/wangjunyong/Roundabout_pedestrians','vehicle_sensor_all/training/velodyne_reduced/')


    fusion_lidar_dir1 = os.path.join('/mnt/disk2/wangjunyong/wangjunyong/Roundabout_pedestrians','vehicle_sensor1/training/velodyne/')
    fusion_lidar_dir2 = os.path.join('/mnt/disk2/wangjunyong/wangjunyong/Roundabout_pedestrians','vehicle_sensor2/training/velodyne/')
    fusion_lidar_dir3 = os.path.join('/mnt/disk2/wangjunyong/wangjunyong/Roundabout_pedestrians','vehicle_sensor3/training/velodyne/')
    #fusion_lidar_dir4 = os.path.join('/mnt/disk2/wangjunyong/wangjunyong/Roundabout_pedestrians','vehicle_sensor4/training/velodyne/')
    fusion_lidar_dir_all = os.path.join('/mnt/disk2/wangjunyong/wangjunyong/Roundabout_pedestrians','vehicle_sensor_all/training/velodyne/')


    lidar_dir_transform_sensor1 = os.path.join('/mnt/disk2/wangjunyong/wangjunyong/Roundabout_pedestrians/sensor1/training/velodyne/')
    lidar_dir_transform_sensor2 = os.path.join('/mnt/disk2/wangjunyong/wangjunyong/Roundabout_pedestrians/sensor2/training/velodyne/')
    lidar_dir_transform_sensor3 = os.path.join('/mnt/disk2/wangjunyong/wangjunyong/Roundabout_pedestrians/sensor3/training/velodyne/')
    #lidar_dir_transform_sensor4 = os.path.join('/mnt/disk2/wangjunyong/wangjunyong/Roundabout_pedestrians/sensor4/training/velodyne/')

    filename_list = os.listdir('/mnt/disk2/wangjunyong/Roundabout_pedestrians/vehicle/training/label_2/')
    reduced()
    Raw_All()

 

 



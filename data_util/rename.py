import os
import pdb
root_dir_vehicle = '/mnt/disk2/wangjunyong/T_junction_pedestrians/vehicle/training/'
root_dir1 = '/mnt/disk2/wangjunyong/T_junction_pedestrians/sensor1/training/'
root_dir2 = '/mnt/disk2/wangjunyong/T_junction_pedestrians/sensor2/training/'

root_dir3 = '/mnt/disk2/wangjunyong/T_junction_pedestrians/sensor3/training/'
root_dir4 = '/mnt/disk2/wangjunyong/T_junction_pedestrians/sensor4/training/'
Path_list = [root_dir_vehicle, root_dir1, root_dir2, root_dir3, root_dir4]
index = 0
for Path in Path_list:
    path = [os.path.join(Path,"calib/"), os.path.join(Path,"image_2/"),  os.path.join(Path,"label_2/"),  os.path.join(Path,"velodyne/"), os.path.join(Path,"velodyne_reduced/")]   # 目标路径
    f = open(os.path.join(Path,"data.txt"), 'w')  
    for j in range(5):
        a = 0 
        filename_list = os.listdir(path[j]) 
        filename_list.sort()
        for i in filename_list:
            used_name = path[j] + filename_list[a]
            #idx = int(filename_list[a][1:6]) + index
            line = '%06d'%(a+index)
            new_name = path[j] + line + filename_list[a][6:10]
            os.rename(used_name,new_name)
            print("文件%s重命名成功,新的文件名为%s" %(used_name,new_name))
            a += 1
            if j == 0:
                f.write(line+'\n')
        f.close()   #将文件关闭



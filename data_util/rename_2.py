import os
import pdb
root_dir1 = '/mnt/disk2/wangjunyong/T_junction_pedestrians/top_image/image_2/'
index = 0


a = 0
filename_list = os.listdir(root_dir1) 
filename_list.sort()
for i in filename_list:
    used_name = root_dir1 + filename_list[a]
    #idx = int(filename_list[a][1:6]) + index
    line = '%06d'%(a+index)
    new_name = root_dir1 + line + filename_list[a][6:10]
    os.rename(used_name,new_name)
    print("文件%s重命名成功,新的文件名为%s" %(used_name,new_name))
    a += 1



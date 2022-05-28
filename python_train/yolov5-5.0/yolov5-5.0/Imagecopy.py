import os
import shutil

if __name__ =='__main__':
    base_dir = os.path.dirname(__file__)   # 获取当前文件目录
    tps = [r'\train',r'\valid']
    names = [r'\images',r'\labels']
    save_root = r'yolov5-5.0\myDatasets1_2_3'  # 文件夹
    origin_root = r'yolov5-5.0\mydata'    # 原始文件完整目录
    save_root = r'yolov5-5.0\myDatasets1_2_3'  # 文件夹
    origin_root = r'yolov5-5.0\mydata'    # 原始文件完整目录
    #files = os.listdir(origin_path)
    for name in names:
        for tp in tps:
            save_path = save_root  + name + tp
            origin_path = origin_root+name+tp
            img_origin_files = os.listdir(origin_path)
            new_img_name = 'd2'
            for file in img_origin_files:
                if file.split('.')[0]=='classes':
                    continue
                new_name = file.split('.')[0]+'_d3.'+file.split('.')[1]
                #print(origin_path+'\\'+file, save_path+'\\'+new_name)
                shutil.copyfile(origin_path+'\\'+file, save_path+'\\'+new_name)

import cv2 
import os
import random
import numpy as np

# 随机亮度
def random_light(num,ImgPath,AugPath):
    
    pass
if __name__=="__main__":
    # 设置随机数种子
    random_seed = 999
    random.seed(random_seed)
    cv2.setRNGSeed(random_seed)
    np.random.seed(random_seed)
    pass
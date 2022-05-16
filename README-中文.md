# YOLOv5.5 Detection with C#
---
**简介:**
本项目使用YOLOv5s模型，用自己的数据集进行训练，识别LED芯片，并且将网络在C#中通过OpencvSharp的DNN模块进行调用推理，将神经网络嵌入到实际的工程项目中。  
**任务:** 
识别LED芯片，并且在C#中进行推理。  

---
**目录：**
- [数据采集](#1-数据采集)
- [数据集制作](#2-数据集制作)
- [yolo模型](#3-yolo模型)
- [训练及推理](#4-训练及推理)
- [导出onnx模型](#5-导出onnx模型)
- [CSharp中调用onnx模型](#6-csharp中调用onnx模型)
- [[option]多线程]()
---
## 1 数据采集 
- 使用实际摄像头进行采集真实的图像：  
  数量：42 images  
<div align=center><img src="https://github.com/lin-tea/YOLOv5DetectionWithCSharp/blob/main/Pictures/datasets.png" width="70%" height="70%"></div>

## 2 数据集制作
- **数据集划分**：training:valid:test = 8:1:2
- **制作标签**：
  工具：[labelImg](https://github.com/tzutalin/labelImg)
  标签文件: ```name.txt```，每一行为 类别 中心x 中心y 宽 高,名字name和图片 name.jpg 对应。
    ```
    class x_center y_center width height
    ...
    ```
  1. 在Python环境中安装labelimg.
    ```shell
      pip install labelimg
    ```
  2. 分好数据集合结构如下：(这里手动分数据集)
    ```
    data/
        |-images/
            |-train/   
            |-valid/
        |-labels/
            |-train/
            |-valid/
    ```
  创建类别文件(.txt),每一行写一种类别名称
  这里创建classes.txt文件，内容如下：
    ```txt
    chip
    ```
  3. 在 data/ 文件处打开CMD、终端，如果是使用conda安装的python环境，注意激活环境。
    ```python
      >(conda activate [your-env-name])
      >labelimg data/images/train classes.txt
    ```
  可以看见如下图，选择标签格式为yolo，保存路径为 data/labels/train，标签，cost time!  
  <div align=center><img src="https://github.com/lin-tea/YOLOv5DetectionWithCSharp/blob/main/Pictures/labeling.png" width="50%" height="50%"></div>
  
## 3 yolo模型
- **什么是YOLOv模型？** see [Blog,Search，有很多大佬写的优秀的博客]()  
- **下载yolov5.5工程文件**([点我下载](https://github.com/ultralytics/yolov5/releases/tag/v5.0)),下载源文件以及选择权重文件，这里选择权重文件为**YOLOv5s**。
  <!-- Unzip Sources Zip file, In VSCode we can open this project and see:  -->
  解压源文件，在vscode打开工程文件：  
   <div align=center>
  <img name="YOLOv5 Og Project" src="https://github.com/lin-tea/YOLOv5DetectionWithCSharp/blob/main/Pictures/originYOLO.png" width="50%" height="50%">
</div>

- **配置自己的工程文件**:
  - data文件：保存数据文件配置 .yaml 格式，我们需要根据自己的数据集、训练目标，建立自己的data配置文件，这个项目配置文件如下：
    ```yaml
    # train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
    # the project will auto go to ../data/labels/train/ to get labels.
    train: ../data/images/train/  # train images  
    val: ../data/images/valid/  # valid images  

    # number of classes
    nc: 1

    # class names
    names: [ 'chip' ]  
    ```
  
  - model配置文件：根据选择模型，在/model/文件夹中，选择相应的配置文件，这里选择```yolov5s.yaml```文件，复制修改后，只修改 parameters中的nc参数为1，即类别数量，修改后如下：
    ```yaml
      # parameters
      nc: 1  # number of classes
      depth_multiple: 0.33  # model depth multiple
      width_multiple: 0.50  # layer channel multiple
      # anchors 模型锚框
      anchors:
        - [10,13, 16,30, 33,23]  # P3/8
        - [30,61, 62,45, 59,119]  # P4/16
        - [116,90, 156,198, 373,326]  # P5/32
      # 框架不变
      # YOLOv5 backbone
      backbone:
      # YOLOv5 head
    ```
  
  - 下载预训练权重([点我下载](https://github.com/ultralytics/yolov5/releases/tag/v5.0)),添加到工程文件中  
- **python环境**: `python>=3.8`
    ```shell
      > pip install -r requirements.txt
    ```
## 4 训练及推理
- **修改 ```train.py``` 文件**
  主要修改参数：
  ```
    --weights: 下载的预训练权重地址
    --cfg: model配置文件地址，步骤3创建的配置文件，如yolov5s.yaml
    --data: data配置文件地址
    --batch-size：根据实际情况
    --img size：32的倍数默认640x640
    --epochs: 根据实际情况选择
    --device: GPU更快
    --workers：线程
  ```
  修改后如下,(只显示改变部分)：
  ```python
    if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument('--weights', type=str, default='yolov5-5.0/weights/yolov5s.pt', help='initial weights path')
      parser.add_argument('--cfg', type=str, default='yolov5-5.0/models/myyolov5s.yaml', help='model.yaml path')
      parser.add_argument('--data', type=str, default='yolov5-5.0/data/chips.yaml', help='data.yaml path')
      parser.add_argument('--epochs', type=int, default=500)
      parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs')
      parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
      parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
      parser.add_argument('--workers', type=int, default=2, help='maximum number of dataloader workers')
  ```
- **Run!**：运行`train.py`程序 And **Wait... Get yourself some Tea 🍵and hear win~win~win!**
   训练结束后：会在当前路径下，生成```run/train/```文件夹输出训练结果、权重:
   <div align=center>
    <img name="TrainOutputFile" src="https://github.com/lin-tea/YOLOv5DetectionWithCSharp/blob/main/Pictures/TrainOutput.png" width="70%" height="70%"></div>

   权重保存在```./run/train/weight/```中，你将可以看到`best.pt`和`last.pt`两个pt模型文件。
- **Detect!**: 跟训练时类似，打开`detect.py`文件，同样需要修改一些参数，然后直接run就行：  
    ```
      --weight   ;训练后的`.pt`权重文件
      --source   ;推理图片或者视频
      --img-size ;模型输入图片大小
      [optional] --conf-thres  ;置信度
      [optional] --iou-thres   ;iou阈值
      [optional] --device
    ```  
    修改后如下：  
    ```python
    if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument('--weights', nargs='+', type=str, default='runs\\train\\exp5\\weights\\best.pt', help='model.pt path(s)')
      parser.add_argument('--source', type=str, default='yolov5-5.0\\myDatasets2\\chip\\images\\test\\8.jpg', help='source')  # file/folder, 0 for webcam
      parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
      parser.add_argument('--conf-thres', type=float, default=0.8, help='object confidence threshold')
      parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
      parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    ```  
    得到结果如下：  
    <div align=center>
      <img name="DetectImage" src="https://github.com/lin-tea/YOLOv5DetectionWithCSharp/blob/main/Pictures/detectImg.jpg" width="60%" height="60%"></div>  
    可以看到，尽管使用的数据集合小，但仍表现不错，只一张图片，在我的电脑(Intel i7-8565U)上用CPU上推理0.08s,画图,0.11s左右。
## 5 导出onnx模型  
- **YOLOv5模型概述**: YOLOv5.5在三种不同的大小的特征图进行目标检测，以每一个特征图的pixie为中心，用三种不同长宽比的anchor进行预测，原本的网络输出对对应每一个特征图上的每一个像素，输出一个预测向量{x,y,h,w,confidence,[classify results]}，长度为`4+1+num of classes`。`detect`用confidence以及nms算法对各种预测锚框进行筛选。
  - 对于预测结果的目标框回归：根据工程文件中`model/yolo.py/`文件中的`Detect.forward()`部分，其代码如下:  
    ```python []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)
    ```  

    ![1](http://latex.codecogs.com/svg.latex?\begin{cases}x=(2Sigmoid(x)-0.5+c)*stride\\\\y=(2Sigmoid(y)-0.5+r)*stride\\\\h=(2Sigmoid(h))^2*anchor_h\\\\w=(2Sigmoid(w))^2*anchor_w\\\\\end{cases})  
    - c表示当前位置在特征图上的宽度坐标(即像素坐标)，表示偏移量
    - r表示当前位置在特征图上的高度坐标(即像素坐标)，表示偏移量
    - stride，压缩到不同大小的特征图时的使用不同stride
    - anchor:anchor_h、anchor_w,不同anchor大小参数
    - tips：参考模型配置文件，anchor={h1,h2,h3,w1,w2,w3},stride={8,16,32}
- **修改工程文件中的切片操作**

- **把结果输出的三个特征图concat成二维的张量**  
  - 正常yolov5输出结构:   
      假设网络输入大小为[h,w],特征图步长stride={s1,s2,s3}，anchor数量为3:    
      `1x(5+num of classes)x3((h/s1)x(w/s1)+(h/s2)x(w/s2)+(h/s3)x(w/s3))`  
  - 修改：flatten成二维，对stride、anchor在(5+num of classes)维度上拼接，对于yolov5.5s最终输出格式为：25200x(5+num of classes)  
  - (当然也可以不修改，只是后续对结果处理需进行一定的改变)  
- **Output ONNX 模型！**
  
## 6 CSharp中调用onnx模型
  **tool**: `opencvsharp.dnn`[Here](https://github.com/shimat/opencvsharp/releases/tag/4.5.3.20211228).  

## Reference:
  [1] [YOLOv5 Document](https://docs.ultralytics.com/).  
  [2] What is Anchor? [Anchor Boxes for Object detection](https://stackoverflow.com/questions/70227234/anchor-boxes-for-object-detection).
  

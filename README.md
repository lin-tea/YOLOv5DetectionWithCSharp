# YOLOv5 Detection with C#
---
**简介:**
本项目使用YOLOv5s模型，用自己的数据集进行训练，识别LED芯片，并且将网络在C#中通过OpencvSharp的DNN模块进行调用推理，将神经网络嵌入到实际的工程项目中。

**任务:** 
识别LED芯片，并且在C#中进行推理。

---
**目录：**
- [数据采集](1.-数据采集)
- [数据集制作](2.-数据集制作)
- [yolo模型](3-yolo模型)
- [训练 & 推理](4.-训练&推理)
- [导出onnx模型](5.-导出onnx模型)
- [绘制结果]()
- [[option]多线程]()
---
## 1. 数据采集
- 使用实际摄像头进行采集真实的图像：

  数量：24 images
  
<div align=center><img src="https://github.com/lin-tea/YOLOv5DetectionWithCSharp/blob/main/Pictures/datasets.png" width="70%" height="70%"></div>

## 2. 数据集制作
- **数据集划分**：training:valid:test = 10:1:1
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
  
## 3. yolo模型
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
## 4. 训练&推理
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
- **Run!**：运行`train.py`程序 And **Wait... Get yourself some Tea and hear win~win~win!**

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
    
## 5. 导出onnx模型

## Reference:

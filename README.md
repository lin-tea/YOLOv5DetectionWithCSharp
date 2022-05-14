# YOLOv5 Detection with C#
---
**简介:**
本项目使用YOLOv5s模型，用自己的数据集进行训练，识别LED芯片，并且将网络在C#中通过OpencvSharp的DNN模块进行调用推理，将神经网络嵌入到实际的工程项目中。

**任务:** 
识别LED芯片，并且在C#中进行推理。

---
**目录：**
- [数据采集](https://github.com/lin-tea/Yolo-detection-In-C-/edit/main/README.md#1-%E6%95%B0%E6%8D%AE%E9%87%87%E9%9B%86)
- [数据集制作](https://github.com/lin-tea/Yolo-detection-In-C-/edit/main/README.md#2-%E6%95%B0%E6%8D%AE%E9%9B%86%E5%88%B6%E4%BD%9C)
- [yolo模型](https://github.com/lin-tea/Yolo-detection-In-C-/edit/main/README.md#3-yolo%E6%A8%A1%E5%9E%8B)
- [训练 & 推理](https://github.com/lin-tea/Yolo-detection-In-C-/edit/main/README.md#4-%E8%AE%AD%E7%BB%83--%E6%8E%A8%E7%90%86)
- [导出onnx模型](https://github.com/lin-tea/Yolo-detection-In-C-/edit/main/README.md#5-%E5%AF%BC%E5%87%BAonnx%E6%A8%A1%E5%9E%8B)
- [绘制结果]()
- [[option]多线程]()
---
## 1. 数据采集
- 使用实际摄像头进行采集真实的图像，如下：
![image](https://github.com/lin-tea/YOLOv5DetectionWithCSharp/blob/main/Pictures/datasets.png)
数量：24 images
## 2. 数据集制作
- **数据集划分**：training:valid:test = 10:1:1
- **制作标签**：
  工具：[labelImg](https://github.com/tzutalin/labelImg)
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
  3. 在 data/ 文件处打开CMD、终端，进入python环境
  ```python
    >>>labelimg data/images/train classes.txt
  ```
  可以看见如下图，选择标签格式为yolo，保存路径为 data/labels/train，标签，cost time
  ![images](https://github.com/lin-tea/YOLOv5DetectionWithCSharp/blob/main/Pictures/labeling.png)
  
## 3. yolo模型

## 4. 训练 & 推理

## 5. 导出onnx模型

## Reference:

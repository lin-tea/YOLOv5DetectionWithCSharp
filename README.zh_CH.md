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
- **网络整体结构**  
   <div align=center>
  <img name="Struct" src="https://github.com/lin-tea/YOLOv5DetectionWithCSharp/blob/main/Pictures/%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84.png" width="75%" height="50%">
</div>  

  - 这次目标只有LED芯片，故对于每一个预测框，其预测的向量为 x,y,h,w,p,c，p表示置信概率，c表示预测类别的概率，故其长度为6。  

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
  - 训练配置文件：参考 `data/hyp.scratch.yaml` 文件,可设置如训练时的图像增强、iou阈值、Mixup概率等。  
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
    <img name="TrainOutputFile" src="https://github.com/lin-tea/YOLOv5DetectionWithCSharp/blob/main/Pictures/train_Output.png" width="70%" height="70%"></div>

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

- **把结果输出的三个特征图concat成二维的张量**  
  - 正常yolov5输出结构:   
      假设网络输入大小为[h,w],特征图步长stride={s1,s2,s3}，anchor数量为3，输出结果如下：    
      stride = s1: `1,(5+num of classes),3(h/s1),3(w/s1)`  
      stride = s2: `1,(5+num of classes),3(h/s2),3(w/s2)`  
      stride = s3: `1,(5+num of classes),3(h/s3),3(w/s3)`  
  - 修改：flatten成二维，对stride、anchor在(5+num of classes)维度上拼接，对于yolov5.5s最终输出格式为：25200x(5+num of classes)  
  - (当然也可以不修改，只是后续对结果处理需进行一定的改变)    
  对`model/yolo.py`中的**Detect类的forward()** 修改如下：
  ```python
        def forward(self, x):
        #导出onnx到opencv
        x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x[i] = x[i].view(bs * self.na * ny * nx, self.no).contiguous() # reshape成二维
        return torch.cat(x) #batch进行Concat
  ```    
- **Output ONNX 模型！**  
  - **方法**: `model/export.py`   
  - 同样修改一些配置:  
  ```python
    if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument('--weights', type=str, default='runs\\train\\exp2\\weights\\best.pt', help='weights path')  # from yolov5/models/
      parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
      parser.add_argument('--batch-size', type=int, default=1, help='batch size')
      parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
      parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
      parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
  ```   
  - Run! 然后onnx文件会保存到与`.pt` 文件相同文件目录下。   
  - **通过netron查看onnx模型**，原来输出：展开后输出：  
   <div class="content_img" align=center>
      <img name="OriginONNXOutput" src="https://github.com/lin-tea/YOLOv5DetectionWithCSharp/blob/main/Pictures/OriginONNXOutput.png" width="300" height="300">     <img name="FlattenOutput" src="https://github.com/lin-tea/YOLOv5DetectionWithCSharp/blob/main/Pictures/FlattenOutput.png" width="300" height="300"><div>Left:Origin & Right:Flattened</div>  
  
## 6 CSharp中调用onnx模型
  **tool**: `opencvsharp.dnn`[Here](https://github.com/shimat/opencvsharp/releases/tag/4.5.3.20211228).   
            `VS2022`[VS2022](https://visualstudio.microsoft.com/).    
- 需要的动态链接库(DLL):
   `OpencvSharp` 在vs的C#中添加引用，根据.NET版本选择opencv包中的 `opencvsharp.dll`以及`opecvsharp.Extention.dll`，并且把的`OpenCvSharp-4.5.3-20211228/NativeLib/win/x64/OpenCvSharpExtern.dll"`复制到vs工程文件的debug、release文件中。  
  - 在C#文件中引用:  
  ```c#
    using OpenCvSharp;
    using OpenCvSharp.Dnn;
  ```
- **现在，创建直接的yolo类吧!**:
  - 读取将ONNX文件添加到在工程文件中与/bin/文件同一个父目录下，创建/Resource/文件，保存onnx网络文件：
  ```c#
        /// <summary>
        /// 读取onnx模型
        /// </summary>
        /// <param name="path">网络文件路径</param>
        /// <returns></returns>
        public bool readModel(string path)
        {
            try
            {
                net = Net.ReadNetFromONNX(path);  //读取网络
                isReadSuccess = true;
            }
            catch (Exception)
            {
                net = null;
                isReadSuccess = false;
                return false;
            }
            net.SetPreferableBackend(Backend.DEFAULT);  //CPU运行
            net.SetPreferableTarget(Target.CPU);
            return true;
        }
  ```  
  - 设置各种属性:
  ```c#
      private  Net net;  //网络
      public bool isReadSuccess = false;
      /// <summary>
      /// yolov5网络参数
      /// </summary>
      // 1. anchors 锚框，后面回归需要用到
      //      由于采样的图片长宽比不大，故直接用原始锚框大小
      //      对于3种不同特征图的锚框，对于每一个特征图有3种锚框
      //      小特征图，用大的锚框可以搜索更大的物体，而大特征图用小的锚框可以搜索更小的物体
      float [,] netAnchors = new float[3,6]{
      { 10.0F, 13.0F, 16.0F, 30.0F, 33.0F, 23.0F }, // 大特征图
      { 30.0F, 61.0F, 62.0F, 45.0F, 59.0F, 119.0F }, //中特征图
      { 116.0F, 90.0F, 156.0F, 198.0F, 373.0F, 326.0F}}; //小特征图
      // 2. stride，锚框的步长
      //      即对应三种特征图在降维时用的步长，根据这个可以得到特征图的box个数
      float[] netStride = new float[3] { 8.0F, 16.0F, 32.0F};
      // 3. 输入图像大小 32倍数
      //    这里为640x640
      float netHeight = 640;
      float netWidth = 640;
      // 4. 各种初始置信概率(阈值)
      //    可改
      float nmsThreshold = 0.45f;  //nms阈值
      float boxThreshold = 0.5f;  //置信度阈值
      float classThreshold = 0.45f; //类别阈值
      List<string> classname = new List<string>{ "chip" };
  ```   
- **推理Detect!** ，在C#中我们只需应用网络进行推理即可
  - 网络规定的图像大小 `[640,640]`，但我们实际的图片并不是相同大小，因此，我们需要对其输入进行resize，也需要把输出框进行resize；  
    ```c#
      // 1) 格式化输入图像，变成 1x3x640x640的tensor，并且归一化，
      // scaleFactor归一化
      // swapRB:BGR -> RGB
      Mat blob = CvDnn.BlobFromImage(ScrImg, scaleFactor:1.0 /255, size:new OpenCvSharp.Size(netWidth,netHeight), mean:new Scalar(0,0,0), swapRB:false,crop:fals);
      // 2) 高、宽的缩放大小倍率,用于最后输出锚框进行resize，复原到原始图像大小
      double ratio_h = ScrImg.Rows /netHeight;  // 输入图像的高: 640
      double ratio_w = ScrImg.Cols / netWidth;  // 输入图像的宽: 640
    ```  
  - Inference 向前传播！
    ```c#
      // 3) 输入网络
      //DateTime t1 = DateTime.Now;
      net.SetInput(blob);
      // 4) 向前传播 
      Mat netOutput = net.Forward();  //25200x6
    ```  
  - 取出输出结果的每一行，筛选置信率高的，锚框回归，添加为备选框
    ```c#
      int r = 0;  // index，第一行
      List<int> classIds = new List<int>(); //保存好的锚框的分类结果
      List<float> confidences = new List<float>(); //保存好的锚框的置信
      List<Rect> bouduary = new List<Rect>();     //保存好的锚框x,y,h,w
      // 5) 遍历所有预测锚框参数
      //    注：网络的输出格式已经将3个特征图对应的所有box的预测的3种不同锚框，flatten成 二维矩阵
      //        从大特征图开始的展开；
      //先按特征图大小进行for遍历
      for (int s = 0; s < 3; s++)
      {
          // 获取box个数
          int grid_x = (int)(netWidth / netStride[s]);
          int grid_y = (int)(netHeight / netStride[s]);
          //MessageBox.Show(grid_x.ToString()); //debug用
          //MessageBox.Show(grid_y.ToString());
          // 对每个anchor进行遍历
          for (int anchor = 0; anchor < 3; anchor++)
          {
              // 得到anchor高、宽大小
              double anchor_w = netAnchors[s, anchor * 2];
              double anchor_h = netAnchors[s, anchor * 2 + 1];
              // 对每一个box的预测结果进行遍历
              for (int j = 0; j < grid_y; j++)
              {
                  for (int i = 0; i < grid_x; i++)
                  {
                      // 在输出结果中获取对应的box的预测结果 1x6的向量
                      Mat pMat = netOutput.Row(r);
                      float[] pdata;
                      // 变成array，更好用
                      pMat.GetArray(out pdata);
                      // 第五个即为置信概率
                      double box_score = Sigmoid(pdata[4]);
                      if (box_score > boxThreshold)
                      {
                          //++num;
                          // pdata第六个开始为类别预测的概率，即第五个之后的向量为one-hot输出结果
                          // classifyOutput保存one-hot输出结果
                          float[] classifyOutput = pdata.Skip(5).ToArray();   
                          Mat score = new Mat(1,classname.Count(),MatType.CV_32FC1, classifyOutput); //转成1,类别数的二维Mat
                          OpenCvSharp.Point classIdpoint = new OpenCvSharp.Point();
                          double max_class_score;
                          // 获取最大值及其对应的index即分类的类别id
                          Cv2.MinMaxLoc(score, out double min_class_score, out max_class_score, out OpenCvSharp.Point minclassIdpoint, out classIdpoint);
                          // sigmoid得到分类的概率
                          max_class_score = Sigmoid((float)max_class_score);
                          if(max_class_score > boxThreshold)
                          {

                              // 是好的锚框，保存,锚框回归
                              double x = (Sigmoid(pdata[0]) * 2 - 0.5 + i) * netStride[s];
                              double y = (Sigmoid(pdata[1]) * 2 - 0.5 + j) * netStride[s];
                              double w = Math.Pow(Sigmoid(pdata[2]) * 2.0, 2.0) * anchor_w;
                              double h = Math.Pow(Sigmoid(pdata[3]) * 2.0, 2.0) * anchor_h;
                              //过滤面积太小的
                              //if (w >=45 && h >=45)
                              //{
                              //    int left = (int)((x - 0.5 * w) * ratio_w);
                              //    int top = (int)((y - 0.5 * h) * ratio_h);
                              //    classIds.Add(classIdpoint.X);
                              //    confidences.Add((float)max_class_score);
                              //    bouduary.Add(new Rect(left, top, (int)(w * ratio_w), (int)(h * ratio_h)));
                              //}
                          }
                      }
                      ++r;
                  }
              }
          }
      }
    ```  
  - Now,使用NMS对备选框进行筛选:
   ```c#
    // 保存最终的最好的锚框在备选框中的index
    int[] final_idx;
    // nms，非极大值抑制,有很多预测锚框是重叠的，故用nms可得到其中最好的一个锚框
    CvDnn.NMSBoxes(bouduary, confidences, classThreshold, nmsThreshold, out final_idx);
    // 根据最终锚框的idx，保存结果
    for(int i = 0; i < final_idx.Length; i++)
    {
        int idx = final_idx[i];
        output temp;
        temp.id = classIds[idx];   // 修改地方 应该获取类别而不是index
        temp.confidence = confidences[idx]; 
        temp.box = bouduary[idx]; 
        Results.Add(temp);
    }
   ```  
 - 用opencv在原图上画出锚框：
 ```c#
     for(int i = 0; i < res.Count; i++)
    {
        int left=res[i].box.X,top = res[i].box.Y;
        Cv2.Rectangle(img, res[i].box, new Scalar(0, 255, 0), thickness: 1, LineTypes.Link8);
        Cv2.PutText(img, (i+1).ToString(), new OpenCvSharp.Point(left, top), HersheyFonts.HersheySimplex, fontScale: 0.35, color: new Scalar(0, 0, 255), thickness: 1);
        if (isDrawLedCenter==true)
        {
            // 绘制十字标
            int x = res[i].box.X + res[i].box.Width / 2, y = res[i].box.Y + res[i].box.Height / 2;
            OpenCvSharp.Point py1 = new OpenCvSharp.Point(x,y-5);
            OpenCvSharp.Point py2 = new OpenCvSharp.Point(x, y + 5);
            OpenCvSharp.Point px1 = new OpenCvSharp.Point(x-5, y);
            OpenCvSharp.Point px2 = new OpenCvSharp.Point(x+5, y);
            Cv2.Line(img, py1, py2, new Scalar(255,0 , 0)); Cv2.Line(img, px1, px2, new Scalar(255, 0, 0));
        }
    }
 ```  
 
 - 输出Output结构：
 ```c#
    /// <summary>
    /// Output数据结构，即为预测锚框参数
    /// </summary>
    public struct output
    {
        public int id;//结果类别id
        public float confidence;//结果置信度
        public Rect box;//矩形框
    }
 ```  
 
 - 结果：
 <div align=center>
      <img name="testCsharp" src="https://github.com/lin-tea/YOLOv5DetectionWithCSharp/blob/main/Pictures/testCsharp.png" width="70%" height="65%"></div>   
## Reference:
  [1] [YOLOv5 Document](https://docs.ultralytics.com/).  
  [2] What is Anchor? [Anchor Boxes for Object detection](https://stackoverflow.com/questions/70227234/anchor-boxes-for-object-detection).
  

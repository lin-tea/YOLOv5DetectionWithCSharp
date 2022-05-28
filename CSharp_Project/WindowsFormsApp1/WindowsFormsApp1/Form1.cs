using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System.Threading;
using yoloDC;
namespace WindowsFormsApp1
{
    public partial class Form1 : Form
    {
        
        private Yolo myyolo;
        private string filepath;  
        private List<output> dcRes;  //保存预测框结构
        private object oj_img = new object();
        private object oj_dcres = new object();
        private bool isThres = false;
        private Bitmap img_copy;
        Thread t;
        public Form1()
        {
            InitializeComponent();
            myyolo = new Yolo();
            dcRes = new List<output>();
            // onnx模型地址
            string path = @"..\..\Sources\best.onnx";
            bool res = myyolo.readModel(path);
            // 设置各种阈值
            myyolo.setThresholds(scoreThreshold: 0.4f, nmsThreshold: 0.35f, classThreshold: 0.85f);
            //MessageBox.Show(Convert.ToString(res));
            t = new Thread(new ThreadStart(DCprocess));
        }
        ///多线程检测

        /// 线程执行函数 
        /// 
        private void DCprocess()
        {
            while (true)
            {
                Mat temp = new Mat();
                Bitmap img = null;
                lock (oj_img) 
                {
                    if (img_copy!=null)
                    {
                        img = new Bitmap(img_copy);
                    }
                }
                if (img != null)
                {
                    temp = OpenCvSharp.Extensions.BitmapConverter.ToMat(img);
                    Cv2.CvtColor(temp, temp, ColorConversionCodes.RGBA2RGB); // 输出RGB图像
                    bool flag = myyolo.Detect(temp, out OpenCvSharp.Size sz, out List<output> t_res);
                    if (flag) 
                    {
                        lock (oj_dcres)
                        {
                            dcRes.Clear();
                            dcRes = t_res;
                        }
                    }
                }
                Thread.Sleep(100);
            }
        }
        /// 推理检测--按钮
        /// 
        private void button1_Click(object sender, EventArgs e)
        {
            try
            {
                if (isThres) return;
                t.Start();
                isThres = true;
                //}
            }catch
            {
                MessageBox.Show("Something Wrong! Reload the image may be helpful.");
            }
        }
        /// 选择图片---按钮
        /// 
        private void button2_Click(object sender, EventArgs e)
        {
            this.openFileDialog1.Reset();
            this.openFileDialog1.InitialDirectory = System.IO.Path.GetFullPath(@"..\..\Sources\images\");
            this.openFileDialog1.Filter = "图片|*.png;*.jpg";
            this.openFileDialog1.RestoreDirectory = true;
            if (this.openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                filepath = this.openFileDialog1.FileName;
                timer1.Interval = 10;
                timer1.Start();
            }
        }
        // 鼠标像素坐标
        private void pictureBox1_MouseClick(object sender, MouseEventArgs e)
        {
            //lock (oj)
            //{
                this.textBox3.Text = e.X.ToString();
                this.textBox4.Text = e.Y.ToString();
            //}
            int idx = -1;
            lock (oj_dcres)
            {
                idx = myyolo.findBox(e.X, e.Y, dcRes);
            }
            if (idx != -1)
            {
                textBox8.Text = (idx + 1).ToString();
            }
            else
            {
                textBox8.Text = idx.ToString();
            }
        }
        private void button3_Click(object sender, EventArgs e) //根据芯片序号，获取中心位置
        {
            int n = Convert.ToInt32(this.textBox5.Text);
            try
            {
                int x, y, h, w;
                lock (oj_dcres)
                {
                    x = dcRes[n - 1].box.X;
                    y = dcRes[n - 1].box.Y;
                    h = dcRes[n - 1].box.Height;
                    w = dcRes[n - 1].box.Width;
                }
                this.textBox6.Text = (x+w/2).ToString();
                this.textBox7.Text = (y+h/2).ToString();
            }
            catch
            {
                MessageBox.Show("Something wrong!");
            }
        }
        private void timer1_Tick(object sender, EventArgs e)
        {

            if (filepath!="")
            {
                try 
                {
                    Image old;
                    Bitmap og;
                    //lock (oj)
                    //{
                        old = pictureBox1.Image;
                        og = (Bitmap)Image.FromFile(filepath);
                    //}
                    lock (oj_dcres)
                    {
                        lock (oj_img)
                        {
                            img_copy = new Bitmap(og);
                        }
                        if (dcRes.Count > 0)
                        {
                            Mat img = OpenCvSharp.Extensions.BitmapConverter.ToMat(og);
                            myyolo.drawPred(ref img, img.Size(), dcRes, false,false,false);
                            og = new Bitmap(img.ToMemoryStream());
                            this.textBox1.Text = dcRes.Count().ToString();
                        }
                    }
                    //lock (oj)
                    //{
                        if(old != null) old.Dispose();
                        pictureBox1.Image = og;
                        //img = (Bitmap)pictureBox1.Image.Clone();
                        pictureBox1.Show();
                    //}
                }
                catch
                {
                    MessageBox.Show("Something Wrong!");
                }
            }
            timer1.Interval = 20;
            timer1.Start();
        }
        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (isThres) { t.Abort();isThres = false; }

        }
    }
}
/// <summary>
/// yolov5 Detector Class
/// </summary>
namespace yoloDC
{
    /// <summary>
    /// Output数据结构，即为预测锚框参数
    /// </summary>
    public struct output
    {
        public int id;//结果类别id
        public float confidence;//结果置信度
        public Rect box;//矩形框
    }
    // yolo类
    /* Tips: 功能
            读取yolov5，onnx模型
            进行detect，在图像上画图锚框
        模型要求：锚框参数、置信度、类别onehot，三输出concat成一个向量，对于1类别，为6；对于(640,640)生成25200个预测锚框
     */
    public class Yolo
    {
        /// <summary>
        /// 各种属性
        /// </summary>
        private  Net net;  //网络
        public bool isReadSuccess = false;
        /// <summary>
        /// yolov5网络参数
        /// </summary>
        // 1. anchors 锚框
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
        // 4. 各种置信概率(阈值)
        //    可改
        float nmsThreshold = 0.45f;  //nms阈值
        float boxThreshold = 0.5f;  //置信度阈值
        float classThreshold = 0.45f; //类别阈值
        List<string> classname = new List<string>{ "chip" };
        
        /// <summary>
        /// 初始化函数
        /// </summary>
        public Yolo() { 
            net = new Net();
        }
        /// <summary>
        /// 各种方法
        /// </summary>
        public double Sigmoid(float x) // sigmoid函数
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
        /// <summary>
        /// 读取onnx模型
        /// </summary>
        /// <param name="path">路径</param>
        /// <returns></returns>
        public bool readModel(string path)
        {
            try
            {
                net = Net.ReadNetFromONNX(path);
                isReadSuccess = true;
            }
            catch (Exception)
            {
                net = null;
                isReadSuccess = false;
                return false;
            }
            net.SetPreferableBackend(Backend.DEFAULT);
            net.SetPreferableTarget(Target.CPU);
            return true;
        }
        /// <summary>
        /// 设置阈值
        /// </summary>
        /// <param name="scoreThreshold">锚框置信度阈值</param>
        /// <param name="nmsThreshold">nms阈值</param>
        /// <param name="classThreshold">类别阈值</param>
        public void setThresholds(float scoreThreshold=0.5f,float nmsThreshold=0.45f,float classThreshold = 0.45f)
        {
            this.boxThreshold = scoreThreshold;
            this.nmsThreshold = nmsThreshold;
            this.classThreshold = classThreshold;
        }
        /// <summary>
        /// 检测，预测
        /// </summary>
        /// <param name="ScrImg">输入图像: RGB</param>
        /// <param name="originSizeout">输出原始大小</param>
        /// <param name="Results">输出预测锚框</param>
        /// <returns></returns>
        public bool Detect(Mat ScrImg,out OpenCvSharp.Size originSizeout,out List<output> Results)
        {
            // 保存结果，格式如上面定义的结构体output一致
            Results = new List<output>();
            // 得到输入图像原本大小，用于后面恢复
            originSizeout = ScrImg.Size();
            if (isReadSuccess)
            {
                if(ScrImg.Empty()) return false;
                //Cv2.ImShow("test image", ScrImg);
                //Cv2.WaitKey(0);
                // 1) 格式化输入格式，变成 1x3x640x640的tensor，并且归一化
                Mat blob = CvDnn.BlobFromImage(ScrImg, scaleFactor:1.0 /255, size:new OpenCvSharp.Size(netWidth,netHeight), mean:new Scalar(0,0,0), swapRB:false,crop:false);
                // int c = blob.Cols; //debug用
                //MessageBox.Show(blob.Cols);
                // 2) 输入网络
                //DateTime t1 = DateTime.Now;
                net.SetInput(blob);
                // 3) 向前传播 

                Mat netOutput = net.Forward();
                //DateTime t2 = DateTime.Now;
                //TimeSpan dt = t2 - t1;
                //MessageBox.Show(Convert.ToString(dt.Milliseconds));
                // 4) 高、宽的缩放大小倍率
                double ratio_h = ScrImg.Rows /netHeight;  // 输入图像的高 / 640
                double ratio_w = ScrImg.Cols / netWidth;  // 输入图像的宽 / 640
                //int net_width = classname.Count() + 5;  //x,y,h,w,confidence,类别one-hot
                //int num = 0; //debug 查看好的锚框数量
                int r = 0;  // index，第一行
                //float[] pdata; //保存每一个box预测向量 长度为net_width
                List<int> classIds = new List<int>(); //保存好的锚框的分类结果
                List<float> confidences = new List<float>(); //保存好的锚框的置信
                List<Rect> bouduary = new List<Rect>();     //保存好的锚框x,y,h,w
                //MessageBox.Show(netOutput.Count().ToString()); //查看网络输出数量,debug
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
                                    float[]classifyOutput = pdata.Skip(5).ToArray();   
                                    Mat score = new Mat(1,classname.Count(),MatType.CV_32FC1, classifyOutput); //转成1,类别数的二维Mat
                                    OpenCvSharp.Point classIdpoint = new OpenCvSharp.Point();
                                    double max_class_score;
                                    // 获取最大值及其对应的index即分类的类别id
                                    Cv2.MinMaxLoc(score, out double min_class_score, out max_class_score, out OpenCvSharp.Point minclassIdpoint, out classIdpoint);
                                    // sigmoid得到分类的概率
                                    max_class_score = Sigmoid((float)max_class_score);
                                    if(max_class_score > boxThreshold)
                                    {

                                        // 是好的锚框，保存
                                        // 计算公式来自于官方，参考loss
                                        double x = (Sigmoid(pdata[0]) * 2 - 0.5 + i) * netStride[s];
                                        double y = (Sigmoid(pdata[1]) * 2 - 0.5 + j) * netStride[s];
                                        double w = Math.Pow(Sigmoid(pdata[2]) * 2.0, 2.0) * anchor_w;
                                        double h = Math.Pow(Sigmoid(pdata[3]) * 2.0, 2.0) * anchor_h;

                                        if (w >=75 && h >=75)
                                        {
                                            int left = (int)((x - 0.5 * w) * ratio_w);
                                            int top = (int)((y - 0.5 * h) * ratio_h);
                                            classIds.Add(classIdpoint.X);
                                            confidences.Add((float)max_class_score);
                                            bouduary.Add(new Rect(left, top, (int)(w * ratio_w), (int)(h * ratio_h)));
                                        }
                                    }
                                }
                                ++r;
                            }
                        }
                    }
                }
                // 保存最终的最好的锚框id
                int[] final_idx;
                // nms，非极大值抑制,有很多预测锚框是重叠的，故用nms可得到其中最好的一个锚框
                CvDnn.NMSBoxes(bouduary, confidences, classThreshold, nmsThreshold, out final_idx);
                // MessageBox.Show(Convert.ToString(final_idx.Length)); //debug,得到最终预测数量
                //MessageBox.Show(num.ToString());
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
            }
            if (Results.Count > 0)
            {
                // 排序
                Results.Sort(delegate (output x, output y)
                {
                    if (x.box.Y >= y.box.Y)
                    {
                        return 1;
                    }
                    else return -1;
                });
                int row_base_Y = Results[0].box.Y,dy=40;
                List<output> temp = new List<output>();
                for (int i = 0; i < Results.Count; i++)
                {
                    List<output> temp_row = new List<output>();
                    while(i<Results.Count && Math.Abs(row_base_Y-Results[i].box.Y)<dy) temp_row.Add(Results[i++]);
                    temp_row.Sort(delegate (output x, output y)
                    {
                        if (x.box.X >= y.box.X)
                        {
                            return 1;
                        }
                        else return -1;
                    });
                    for(int j = 0; j < temp_row.Count; j++)
                    {
                        temp.Add(temp_row[j]);
                    }
                    if(i<Results.Count)row_base_Y = Results[i].box.Y;
                    --i;
                }
                Results = temp;
                return true;
            }
            else
                return false;
        }
        /// <summary>
        /// 画出锚框
        /// </summary>
        /// <param name="img">输入图像</param> 
        /// <param name="originSize">图像输出大小</param>
        /// <param name="res">锚框,数据为output结构</param>
        /// <param name="isShow">显示结果图片</param>
        /// <param name="isDrawImgCenter">是否显示图像中心</param>
        /// <param name="isDrawLedCenter">是否显示LED中心</param>
        public void drawPred(ref Mat img, OpenCvSharp.Size originSize,List<output> res,bool isShow=false,bool isDrawLedCenter=false,bool isDrawImgCenter=true)
        {
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
            if (isDrawImgCenter)
            {
                // 绘制十字标
                int x = img.Width, y = img.Height;
                OpenCvSharp.Point py1 = new OpenCvSharp.Point(x/2,10 );
                OpenCvSharp.Point py2 = new OpenCvSharp.Point(x/2,y-10);
                OpenCvSharp.Point px1 = new OpenCvSharp.Point(10, y/2);
                OpenCvSharp.Point px2 = new OpenCvSharp.Point(x-10, y/2);
                Cv2.Line(img, py1, py2, new Scalar(0, 0, 255)); Cv2.Line(img, px1, px2, new Scalar(0,0, 255));
            }
            Cv2.Resize(img, img, originSize);
            if (isShow)
            {
                Cv2.ImShow("test", img);
                Cv2.WaitKey(0);
            }
        }
        /// <summary>
        ///  根据像素位置得到所在锚框
        /// </summary>
        /// <param name="x">像素坐标x</param>
        /// <param name="y">像素坐标y</param>
        /// <param name="bound">锚框</param>
        /// <returns> index，若为-1表示无
        /// </returns>
        public int findBox(int x,int y, List<output> bound)
        {
            if (bound.Count>0)
            {
                for(int i = 0; i < bound.Count; i++)
                {
                    if(x>=bound[i].box.X && y >= bound[i].box.Y)
                    {
                        if(x<=bound[i].box.Right && y <= bound[i].box.Bottom)
                        {
                            return i;
                        }
                    }
                }
            }
            return -1;
        }
    }
}
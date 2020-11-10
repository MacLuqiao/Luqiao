package luqiao;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;
import org.opencv.objdetect.HOGDescriptor;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class MySVM {
    static {System.loadLibrary( Core.NATIVE_LIBRARY_NAME );}
    /*
     * 初始化特征提取器
     * @param _winSize 特征提取检测的窗口大小
     * @param _blockSize 块大小，。
     * @param _blockStride 检测步长。
     * @param _cellSize 胞元，胞元是在块中。
     * @param _nbins 检测方向 在一个胞元内统计9个方向的梯度直方图，每个方向为180/9=20度。
     */
    public static Size HOG_winSize = new Size(64, 128);
    public static Size HOG_blockSize = new Size(16, 16);
    public static Size HOG_blockStride = new Size(8, 8);
    public static Size HOG_cellSize = new Size(8, 8);
    public static int HOG_nbins = 9;
    public static int ITERATION_NUM = 100;
    public static int PIC_WIDTH = 128;
    public static int PIC_HEIGHT = 128;
    public static HOGDescriptor hogDescriptor = new HOGDescriptor(HOG_winSize, HOG_blockSize, HOG_blockStride, HOG_cellSize, HOG_nbins);
    public static Mat trainImages;
    public static Mat trainLabels;
    public static Mat testImages;
    public static Mat testLabels;
    private SVM svm;

    public MySVM(){}

    public MySVM(String path){
        System.out.println("开始加载SVM");
        File file = new File(path);
        if(file.exists())
            System.out.println("文件存在！！！！！！");
        else
            System.out.println("文件不存在！！！！！");
        long start = System.currentTimeMillis();
        svm = SVM.load(path);
        System.out.println("加载SVM成功，用时："+(System.currentTimeMillis()-start)+"ms");
    }

    public Mat getHOGdescriptors(Mat src) {
//        System.out.println(path);
//        Mat src = Imgcodecs.imread(path);//待匹配图片
        src = resizeImage(src);
//        HighGui.imshow("展示图片", src);
//        HighGui.waitKey();
        Mat gray=new Mat();
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
        MatOfFloat descriptors = new MatOfFloat();
        hogDescriptor.compute(gray, descriptors);
        //System.out.println(descriptors.size());
        return descriptors.reshape(0,1);
    }

    public Mat getHOGdescriptors(String path) {
        //System.out.println(path);
        Mat src = Imgcodecs.imread(path);//待匹配图片
        src = resizeImage(src);
//        HighGui.imshow("展示图片", src);
//        HighGui.waitKey();
        Mat gray=new Mat();
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
        MatOfFloat descriptors = new MatOfFloat();
        hogDescriptor.compute(gray, descriptors);
        //System.out.println(descriptors.size());
        return descriptors.reshape(0,1);
    }

    public Mat resizeImage(Mat src){
        int w = src.cols();
        int h = src.rows();
        int padw = PIC_WIDTH-w,padh=PIC_HEIGHT-h;
        padw = padw<0?0:padw;
        padh = padh<0?0:padh;
        Mat p=new Mat();
        Core.copyMakeBorder(src, p,padh/2,padh/2,padw/2,padw/2,Core.BORDER_CONSTANT,new Scalar(0));
        Imgproc.resize(p, src, new Size(PIC_WIDTH,PIC_HEIGHT), 0, 0, Imgproc.INTER_LINEAR);
        return src;
    }

    public void trainSVM(Mat data,Mat label){
        System.out.println("开始训练SVM");
        SVM svm = SVM.create();
        svm.setType(SVM.C_SVC);
        svm.setKernel(SVM.LINEAR);
        svm.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER, ITERATION_NUM, 1e-6));
        boolean success = svm.train(data, Ml.ROW_SAMPLE, label);
        System.out.println("SVM训练完成");
        System.out.println(success);
        svm.save("d:/aaa.xml");
        System.out.println("成功保存SVM模型");
    }

    public void testSVM(Mat data,Mat label){
        assert svm != null;
        System.out.println("开始测试SVM");
        Mat pred = new Mat();
        svm.predict(data, pred, 0);
        float acc=0;
        //System.out.println(pred.dump());
        for(int i=0;i<label.cols();++i){
            System.out.print(pred.get(i,0)[0]);
            System.out.print("->");
            System.out.println(label.get(i,0)[0]);
            if((int)label.get(i,0)[0]==(int)pred.get(i,0)[0])
                acc+=1;
        }
        System.out.println("SVM测试完成");
        System.out.println("测试精度为："+acc/label.cols()*100+"%");
    }

    public int predict(Mat data){
        assert svm != null;
        //System.out.println(data.size(1));
        System.out.println("开始预测单样本");
        long start = System.currentTimeMillis();
        Mat pred = new Mat();
        svm.predict(data, pred, 0);
        System.out.println("预测成功："+(int) pred.get(0,0)[0]);
        System.out.println("用时："+(System.currentTimeMillis()-start)+"ms");
        return (int) pred.get(0,0)[0];
    }

    public List<Mat> makeTrainDataset(String dir){
        System.out.println("开始制备训练数据集");
        int sample_cnt = 0;
        ArrayList<Mat> trainingImages = new ArrayList<Mat>();
        ArrayList<Integer> trainingLabels = new ArrayList<Integer>();
        for (String file:new File(dir+"/cars").list()){
            sample_cnt++;
            Mat descriptors = getHOGdescriptors(dir+"/cars/"+file);
            //System.out.println(descriptors.size());
            trainingImages.add(descriptors);
            trainingLabels.add(1);
        }
        System.out.println("正例样本："+sample_cnt+"个");
        for (String file:new File(dir+"/others").list()){
            //System.out.println(file);
            sample_cnt++;
            Mat descriptors = getHOGdescriptors(dir+"/others/"+file);
            trainingImages.add(descriptors);
            trainingLabels.add(-1);
        }
        int dim = trainingImages.get(0).cols();
//        System.out.println(sample_cnt);
//        System.out.println(dim);
        trainImages = new Mat(sample_cnt, dim, CvType.CV_32FC1);
        trainLabels = new Mat(sample_cnt, 1, CvType.CV_32S);
        Core.vconcat(trainingImages, trainImages);
        for (int i = 0; i < trainingLabels.size(); i++) {
            int[] val = {trainingLabels.get(i)};
            trainLabels.put(i, 0, val);
        }
        List list = new ArrayList();
        list.add(trainImages);
        list.add(trainLabels);
        System.out.println("总样本数："+sample_cnt+"个");
        System.out.println("HOG特征维度："+dim);
        System.out.println(trainImages.size());
        System.out.println(trainLabels.size());
        System.out.println("制备训练集完成!");
        return list;
    }

    public List<Mat> makeTestDataset(String dir){
        System.out.println("开始制备测试数据集");
        int sample_cnt = 0;
        ArrayList<Mat> testingImages = new ArrayList<Mat>();
        ArrayList<Integer> testingLabels = new ArrayList<Integer>();
        for (String file:new File(dir+"/cars").list()){
            sample_cnt++;
            Mat descriptors = getHOGdescriptors(dir+"/cars/"+file);
            //System.out.println(descriptors.size());
            testingImages.add(descriptors);
            testingLabels.add(1);
        }
        System.out.println("正例样本："+sample_cnt+"个");
        for (String file:new File(dir+"/others").list()){
            //System.out.println(file);
            sample_cnt++;
            Mat descriptors = getHOGdescriptors(dir+"/others/"+file);
            testingImages.add(descriptors);
            testingLabels.add(-1);
        }
        int dim = testingImages.get(0).cols();
//        System.out.println(sample_cnt);
//        System.out.println(dim);
        testImages = new Mat(sample_cnt, dim, CvType.CV_32FC1);
        testLabels = new Mat(sample_cnt, 1, CvType.CV_32S);
        Core.vconcat(testingImages, testImages);
        for (int i = 0; i < testingLabels.size(); i++) {
            int[] val = {testingLabels.get(i)};
            testLabels.put(i, 0, val);
        }
        List list = new ArrayList();
        list.add(testImages);
        list.add(testLabels);
        System.out.println("总样本数："+sample_cnt+"个");
        System.out.println("HOG特征维度："+dim);
        System.out.println(testImages.size());
        System.out.println(testLabels.size());
        System.out.println("制备测试集完成!");
        return list;
    }
}

package luqiao;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.*;
import org.opencv.videoio.*;
import org.opencv.video.*;
import org.opencv.imgproc.*;
//c3
//            lineInforms = "869,104-1025,94-1907,1035-80,996";

//c22
//            lineInforms = "287,96-614,70-1899,512-293,1071";
public class Demo {
    public static String TXT_PATH = "D:\\b.txt";              // txt文件路径
    public static String VIDEO_PATH = "D:\\a.avi";            // 读取视频文件路径
    public static String SAVE_PATH = "d:/test_c22_61_53.avi";              // 存储视频文件路径
    public static String detectRegion = "287,96-614,70-1899,512-293,1071";                   // 抛洒物检测区域
    public static String prohibitRegion = "";      // 不检测区域
    public static String crossLines = "245,957,1186,277-1186,277,1282,181-672,994,1303,319-1303,319,1377,185";  //道路线信息
    public static double objectScore = 0.6;                   // 检测框阈值
    public static int framePerSecond = 5;                     // 每秒取的视频帧个数
    public static MySVM mySVM = new MySVM("d:/SVM128x128.xml"); // 加载SVM模型
    public static BackgroundSubtractorMOG2 fgbg =  Video.createBackgroundSubtractorMOG2(1000, 800);
    public static Accident accident = new Accident();
    public static Tracker tracker = new Tracker();
    // 加载Opencv库
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
    public static void main(String[] args) {
        // 读取txt数据
        List<List<YoloObject>> results = new ArrayList<>();
        try {
            String encoding="GBK";
            List<YoloObject> res = new ArrayList<>();
            File file = new File(TXT_PATH);
            if(file.isFile() && file.exists()) {
                InputStreamReader read = new InputStreamReader(new FileInputStream(file),encoding);
                BufferedReader bufferedReader = new BufferedReader(read);
                String lineTxt = "";
                while((lineTxt = bufferedReader.readLine()) != null){
                    if (lineTxt.isEmpty()){
                        results.add(res);
                        res = new ArrayList<>();
                    }
                    else {
                        String[] strings = lineTxt.split(" ");
                        float score = Float.parseFloat(strings[0]);
                        int label = Integer.parseInt(strings[1]);
                        if(score > objectScore){          // 判断检测框置信度
                            int[] box = {0,0,0,0,0,0};
                            for(int i=2;i<strings.length;i++)
                                box[i-2] = Integer.parseInt(strings[i]);
                            DetectBox detectBox = new DetectBox(box[3], box[2], box[5], box[4]);
                            YoloObject obj = new YoloObject(label, score, detectBox);
                            res.add(obj);
                        }
                    }
                }
                read.close();
            }else{
                System.out.println("找不到指定的文件");
            }
        } catch (Exception e) {
            System.out.println("读取文件内容出错");
            e.printStackTrace();
        }

        // 读取视频，根据检测框匹配车辆，检测事件
        VideoCapture capture = new VideoCapture();
        capture.open(VIDEO_PATH);
        int cnt = 0;
        int frame_index = -1;
        Size size = new Size(capture.get(Videoio.CAP_PROP_FRAME_WIDTH),capture.get(Videoio.CAP_PROP_FRAME_HEIGHT));
        VideoWriter writer = new VideoWriter(SAVE_PATH, VideoWriter.fourcc('M', 'J', 'P', 'G'), 25.0, size);
        Mat frame = new Mat();
        if (!capture.isOpened()) {
            System.out.println("Error opening video stream or file");
            return;
        }
        while(capture.isOpened()) {
            capture.read(frame);
            if(frame.empty())
                break;
            frame_index += 1;
            if (frame_index % (25/framePerSecond) == 0)
                frame_index = 0;
            else {
                writer.write(frame);
                continue;
            }
            if(results.get(cnt).size()>0) {
                System.out.println(cnt);
                // 匹配跟踪
                tracker.step(results.get(cnt), accident);
                // 混合高斯背景减除模型
                Mat fgmask = new Mat();
                fgbg.apply(frame, fgmask);
                GMM gmm = new GMM(mySVM, frame, fgmask, tracker);
                List<DetectBox> fgBox = gmm.fg_box();
                // 检测事件
                List<AccidentInform> accidentInforms = Accident.checkAccident(tracker.track_box, fgBox, detectRegion, prohibitRegion, crossLines);
                // 画框
                DrawUtils.draw(frame, accidentInforms, crossLines);
            }
            cnt++;
            writer.write(frame);
        }
    }
}

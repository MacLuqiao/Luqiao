package luqiao;

import org.opencv.imgproc.Imgproc;
import org.opencv.core.*;
import org.opencv.core.Rect;


import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class GMM {
    MySVM mySVM;
    Mat frame;
    Mat fgmask;
    Tracker tracker;
    public GMM(MySVM _mySVM, Mat _frame, Mat _fgmask, Tracker _tracker){
        mySVM = _mySVM;
        frame = _frame;
        fgmask = _fgmask;
        tracker = _tracker;
    }
    public List<DetectBox> fg_box(){
        // GMM
        int width = fgmask.cols(), height = fgmask.rows();
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                double[] clone = fgmask.get(i, j).clone();
                if(clone[0] == 127) {
                    clone[0] = 0;
                    fgmask.put(i, j, clone);
                }
            }
        }
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(fgmask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        List<DetectBox> contours_no_sb = new ArrayList();
        Iterator<MatOfPoint> iterator = contours.iterator();

        while (iterator.hasNext()) {
            MatOfPoint contour = iterator.next();
            double area = Imgproc.contourArea(contour);
            if (area < 400 || area > 1920.0 * 1080 / 4) continue;
            boolean spill_flag = true;
            Rect r = Imgproc.boundingRect(contour);
            DetectBox spi_box = new DetectBox(r.x,r.y,r.x + r.width, r.y + r.height);
            for (YoloObject yoloObject : tracker.track_box) {
                DetectBox box = yoloObject.detectBox;
                if (interact(box, spi_box)){
                    spill_flag = false;
                    break;
                }
            }
            if(spill_flag && cheg_neg(spi_box,frame)){
                contours_no_sb.add(spi_box);
            }
        }
        return contours_no_sb;
    }

    public Boolean cheg_neg(DetectBox box, Mat frame){
        int xmin = box.left, ymin = box.top, xmax = box.right, ymax = box.bottom;
        int w = Math.abs(xmax - xmin);
        int h = Math.abs(ymax - ymin);
        Mat img = new Mat();
        Imgproc.getRectSubPix(frame,new Size(w, h),new Point((xmin+xmax)/2, (ymin+ymax)/2),img);
        Mat t = img.clone();
        int p = mySVM.predict(mySVM.getHOGdescriptors(img));
//        Imgcodecs.imwrite("./c3/"+ccc+"-"+p+".jpg",t);
//        ccc++;
        return p==-1;
    }

    public Boolean interact(DetectBox box1, DetectBox box2){
        int xmin = Math.max(box1.left, box2.left);
        int ymin = Math.max(box1.top, box2.top);
        int xmax = Math.min(box1.right, box2.right);
        int ymax = Math.min(box1.bottom, box2.bottom);

        int w = Math.max(0, xmax - xmin);
        int h = Math.max(0, ymax - ymin);
        int area = w * h;
        if(area > 0)return true;
        return false;
    }
}

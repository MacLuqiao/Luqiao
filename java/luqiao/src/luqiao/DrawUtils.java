package luqiao;


import java.util.ArrayList;
import java.util.List;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

public class DrawUtils {
    public static boolean _DEBUGGER_ = true;
    public static Scalar redPaint = new Scalar(0, 0, 255);
    public static Scalar greenPaint = new Scalar(0, 255, 0);
    public static Scalar balckPaint = new Scalar(0, 0, 0);

    public static void drawBox(Mat img, DetectBox box, String accident, int track_id){
        int thickness = 3;
        int font = Imgproc.FONT_HERSHEY_SIMPLEX;
        int fontsize = 1;
        int[] baseline = new int[1];
        Size cat_size = Imgproc.getTextSize(accident, font, fontsize, thickness,baseline);
        if(accident.equals("jam"))
            Imgproc.putText(img, "jam", new Point(30,70), 2, 10, redPaint);
        else if(accident.equals("no")) {
            Imgproc.rectangle(img, new Point(box.left, box.top), new Point(box.right, box.bottom), greenPaint, thickness);
            Imgproc.rectangle(img, new Point(box.left, box.top-cat_size.height-thickness), new Point(box.left+cat_size.width+thickness, box.top), greenPaint, -1);
            if(_DEBUGGER_)
                Imgproc.putText(img, String.valueOf(track_id), new Point(box.left, box.top-thickness-1),0, 1, greenPaint, thickness);
            Imgproc.putText(img, String.valueOf(track_id), new Point(box.left, box.top-thickness-1), font, fontsize, new Scalar(0, 0, 0), 1, Imgproc.LINE_AA);
        }else {
            Imgproc.rectangle(img, new Point(box.left, box.top), new Point(box.right, box.bottom), redPaint, thickness);
            Imgproc.rectangle(img, new Point(box.left, box.top-cat_size.height-thickness), new Point(box.left+cat_size.width+thickness, box.top), redPaint, -1);
            String text = accident;
            if(_DEBUGGER_)
                text = track_id+":"+accident;
            Imgproc.putText(img, text, new Point(box.left, box.top-thickness-1), font, fontsize, new Scalar(0, 0, 0), 1, Imgproc.LINE_AA);
        }
    }

    public static void drawCrossLine(Mat img, String lineInforms){
        List<List<Integer>> cross_dict = new ArrayList<List<Integer>>();
        if(lineInforms.equals(""))
            return;
        String[] lines = lineInforms.split("-");
        for(String line:lines){
            String[] points = line.split(",");
            List<Integer> tempList = new ArrayList<Integer>();
            for(String point:points)
                tempList.add(Integer.parseInt(point.trim()));
            cross_dict.add(tempList);
        }
        for(List<Integer> line:cross_dict)
            Imgproc.line(img, new Point(line.get(0), line.get(1)), new Point(line.get(2), line.get(3)), redPaint, 4);
    }

    public static String eventNum2String(int num){
        String res = "";
        switch (num){
            case 0:
                res = "no";
                break;
            case 1:
                res = "park";
                break;
            case 2:
                res = "retrograde";
                break;
            case 3:
                res = "person";
                break;
            case 4:
                res = "spill";
                break;
            case 5:
                res = "jam";
                break;
            case 6:
                res = "cross";
                break;
        }
        return res;
    }
    public static void draw(Mat img, List<AccidentInform> accidentInforms, String lineInforms){
        for(AccidentInform accidentInform : accidentInforms)
            drawBox(img, accidentInform.detectBox, eventNum2String(accidentInform.accidentType), accidentInform.trackID);
//        drawCrossLine(img, lineInforms);
    }
}

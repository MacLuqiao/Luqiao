package luqiao;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
//luyao11.10//
import java.awt.geom.Point2D;
import java.awt.geom.GeneralPath;

public class Accident {
    // DEBUG MODE
    public static boolean _DEBUGGER_ = true;

    // Video params
    public static int video_w = 1920;
    public static int video_h = 1080;
    public static int frame_per_second = 5;

    // Each frame information
    static List<Map<Integer, DetectBox>> frame_info_list = new LinkedList<Map<Integer, DetectBox>>(); // 每帧检测框信息
    static int frame_info_list_maxsize = frame_per_second * 10; // 储存视频帧长度，默认10s

    // Object ID with accident
    static Set<Integer> accident_track_ids = new HashSet<Integer>();   // 发生事件的id集合

    // Park & Jam
    static int park_jam_time_thres = frame_per_second * 2;

    // Jam
    static boolean jam_switch = true;                          // 拥堵事件检测开关
    static int jam_frame_count_init = frame_per_second * 8;     //
    static int jam_frame_count = 0;                             // 拥堵帧数
    static int jam_car_count_thres = 4;                         // 拥堵事件的车辆数量判断阈值
    static double jam_pos_thres;                                // 拥堵事件车辆位移阈值
    static double jam_pos_factor = 1.5;                           // 位移计算因子
    static double jam_box_area_min = 4000;                      // 拥堵最小box面积
    static boolean jam_flag = false;                            // 判断当前帧是否为拥堵帧
    static boolean jam_pre_flag = false;                        // 判断前一帧是否为拥堵帧
    static boolean jam_print_count = true;                      // 显示车辆数

    // Person
    static boolean person_switch = true;
    static double person_conf_thres = 0.8;                      // 行人置信度阈值

    //luyao11.10//
    // Spill
//    static List<Point2D.Double> region = new ArrayList<>();

    // Park
    static boolean park_switch = true;                           // 停车事件检测开关
    static double park_pos_factor = 0.1;                         // 位移计算因子
    static double park_pos_thres;                                // 停车事件车辆位移阈值
    static double park_box_area_min = 6000;                      // 停车最小box面积
    static int park_count_init = frame_per_second * 2;
    static Map<Integer, Boolean> park_flag = new HashMap();
    static Map<Integer, Integer> park_frame_count = new HashMap();

    // Cross
    static boolean cross_switch = false;                                             // 变道事件检测开关
    static List<List<Integer>> cross_dict = new ArrayList<List<Integer>>();         // 变道线边缘点
    static List<List<Double>> cross_k_b_list = new ArrayList<List<Double>>();       // 变道线斜率 截距
    static Map<Integer, Map<Integer, Set<String>>> cross_id = new HashMap<Integer, Map<Integer, Set<String>>>(); // 记录变道信息
    static int cross_pos_x_thres = 16;                                              // check_cross参数

    // Retrograde
    static boolean retrograde_switch = false;                                         // 逆行事件检测开关
    static Map<String, Map<String, List<List<Integer>>>> retrograde_dict = new HashMap<String, Map<String, List<List<Integer>>>>();   // 逆行线边缘点
    static Map<String, Map<String, List<List<Double>>>> retrograde_k_b_list = new HashMap<String, Map<String, List<List<Double>>>>(); // 逆行线斜率 截距
    static Map<Integer, DetectBox> retrograde_appear_info = new HashMap<Integer, DetectBox>(); // 每个id第一次出现的位置
    static int retrograde_pos_y_thres = 5;                                            // check_retrograde参数

    // Spill
    static boolean spill_switch = false;   // 抛洒物事件检测开关
    static List<MyPoint2D> detect_region = new ArrayList<MyPoint2D>();
    static List<MyPoint2D> prohibit_region = new ArrayList<MyPoint2D>();

    public Accident(){
        init_retrograde_line();
//        init_region();//luyao11.10//
    }

    static void init_detect_region(String lineInforms){

//        // 盒子205     XAT-99
//        MyPoint2D tl = new MyPoint2D(971, 118), tr = new MyPoint2D(1234, 115), bl = new MyPoint2D(1816, 730), br = new MyPoint2D(295, 746);

//        // 盒子66      XAT-107
//        MyPoint2D tl = new MyPoint2D(769, 148), tr = new MyPoint2D(1150, 148), bl = new MyPoint2D(1895, 888), br = new MyPoint2D(12, 846);
//
//        // 盒子65      HCT-26
//        MyPoint2D tl = new MyPoint2D(1408, 141), tr = new MyPoint2D(1746, 285), bl = new MyPoint2D(1718, 1025), br = new MyPoint2D(3, 829);

        // 已经初始化，则返回
        if(!detect_region.isEmpty())
            return ;

        // 默认值 XAT-61
        if(lineInforms.isEmpty())
            lineInforms = "230,413-909,406-1587,591-1721,876-269,900";

        String[] lines = lineInforms.split("-");
        for(String line:lines){
            String[] points = line.split(",");
            MyPoint2D point = new MyPoint2D(Double.parseDouble(points[0]), Double.parseDouble(points[1]));
            detect_region.add(point);
        }
    }

    static void init_prohibit_region(String lineInforms){
        // 已经初始化，则返回
        if(!prohibit_region.isEmpty())
            return ;

        // 默认值 XAT-61
        if(lineInforms.isEmpty())
            lineInforms = "76,326-206,326-207,737-79,739";

        String[] lines = lineInforms.split("-");
        for(String line:lines){
            String[] points = line.split(",");
            MyPoint2D point = new MyPoint2D(Double.parseDouble(points[0]), Double.parseDouble(points[1]));
            prohibit_region.add(point);
        }
    }

    static void init_cross_line(String lineInforms){
        // 已经初始化，则返回
        if(!cross_dict.isEmpty())
            return;

        // 默认值 XAT-61
        if(lineInforms.isEmpty())
            lineInforms = "245,957,1186,277-1186,277,1282,181-672,994,1303,319-1303,319,1377,185";

        String[] lines = lineInforms.split("-");
        for(String line:lines){
            String[] points = line.split(",");
            List<Integer> tempList = new ArrayList<Integer>();
            for(String point:points)
                tempList.add(Integer.parseInt(point.trim()));
            cross_dict.add(tempList);
        }
        for(List tup : cross_dict) {
            Double k = (Double.valueOf(tup.get(3).toString()) - Double.valueOf(tup.get(1).toString())) / (Double.valueOf(tup.get(2).toString()) - Double.valueOf(tup.get(0).toString()));
            Double b = Double.valueOf(tup.get(1).toString()) - Double.valueOf(tup.get(0).toString()) * k;
            List<Double> list = new ArrayList<Double>();
            list.add(k);
            list.add(b);
            cross_k_b_list.add(list);
        }
    }

    static void init_retrograde_line(){
        // create video retrograde lines points
        List<Integer> tempList1 = Arrays.asList(77, 1080, 1167, 300);
        List<Integer> tempList2 = Arrays.asList(1167, 300, 1283, 185);
        List<Integer> tempList3 = Arrays.asList(592, 1080, 1302, 333);
        List<Integer> tempList4 = Arrays.asList(1302, 333, 1383, 185);
        List<List<Integer>> point_forward_left_list = new ArrayList<List<Integer>>();
        List<List<Integer>> point_forward_right_list = new ArrayList<List<Integer>>();
        List<List<Integer>> point_backward_left_list = new ArrayList<List<Integer>>();
        List<List<Integer>> point_backward_right_list = new ArrayList<List<Integer>>();
        point_forward_left_list.add(tempList1);
        point_forward_left_list.add(tempList2);
        point_forward_right_list.add(tempList3);
        point_forward_right_list.add(tempList4);
        Map<String, List<List<Integer>>> point_forward_map = new HashMap<String, List<List<Integer>>>();
        Map<String, List<List<Integer>>> point_backward_map = new HashMap<String, List<List<Integer>>>();
        point_forward_map.put("left", point_forward_left_list);
        point_forward_map.put("right", point_forward_right_list);
        point_backward_map.put("left", point_backward_left_list);
        point_backward_map.put("right", point_backward_right_list);
        retrograde_dict.put("FORWARD", point_forward_map);
        retrograde_dict.put("BACKWARD", point_backward_map);

        // create video retrograde lines k and b
        List<List<Double>> forward_left_list = new ArrayList<List<Double>>();
        List<List<Double>> forward_right_list = new ArrayList<List<Double>>();
        List<List<Double>> backward_left_list = new ArrayList<List<Double>>();
        List<List<Double>> backward_right_list = new ArrayList<List<Double>>();

        Map<String, List<List<Double>>> forward_map = new HashMap<String, List<List<Double>>>();
        Map<String, List<List<Double>>> backward_map = new HashMap<String, List<List<Double>>>();
        forward_map.put("left", forward_left_list);
        forward_map.put("right", forward_right_list);
        backward_map.put("left", backward_left_list);
        backward_map.put("right", backward_right_list);
        retrograde_k_b_list.put("FORWARD", forward_map);
        retrograde_k_b_list.put("BACKWARD", backward_map);
        for(String direction : retrograde_dict.keySet()) {
            for(String side : retrograde_dict.get(direction).keySet()) {
                for(List tup : retrograde_dict.get(direction).get(side)) {
                    Double k = (Double.valueOf(tup.get(3).toString()) - Double.valueOf(tup.get(1).toString())) / (Double.valueOf(tup.get(2).toString()) - Double.valueOf(tup.get(0).toString()));
                    Double b = Double.valueOf(tup.get(1).toString()) - Double.valueOf(tup.get(0).toString()) * k;
                    List<Double> list = new ArrayList<Double>();
                    list.add(k);
                    list.add(b);
                    retrograde_k_b_list.get(direction).get(side).add(list);
                }
            }
        }
    }

    // 同时检测park & jam
    static Map<String, Boolean> check_park_jam(DetectBox frame_info_p1, DetectBox frame_info_p2) {
        double box_area = (frame_info_p1.right - frame_info_p1.left) * (frame_info_p1.bottom - frame_info_p1.top);
        Map<String, Boolean> park_jam = new HashMap<>();
        if(box_area < park_box_area_min) {
            park_jam.put("park", false);
        }
        else{
            park_pos_thres = park_pos_factor * Math.min(frame_info_p1.right-frame_info_p1.left, frame_info_p1.bottom-frame_info_p1.top);
            park_jam.put("park", Math.pow(((Math.pow(frame_info_p1.right-frame_info_p2.right, 2)) + Math.pow(frame_info_p1.bottom-frame_info_p2.bottom, 2)), 0.5) <= park_pos_thres);
        }
        if(box_area < jam_box_area_min){
            park_jam.put("jam", false);
        }
        else {
            jam_pos_thres = jam_pos_factor * Math.min(frame_info_p1.right-frame_info_p1.left, frame_info_p1.bottom-frame_info_p1.top);
            park_jam.put("jam", Math.pow(((Math.pow(frame_info_p1.right-frame_info_p2.right, 2)) + Math.pow(frame_info_p1.bottom-frame_info_p2.bottom, 2)), 0.5) <= jam_pos_thres);
        }

        return park_jam;
    }

    static boolean check_cross(DetectBox box, int track_id){
        Set<String> points = new HashSet<String>();
        points.add("boxCenter");
        points.add("bottomCenter");
        for(String point : points){
            double p_x = box.center_x;
            double p_y = point == "boxCenter" ? box.center_y : box.bottom;
            for(int i = 0; i < cross_k_b_list.size(); i++) {
                if(p_y >= cross_dict.get(i).get(1) || p_y <= cross_dict.get(i).get(3))
                    continue;
                // 如果压线，则加入map中
                if((Math.abs((p_y - cross_k_b_list.get(i).get(1)) / cross_k_b_list.get(i).get(0) - p_x)) <= cross_pos_x_thres - (video_h - p_y)*0.01){
                    // 如果还未记录过当前id信息，则新增
                    if(!cross_id.containsKey(track_id))
                        cross_id.put(track_id, new  HashMap<Integer, Set<String>>());
                    Map<Integer, Set<String>> cur_id_line_inform = cross_id.get(track_id);
                    // 如果不包含当前线，则新增
                    if(!cur_id_line_inform.containsKey(i))
                        cur_id_line_inform.put(i, new HashSet<String>());
                    // 添加当前点到对应线信息中
                    cur_id_line_inform.get(i).add(point);
                    // 如果压了两个点，则说明违章变道
                    if(cur_id_line_inform.get(i).size() == 2)
                        return true;
                }
            }
        }
        return false;
    }

    static boolean check_retrograde(DetectBox frame_info_p1, DetectBox frame_info_p2) {
        boolean flag = false;
        String direction;
        List<Double> now;
        double p1_x = (double)(frame_info_p1.left + frame_info_p1.right) / 2;
        double p1_y = (double)(frame_info_p1.top + frame_info_p1.bottom) / 2;
        double p2_x = (double)(frame_info_p2.left + frame_info_p2.right) / 2;
        double p2_y = (double)(frame_info_p2.top + frame_info_p2.bottom) / 2;
        if(Math.abs(p2_y - p1_y) >= retrograde_pos_y_thres) {
            direction = (p2_y - p1_y >= 0) ? "BACKWARD" : "FORWARD";
            for(int i  = 0; i < retrograde_k_b_list.get(direction).get("left").size(); i++) {
                now = retrograde_k_b_list.get(direction).get("left").get(i);
                if(p1_x >= ((p1_y - now.get(1)) / now.get(0))) {
                    flag = true;
                    break;
                }
            }
            if(flag) {
                for (int i = 0; i < retrograde_k_b_list.get(direction).get("right").size(); i++) {
                    now = retrograde_k_b_list.get(direction).get("right").get(i);
                    if (p2_x <= ((p2_y - now.get(1)) / now.get(0)))
                        return true;
                }
            }
        }
        return false;
    }

    static boolean check_box_in_region(DetectBox sp_box, String type){
        List<MyPoint2D> region = null;
        if(type.equals("detect"))
            region = detect_region;
        else if(type.equals("prohibit"))
            region = prohibit_region;
        else
            System.out.println("Wrong region type!");
        double p_x = (sp_box.left + sp_box.right) / 2.0;
        double p_y = (sp_box.top + sp_box.bottom) / 2.0;
        // 将要判断的横纵坐标组成一个点
        MyPoint2D point = new MyPoint2D(p_x, p_y);
        int N = region.size();
        boolean boundOrVertex = true; //如果点位于多边形的顶点或边上，也算做点在多边形内，直接返回true
        int intersectCount = 0;//cross points count of x
        double precision = 2e-10; //浮点类型计算时候与0比较时候的容差
        MyPoint2D p1, p2;//neighbour bound vertices
        MyPoint2D p = point; //当前点

        p1 = region.get(0);//left vertex
        for(int i = 1; i <= N; ++i){//check all rays
            if(p.equals(p1)){
                return boundOrVertex;//p is an vertex
            }

            p2 = region.get(i % N);//right vertex
            if(p.x < Math.min(p1.x, p2.x) || p.x > Math.max(p1.x, p2.x)){//ray is outside of our interests
                p1 = p2;
                continue;//next ray left point
            }

            if(p.x > Math.min(p1.x, p2.x) && p.x < Math.max(p1.x, p2.x)){//ray is crossing over by the algorithm (common part of)
                if(p.y <= Math.max(p1.y, p2.y)){//x is before of ray
                    if(p1.x == p2.x && p.y >= Math.min(p1.y, p2.y)){//overlies on a horizontal ray
                        return boundOrVertex;
                    }

                    if(p1.y == p2.y){//ray is vertical
                        if(p1.y == p.y){//overlies on a vertical ray
                            return boundOrVertex;
                        }else{//before ray
                            ++intersectCount;
                        }
                    }else{//cross point on the left side
                        double xinters = (p.x - p1.x) * (p2.y - p1.y) / (p2.x - p1.x) + p1.y;//cross point of y
                        if(Math.abs(p.y - xinters) < precision){//overlies on a ray
                            return boundOrVertex;
                        }

                        if(p.y < xinters){//before ray
                            ++intersectCount;
                        }
                    }
                }
            }else{//special case when ray is crossing through the vertex
                if(p.x == p2.x && p.y <= p2.y){//p crossing over p2
                    MyPoint2D p3 = region.get((i+1) % N); //next vertex
                    if(p.x >= Math.min(p1.x, p3.x) && p.x <= Math.max(p1.x, p3.x)){//p.x lies between p1.x & p3.x
                        ++intersectCount;
                    }else{
                        intersectCount += 2;
                    }
                }
            }
            p1 = p2;//next ray left point
        }

        if(intersectCount % 2 == 0){//偶数在多边形外
            return false;
        } else { //奇数在多边形内
            return true;
        }

    }

    static void reset_retrograde_params(){

    }

    public static List<AccidentInform> checkAccident(List<YoloObject> yoloObjects, List<DetectBox> spi_box, String detectRegion, String prohibitRegion, String crossLines) {
//    public static List<AccidentInform> checkAccident(List<YoloObject> yoloObjects, String lineInforms) {
        // 初始化车道线和区域信息
        init_cross_line(crossLines);
        init_detect_region(detectRegion);
        init_prohibit_region(prohibitRegion);
        // 当前帧的违章事件列表
        List<AccidentInform> accidentInforms = new ArrayList<AccidentInform>();
        // 当前帧的检测框信息 不含行人
        Map<Integer, DetectBox> cur_frame_info = new HashMap<Integer, DetectBox>();

        // Jam
        int jam_car_count = 0;            // 当前帧车辆数
        jam_pre_flag = jam_flag;

        // Traverse each object in current frame
        for(YoloObject yoloObject : yoloObjects) {
            DetectBox box = yoloObject.detectBox;
            double score = yoloObject.score;
            int cls = yoloObject.label;
            int tracking_id = yoloObject.tracking_id;

            // 禁止检测区域
            if (check_box_in_region(box, "prohibit"))
                continue;

            // Person
            if((cls == 0 || cls == 1)) {
                if(!person_switch)
                    continue;
                if(score >= person_conf_thres) {
                    if(_DEBUGGER_)      // Debug
                        accidentInforms.add(new AccidentInform(3, tracking_id, box));
                    else {              // Normal
                        if(!accident_track_ids.contains(tracking_id)) {
                            accident_track_ids.add(tracking_id);
                            accidentInforms.add(new AccidentInform(3, tracking_id, box));
                        }
                    }
                }
                continue;
            }

            // 记录当前box信息
            cur_frame_info.put(tracking_id, box);
            // 判断当前box是否为事件
            boolean isAccident = false;

            // Jam & Park
            // 在2s前的缓存帧中寻找相同id车判断位移是否足够小，满足则（1）拥堵车数量增加（2）判断当前为停车
            boolean park_no_jump = true;
            boolean jam_no_jump = true; // jam_count 只加一次
            if(!park_flag.containsKey(tracking_id))
                park_flag.put(tracking_id, false);
            if(jam_switch && park_switch && frame_info_list.size() >= park_jam_time_thres){
                for(int i = 0; i < (frame_info_list.size() - park_jam_time_thres); i++) {
                    if(frame_info_list.get(i).containsKey(tracking_id)){
                        Map<String, Boolean> park_jam = check_park_jam(box, frame_info_list.get(i).get(tracking_id));
                        if(park_jam.get("park")){
                            park_flag.put(tracking_id, true);
                            park_no_jump = false;
                            // 如果不是拥堵状态，则为停车
                            if(!jam_flag) {
                                isAccident = true;
                                if(!accident_track_ids.contains(tracking_id)) {
                                    accident_track_ids.add(tracking_id);
                                    accidentInforms.add(new AccidentInform(1, tracking_id, box));
                                }
                            }
                            if(!park_frame_count.containsKey(tracking_id))
                                park_frame_count.put(tracking_id, park_count_init);
                        }
                        if(park_jam.get("jam") && jam_no_jump){
                            jam_car_count++;
                            jam_no_jump = false;
                        }
                        if(!park_no_jump && !jam_no_jump)
                            break;
                    }
                }
            }

            // Cross
            if(cross_switch) {
                // 如果两个关键点压到相同的线，则变道
                if(check_cross(box, tracking_id)){
                    isAccident = true;
                    if(_DEBUGGER_)      // Debug
                        accidentInforms.add(new AccidentInform(6, tracking_id, box));
                    else {              // Normal
                        if(!accident_track_ids.contains(tracking_id)) {
                            accident_track_ids.add(tracking_id);
                            accidentInforms.add(new AccidentInform(6, tracking_id, box));
                        }
                    }
                }
            }

            // Retrograde
            if(retrograde_switch){
                if((retrograde_appear_info.containsKey(tracking_id) && check_retrograde(box, retrograde_appear_info.get(tracking_id)))) {
                    isAccident = true;
                    if(_DEBUGGER_)      // Debug
                        accidentInforms.add(new AccidentInform(2, tracking_id, box));
                    else {              // Normal
                        if(!accident_track_ids.contains(tracking_id)) {
                            accident_track_ids.add(tracking_id);
                            accidentInforms.add(new AccidentInform(2, tracking_id, box));
                        }
                    }
                }
                else if(accident_track_ids.contains(tracking_id))
                    accident_track_ids.remove(tracking_id);
                // 记入每辆车第一次出现的位置
                if(!retrograde_appear_info.containsKey(tracking_id))
                    retrograde_appear_info.put(tracking_id, box);
                reset_retrograde_params();
            }

            // No accident box
            if(!isAccident && _DEBUGGER_)      // Debug
                accidentInforms.add(new AccidentInform(0, tracking_id, box));
        }

        // Save frame detected box informs
        if(frame_info_list.size() >= frame_info_list_maxsize)
            frame_info_list.remove(0);
        frame_info_list.add(cur_frame_info);

        // Park
        for(Integer key: park_frame_count.keySet()){
            if(park_frame_count.get(key)>0){
                int value = park_frame_count.get(key) - 1;
                park_frame_count.put(key, value);
            }
        }
        for(Integer key: park_frame_count.keySet()){
            if(park_frame_count.get(key)==0)
                park_flag.put(key, false);
        }

        // Jam
        if(jam_switch){
            if(jam_print_count)
                System.out.printf("jam_car_count: %d \n", jam_car_count);
            // 如果拥堵车辆大于阈值，则画框，设置当前帧为拥堵帧
            if(jam_car_count >= jam_car_count_thres) {
                //add_accident_bbox(image,box_,0,"jam");
                jam_flag = true;
                jam_frame_count = jam_frame_count_init;
            }
            // 如果拥堵车辆数达到阈值，则判断为停车事件
            if(jam_frame_count > 0) {
                //add_accident_bbox(image,box_,0,"jam");
                jam_flag = true;
                jam_frame_count --;
            }
//            🤗
            else
                jam_flag = false;
            // 如果是拥堵事件起始帧，则返回事件
            if(!jam_pre_flag && jam_flag){
                accidentInforms.add(new AccidentInform(5, -1, new DetectBox()));
                System.out.println("55555检测到拥堵物事件！！！");
            }

        }

        // Spill
        if(spill_switch){
            for (DetectBox s_box : spi_box)
                if(check_box_in_region(s_box, "detect")){
                    accidentInforms.add(new AccidentInform(4, -1, s_box));
                    System.out.println("44444检测到抛洒物事件！！！");
//                    System.exit(0);
                }

        }

        return accidentInforms;
    }

    static void reset() {
        accident_track_ids.clear();
        park_frame_count.clear();
        park_flag.clear();
        cross_id.clear();
    }
}
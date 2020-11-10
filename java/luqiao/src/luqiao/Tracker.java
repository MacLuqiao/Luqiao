package luqiao;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Tracker {
    public int id_count = 1;                                          // 初始id
    public static int id_upper_limit = 100000;                        // id数上限
    public List<YoloObject> track_box = new ArrayList<YoloObject>();  // 上一帧的检测框跟踪id

    public Tracker(){
        track_box.clear();
    }

    // 计算框距离
    int cal_dist(YoloObject a, YoloObject b){
        return (int)((a.detectBox.center_x - b.detectBox.center_x) * (a.detectBox.center_x - b.detectBox.center_x) +
                (a.detectBox.center_y - b.detectBox.center_y) * (a.detectBox.center_y - b.detectBox.center_y));
    }

    // 更新跟踪id
    public void step(List<YoloObject> detect_box, Accident accident) {
        List<YoloObject> new_box = new ArrayList<YoloObject>();
        List<Integer> cur_frame_box_size = new ArrayList<Integer>();  // 当前帧每个检测框大小列表
        List<Integer> last_frame_box_size= new ArrayList<Integer>();  // 上一帧每个检测框大小列表
        Set<Integer> matched_id = new HashSet<Integer>();             // 已匹配的上一帧id

        // 计算当前帧检测框大小
        for(int i=0; i<detect_box.size(); i++)
            cur_frame_box_size.add((detect_box.get(i).detectBox.right-detect_box.get(i).detectBox.left) * (detect_box.get(i).detectBox.bottom-detect_box.get(i).detectBox.top));
        // 计算上一帧检测框大小
        for(int i=0; i<track_box.size(); i++)
            last_frame_box_size.add((track_box.get(i).detectBox.right-track_box.get(i).detectBox.left) * (track_box.get(i).detectBox.bottom-track_box.get(i).detectBox.top));
        // 匹配当前帧和上一帧的所有检测框
        for(int i=0; i<detect_box.size(); i++){
            int min_dist = Integer.MAX_VALUE;
            int trk_index = -1;
            for(int j=0; j<track_box.size(); j++){
                if(matched_id.contains((track_box.get(j).tracking_id)))  // matched
                    continue;
                int cur_dist = cal_dist(detect_box.get(i), track_box.get(j));
                if (cur_dist < min_dist && cur_dist < 1.2 * cur_frame_box_size.get(i) && cur_dist < 1.2 * last_frame_box_size.get(j)){
                    min_dist = cur_dist;
                    trk_index = j;
                }
            }
            if (trk_index != -1){    // matched
                detect_box.get(i).tracking_id = track_box.get(trk_index).tracking_id;
                matched_id.add(detect_box.get(i).tracking_id);
            }
            else                    // unmatched
                detect_box.get(i).tracking_id = id_count++;
            if(id_count >= id_upper_limit){
                id_count = 1;
                accident.reset();
            }
            new_box.add(detect_box.get(i));
        }

        // 清空上一帧跟踪id
        track_box.clear();
        track_box.addAll(new_box);

        // Log.d("--------->total id", String.valueOf(id_count));
    }

}


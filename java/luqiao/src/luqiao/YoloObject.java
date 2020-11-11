package luqiao;

public class YoloObject {
    int label;
    double score;
    DetectBox detectBox;
    int tracking_id;
    public YoloObject(int _label, double _score, DetectBox _detectBox) {
        this.label = _label;
        this.score = _score;
        this.detectBox = _detectBox;
        this.tracking_id = -1;
    }
}


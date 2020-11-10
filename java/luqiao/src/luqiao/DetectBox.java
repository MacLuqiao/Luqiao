package luqiao;

public class DetectBox {
    double center_x;
    double center_y;
    int left;
    int top;
    int right;
    int bottom;
    public DetectBox(int _left, int _top, int _right, int _bottom) {
        left = _left;
        top = _top;
        right = _right;
        bottom = _bottom;
        center_x = (left + right) / 2.0;
        center_y = (top + bottom) / 2.0;
    }

    public DetectBox(){
        center_x = center_y = left = top = right = bottom = 0;
    }
}

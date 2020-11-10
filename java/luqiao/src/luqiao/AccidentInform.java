package luqiao;

public class AccidentInform {
    int accidentType;
    int trackID;
    DetectBox detectBox;
    public AccidentInform(int _accidentType, int _trackID,  DetectBox _detectBox){
        accidentType = _accidentType;
        trackID = _trackID;
        detectBox = _detectBox;
    }
}

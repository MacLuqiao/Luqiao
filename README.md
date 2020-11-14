# Luqiao

## 版本信息描述

### Java

- 拥堵和停车事件合并处理
- 抛洒物还检测不出任何玩意儿
- Accident.java中增加了prohibitRegion，以排除广告牌干扰，抛洒物检测区域的变量名改为detectRegion

### Python

- yolov3 + Centertrack
- GMM检测抛洒物（带region)

### 停车/拥堵逻辑
- 当某ID车在设定时间阈值内移动距离<设定的移动阈值（车身距离 x factor）时，则判定该ID车为停车或拥堵计数+1，当拥堵计数大于阈值时，设定该帧为拥堵事件
- 停车和拥堵事件设定延迟时间，当发生拥堵事件时会屏蔽停车事件的检测

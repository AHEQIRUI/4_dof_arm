# dofbot_arm

6-DOF 机械臂控制系统（可看作4自由度），基于 Raspberry Pi，使用 YOLOv8 进行视觉引导抓取。

## 硬件

- Raspberry Pi
- 6-DOF 机械臂（6个总线舵机）
- 相机（640x480）

## 软件架构

```
dofbot_arm/
├── grasp_yolo.py          # 主程序：YOLO检测 + 机械臂抓取
├── src/
│   ├── ik.py             # 逆/正运动学
│   └── chessboard_calibration.py  # 相机标定
├── Arm_Lib/              # 舵机I2C控制库
├── config/               # 相机标定参数
└── models/               # YOLO模型
```


## 使用方法

### 相机标定

```bash
# 1. 采集棋盘格图像（建议40张）
python src/chessboard_calibration.py --mode capture --num 40

# 2. 计算相机内参
python src/chessboard_calibration.py --mode calibrate
```

### 运行抓取程序

```bash
python grasp_yolo.py --model models/best.pt
```

### 交互操作

- `d` - 检测物体
- `g` - 抓取选中目标
- `q` - 退出
- 鼠标点击 - 选中检测目标

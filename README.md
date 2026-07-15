# Yahboom Dofbot Pi

6-DOF 机械臂控制系统，基于 Raspberry Pi，使用 YOLOv8 进行视觉引导抓取。

# Demo

![image](https://github.com/AHEQIRUI/4_dof_arm/blob/master/demo/demo.gif)

## 硬件

- Raspberry Pi
- Yahboom Dofbot Pi
- USB Camera

## 软件架构

```
dofbot_arm/
├── grasp_yolo.py          # 主程序：YOLO检测 + 机械臂抓取
├── src/
│   ├── ik.py             # 逆/正运动学
│   └── chessboard_calibration.py  # 相机内参标定
├── Arm_Lib/              # 舵机I2C控制库
├── config/               # 相机标定参数
└── models/               # YOLO模型
```

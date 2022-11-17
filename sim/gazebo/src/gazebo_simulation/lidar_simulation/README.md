# README





​																																																																		2020-11-13  欧阳俊源

## 简述

本gazebo仿真包基于[velodyne_simulator](https://bitbucket.org/DataspeedInc/velodyne_simulator.git)，实现非均匀激光雷达的仿真并发布ROS-Topic。

实现方法：通过向一个model添加多个ray-sensor 然后由编写的model的plugin来控制接受多个ray-sensor的gazebo信息，并合并转发到ROS的topic下。



## 使用

给出了[example-launch](./velodyne_description/launch/example.launch)和[example-xacro](./velodyne_description/urdf/example.urdf.xacro)。在本ws下source即可运行。



## 如何修改

### 修改话题名或坐标系

launch：23-24lines

```xmal
  <arg name="LiDARName" default="LiDAR_non-uniform"/>
  <arg name="LiDARFrame" default="LiDARFrame"/>
```



### 多传感器

本example-launch和example-xacro 只实现了单个传感器的使用。后续实现





### 修改线束间距和个数

[example-xacro](./velodyne_description/urdf/LiDAR-80.xacro):103-105lins:

模仿下面书写即可。

目前请不要修改`name="sensor-sector" ` 

且按线束分块从上到下填写`suffix="*"`  

其中`samples`是水平的线数,希望保持相同 

`v_max_angle和v_min_angle`是垂直线束扇区的上和下界限夹角，水平方向为0°



```xaml
 <xacro:LiDARSensorGen name="sensor-sector" suffix="1" hz="${hz}" samples="${samples}" lasers="5+1"  v_max_angle="${5*M_PI/180.0}"     v_min_angle="${0*M_PI/180.0}"/>
      <xacro:LiDARSensorGen name="sensor-sector" suffix="2" hz="${hz}" samples="${samples}" lasers="25-2" v_max_angle="${-1/3*M_PI/180.0}"  v_min_angle="-${24/3*M_PI/180.0}"/>
      <xacro:LiDARSensorGen name="sensor-sector" suffix="3" hz="${hz}" samples="${samples}" lasers="11"   v_max_angle="-${25/3*M_PI/180.0}" v_min_angle="-${(25/3+10)*M_PI/180.0}"/>
    
```



注意事项1：被依附的link必须具有inertial属性，否则有被sdf合并优化掉。


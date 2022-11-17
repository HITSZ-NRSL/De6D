//
// Created by ou on 2021/6/20.
//

#ifndef VELODYNE_GAZEBO_PLUGINS_GAZEBO_ROS_TRIGGERED_LIDAR_H
#define VELODYNE_GAZEBO_PLUGINS_GAZEBO_ROS_TRIGGERED_LIDAR_H

#include "gazebo_ros_lidar_utils.h"

#if GAZEBO_GPU_RAY
#define GazeboRosVelodyneLaser GazeboRosVelodyneGpuLaser
#define RayPlugin GpuRayPlugin
#define RaySensorPtr GpuRaySensorPtr
#define RaySensor GpuRaySensor
#define STR_Gpu  "Gpu"
#define STR_GPU_ "GPU "
#else

#define STR_Gpu  ""
#define STR_GPU_ ""
#endif
typedef struct {
    double angle[2];
    int count;
    double pixel;
    int step;
} SECTOR;
namespace gazebo {
    class GazeboRosTriggeredLidar : public RayPlugin, GazeboRosLidarUtils {
    public:
        GazeboRosTriggeredLidar() = default;

        ~GazeboRosTriggeredLidar() = default;

        void Load(sensors::SensorPtr _parent, sdf::ElementPtr _sdf);

        virtual void TriggerLidar();

        virtual bool CanTriggerLidar();

        void SetLidarEnabled(const bool _enabled);

        event::ConnectionPtr updateConnection;

    protected:
        void OnUpdate();
        void OnNewLaserScans();
        std::vector<SECTOR> sectors_;
        std::vector<bool> map_;
        int triggered = 0;
        std::mutex mutex;

    };
}
#endif //VELODYNE_GAZEBO_PLUGINS_GAZEBO_ROS_TRIGGERED_LIDAR_H

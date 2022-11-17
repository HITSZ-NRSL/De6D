//
// Created by ou on 2021/1/25.
//
#ifndef VELODYNE_GAZEBO_PLUGINS_GAZEBOROSCUSTOMLIDAR_H
#define VELODYNE_GAZEBO_PLUGINS_GAZEBOROSCUSTOMLIDAR_H
#ifndef GAZEBO_GPU_RAY
#define GAZEBO_GPU_RAY 0
#endif

//
// Created by ou on 2020/11/12.
//
#include <functional>
#include <sdf/Param.hh>

#include <gazebo/physics/physics.hh>
#include <gazebo/transport/TransportTypes.hh>
#include <gazebo/transport/Node.hh>
#include <gazebo/msgs/MessageTypes.hh>
#include <gazebo/common/Time.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/sensors/sensors.hh>
#include <gazebo/sensors/SensorTypes.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/msgs/laserscan_stamped.pb.h>

#if GAZEBO_GPU_RAY
#include <gazebo/plugins/GpuRayPlugin.hh>
#else

#include <gazebo/plugins/RayPlugin.hh>

#endif

#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <ros/advertise_options.h>
#include <sensor_msgs/PointCloud2.h>

#include <boost/algorithm/string/trim.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_guard.hpp>

#if GAZEBO_GPU_RAY
#define GazeboRosVelodyneLaser GazeboRosVelodyneGpuLaser
#define RayPlugin GpuRayPlugin
#define RaySensorPtr GpuRaySensorPtr
#endif

using namespace std;
typedef struct {
    double angle[2];
    int count;
    double pixel;
    int step;
} SECTOR;
namespace gazebo {
    class GazeboRosCustomizedLiDAR : public RayPlugin {
    public:
        GazeboRosCustomizedLiDAR();

        ~GazeboRosCustomizedLiDAR();

        void Load(sensors::SensorPtr _parent, sdf::ElementPtr _sdf);

    private:  // for gazebo
        sensors::RaySensorPtr ray_sensor_;

        // for gazebo communication
        gazebo::transport::NodePtr gazebo_node_;
        gazebo::transport::SubscriberPtr gazebo_sub_;

        void OnScan(const ConstLaserScanStampedPtr &_msg);

        // input parameter about gazebo lidar
        string frame_name_;
        double min_intensity_ = 0;
        double min_range_ = 0;
        double max_range_ = 0;
        vector<SECTOR> sectors_;
        vector<bool> map_;

    private:  // for ros
        ros::NodeHandle *nh_;

        // for ros communication
        ros::Publisher ros_pub_;
        ros::Subscriber trigger_subscriber_;
        sensor_msgs::PointCloud2 msg;
        ros::CallbackQueue laser_queue_;

        void ConnectCb();

        // input parameter about ros
        string ns;
        string lidar_topic_name_;
        string trigger_topic_name_;

    private:  // for lidar-data queue thread
        void laserQueueThread();

        boost::thread callback_laser_queue_thread_;
        boost::mutex lock_;
    };
}
#endif //VELODYNE_GAZEBO_PLUGINS_GAZEBOROSCUSTOMLIDAR_H

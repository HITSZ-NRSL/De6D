//
// Created by ou on 2021/6/20.
//

#ifndef VELODYNE_GAZEBO_PLUGINS_GAZEBO_ROS_LIDAR_UTILS_H
#define VELODYNE_GAZEBO_PLUGINS_GAZEBO_ROS_LIDAR_UTILS_H

// boost stuff
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/algorithm/string.hpp>

// ros stuff
#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <ros/advertise_options.h>

// ros messages stuff
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Empty.h>

// Gazebo stuff
#include <gazebo/physics/physics.hh>
#include <gazebo/sensors/RaySensor.hh>
#include <gazebo/sensors/SensorTypes.hh>
#include <gazebo/plugins/CameraPlugin.hh>
#include <gazebo/plugins/GpuRayPlugin.hh>
#include <gazebo/plugins/RayPlugin.hh>

namespace gazebo {
    class GazeboRosTriggeredLidar;
    class GazeboRosLidarUtils {
    public:
        GazeboRosLidarUtils();

        ~GazeboRosLidarUtils();

        void Load(sensors::SensorPtr _parent, sdf::ElementPtr _sdf);

    private:
        void LoadRos();

        void Init();

    protected:
        sensors::RaySensorPtr parentSensor_;
        physics::WorldPtr world_;
        sdf::ElementPtr sdf_;
        std::string lidar_name_;
    protected:
        bool was_active_;
        int lidar_connect_count_;
        boost::mutex lidar_connect_count_lock_;

        void LidarConnect();

        void LidarDisconnect();

    protected:
        ros::NodeHandle *nh_;
        ros::Publisher pointcloud2_pub_;

        sensor_msgs::PointCloud2 pointcloud2_msg_;

        std::string lidar_topic_name_;
        std::string robot_namespace_;
        std::string tf_prefix_;
        std::string frame_name_;

        double min_range_;
        double max_range_;
    protected:
        virtual void TriggerLidar();

        virtual bool CanTriggerLidar();

    private:
        void TriggerCameraInternal(const std_msgs::Empty::ConstPtr &dummy);

        ros::Subscriber trigger_subscriber_;
        std::string trigger_topic_name_;

    protected:
        boost::mutex lock_;
        ros::CallbackQueue lidar_queue_;

        void lidarQueueThread();

        boost::thread callback_queue_thread_;

    protected:
        bool initialized_;

    public:
        template<class T>
        inline void ReadSdfTag(std::string _tag_name, T &_output, T _defalut) {
            bool exists = !this->sdf_->HasElement(_tag_name);
            _output = exists ? this->sdf_->GetElement(_tag_name)->Get<T>() : _defalut;
            ROS_INFO_STREAM((exists ? "Read" : "Miss") << " Tag <" << _tag_name << ">, set to" << _output);
        }
        friend class GazeboRosTriggeredLidar;
    };
}


#endif //VELODYNE_GAZEBO_PLUGINS_GAZEBO_ROS_LIDAR_UTILS_H

//
// Created by ou on 2021/6/20.
//

#include "gazebo_ros_lidar_utils.h"
#include <tf/tf.h>
#include <tf/transform_listener.h>
namespace gazebo {
    GazeboRosLidarUtils::GazeboRosLidarUtils() {
        initialized_ = false;
    }

    GazeboRosLidarUtils::~GazeboRosLidarUtils() {
        this->parentSensor_->SetActive(false);
        this->nh_->shutdown();
        this->lidar_queue_.clear();
        this->lidar_queue_.disable();
        this->callback_queue_thread_.join();
        delete this->nh_;
    }

    void GazeboRosLidarUtils::Load(sensors::SensorPtr _parent, sdf::ElementPtr _sdf) {
        std::string world_name = _parent->WorldName();
        this->world_ = physics::get_world(world_name);
        this->parentSensor_ = std::dynamic_pointer_cast<sensors::RaySensor>(_parent);
        this->sdf_ = _sdf;

        ReadSdfTag("nameSpace", this->robot_namespace_, std::string(""));
        ReadSdfTag("topicName", this->lidar_topic_name_, std::string("/points"));
        ReadSdfTag("frameName", this->frame_name_, std::string("/world"));
        ReadSdfTag("lidarName", this->lidar_name_, this->lidar_topic_name_);
        ReadSdfTag("minRange", this->min_range_, DBL_MIN);
        ReadSdfTag("maxRange", this->max_range_, DBL_MAX);

        this->lidar_connect_count_ = 0;
        this->was_active_ = false;
        this->parentSensor_->SetActive(false);
        LoadRos();
    }

    void GazeboRosLidarUtils::LoadRos() {
        if (!ros::isInitialized()) {
            ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin. "
                                     << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
            return;
        }
        this->nh_ = new ros::NodeHandle(this->robot_namespace_);

        this->tf_prefix_ = tf::getPrefixParam(*this->nh_);
        if (this->tf_prefix_.empty()) {
            this->tf_prefix_ = this->robot_namespace_;
            boost::trim_right_if(this->tf_prefix_, boost::is_any_of("/"));
        }
        this->frame_name_ = tf::resolve(this->tf_prefix_, this->frame_name_);
        ROS_INFO_STREAM("frame_name reset to " << this->frame_name_);

        auto ao = ros::AdvertiseOptions::create<sensor_msgs::PointCloud2>(
                lidar_topic_name_, 2,
                boost::bind(&GazeboRosLidarUtils::LidarConnect, this),
                boost::bind(&GazeboRosLidarUtils::LidarDisconnect, this),
                ros::VoidPtr(), &this->lidar_queue_);
        this->pointcloud2_pub_ = this->nh_->advertise(ao);

        if (this->CanTriggerLidar()) {
            ros::SubscribeOptions trigger_so =
                    ros::SubscribeOptions::create<std_msgs::Empty>(
                            this->trigger_topic_name_, 2,
                            boost::bind(&GazeboRosLidarUtils::TriggerCameraInternal, this, _1),
                            ros::VoidPtr(), &this->lidar_queue_);
            this->trigger_subscriber_ = this->nh_->subscribe(trigger_so);
        }

        this->Init();
    }

    void GazeboRosLidarUtils::Init() {
        static uint32_t POINT_STEP = 4 + 4 + 4 + 4;

        int totalPoints = this->parentSensor_->VerticalRangeCount()
                          * this->parentSensor_->RangeCount();

        this->pointcloud2_msg_.header.frame_id = frame_name_;
        this->pointcloud2_msg_.point_step = POINT_STEP;
        this->pointcloud2_msg_.data.resize(totalPoints * POINT_STEP);

        this->pointcloud2_msg_.fields.resize(3);
        this->pointcloud2_msg_.fields[0].name = "x";
        this->pointcloud2_msg_.fields[0].offset = 0;
        this->pointcloud2_msg_.fields[0].datatype = sensor_msgs::PointField::FLOAT32;
        this->pointcloud2_msg_.fields[0].count = 1;

        this->pointcloud2_msg_.fields[1].name = "y";
        this->pointcloud2_msg_.fields[1].offset = 4;
        this->pointcloud2_msg_.fields[1].datatype = sensor_msgs::PointField::FLOAT32;
        this->pointcloud2_msg_.fields[1].count = 1;

        this->pointcloud2_msg_.fields[2].name = "z";
        this->pointcloud2_msg_.fields[2].offset = 8;
        this->pointcloud2_msg_.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
        this->pointcloud2_msg_.fields[2].count = 1;

        this->pointcloud2_msg_.fields[3].name = "intensity";
        this->pointcloud2_msg_.fields[3].offset = 12;
        this->pointcloud2_msg_.fields[3].datatype = sensor_msgs::PointField::FLOAT32;
        this->pointcloud2_msg_.fields[3].count = 1;


        this->callback_queue_thread_ = boost::thread(boost::bind(&GazeboRosLidarUtils::lidarQueueThread, this));
        this->initialized_ = true;
    }

    void GazeboRosLidarUtils::TriggerLidar() {

    }

    bool GazeboRosLidarUtils::CanTriggerLidar() {
        return false;
    }

    void GazeboRosLidarUtils::TriggerCameraInternal(const std_msgs::Empty::ConstPtr &dummy) {
        TriggerLidar();
    }

    void GazeboRosLidarUtils::LidarConnect() {
        boost::mutex::scoped_lock lock(this->lidar_connect_count_lock_);
        ROS_INFO("%s has %d subscribers", lidar_topic_name_.c_str(), this->pointcloud2_pub_.getNumSubscribers());
        if ((this->lidar_connect_count_) == 0)
            this->was_active_ = this->parentSensor_->IsActive();

        this->lidar_connect_count_++;

        this->parentSensor_->SetActive(true);
    }

    void GazeboRosLidarUtils::LidarDisconnect() {
        boost::mutex::scoped_lock lock(this->lidar_connect_count_lock_);
        ROS_INFO("%s has %d subscribers", lidar_topic_name_.c_str(), this->pointcloud2_pub_.getNumSubscribers());
        this->lidar_connect_count_--;

        // if there are no more subscribers, but camera was active to begin with,
        // leave it active.  Use case:  this could be a multicamera, where
        // each camera shares the same parentSensor_.
        if (this->lidar_connect_count_ <= 0 && !this->was_active_){
            this->parentSensor_->SetActive(false);
            ROS_INFO("%s is disconnected", lidar_topic_name_.c_str());
        }
    }

    void GazeboRosLidarUtils::lidarQueueThread() {
        static const double timeout = 0.001;
        while (this->nh_->ok()) {
            this->lidar_queue_.callAvailable(ros::WallDuration(timeout));
        }
    }
}
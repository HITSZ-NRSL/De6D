#include "velodyne_gazebo_plugins/GazeboROSCostomizedLiDAR.h"
#include <tf/tf.h>
#include <string.h>

#if GAZEBO_GPU_RAY

#include <gazebo/sensors/GpuRaySensor.hh>

#else
#include <gazebo/sensors/RaySensor.hh>
#endif
#if GAZEBO_GPU_RAY
#define RaySensor GpuRaySensor
#define STR_Gpu  "Gpu"
#define STR_GPU_ "GPU "
#else
#define STR_Gpu  ""
#define STR_GPU_ ""
#endif
namespace gazebo {
    GZ_REGISTER_SENSOR_PLUGIN(GazeboRosCustomizedLiDAR)

    GazeboRosCustomizedLiDAR::GazeboRosCustomizedLiDAR() : nh_(NULL) { ; }

    GazeboRosCustomizedLiDAR::~GazeboRosCustomizedLiDAR() {
        laser_queue_.clear();
        laser_queue_.disable();
        if (nh_) {
            nh_->shutdown();
            delete nh_;
            nh_ = NULL;
        }
        callback_laser_queue_thread_.join();
    }

    void GazeboRosCustomizedLiDAR::Load(sensors::SensorPtr _parent, sdf::ElementPtr _sdf) {
        ROS_INFO("-------------------");
        ROS_INFO("%s %splugin is loading ...", _sdf->Get<string>("name").c_str(), STR_GPU_);

        // load plugin
        RayPlugin::Load(_parent, _sdf);
        this->ray_sensor_ = std::dynamic_pointer_cast<sensors::RaySensor>(_parent);

        // read param
        if (!_sdf->HasElement("nameSpace")) {
            ROS_INFO("laser plugin missing <nameSpace>, defaults to '/' ");
            ns = "/";
        } else {
            ns = _sdf->GetElement("nameSpace")->Get<string>();
            ns = ns.empty() ? "/" : ns;
        }

        if (!_sdf->HasElement("topicName")) {
            lidar_topic_name_ = "/points";
        } else {
            lidar_topic_name_ = _sdf->GetElement("topicName")->Get<string>();
        }

        if (!_sdf->HasElement("frameName")) {
            ROS_INFO("laser plugin missing <frameName>, defaults to /world");
            frame_name_ = "/world";
        } else {
            frame_name_ = _sdf->GetElement("frameName")->Get<std::string>();
        }

        if (!_sdf->HasElement("minRange")) {
            ROS_INFO("laser plugin missing <minRange>, defaults to 0");
            min_range_ = 0;
        } else {
            min_range_ = _sdf->GetElement("minRange")->Get<double>();
        }

        if (!_sdf->HasElement("maxRange")) {
            ROS_INFO("laser plugin missing <maxRange>, defaults to infinity");
            max_range_ = INFINITY;
        } else {
            max_range_ = _sdf->GetElement("maxRange")->Get<double>();
        }
        if (!_sdf->HasElement("vDistribution")) {
            ROS_INFO("laser plugin missing <vDistribution>, defaults to whold");
            sectors_.empty();
        } else {
            stringstream info;
            info.precision(2);
            info.setf(ios::right);
            info.width(10);

            auto vcount = ray_sensor_->VerticalRangeCount();
            double pixel =
                    (ray_sensor_->VerticalAngleMax().Degree() - ray_sensor_->VerticalAngleMin().Degree()) / vcount;
            map_ = vector<bool>(vcount, false);

            auto vdstbt = _sdf->GetElement("vDistribution");
            int idx = 0;
            for (int j = 0;; ++j) {
                stringstream ss;
                ss << "sector" << j;
                auto ch = ss.str().c_str();
                if (vdstbt->HasElement(ch)) {
                    auto sector = vdstbt->GetElement(ch);
                    double begin, end;
                    int lines;
                    sector->GetAttribute("begin")->Get(begin);
                    sector->GetAttribute("end")->Get(end);
                    sector->GetAttribute("lines")->Get(lines);
                    int step = static_cast<int>((begin - end) / lines / pixel);

                    SECTOR temp{.angle={begin, end}, .count=lines, .pixel=step * pixel, step = step};
                    sectors_.push_back(temp);

                    auto offset = ray_sensor_->VerticalAngleMin().Degree();
                    int cnt = 1;
                    for (int i = floorf((begin - offset) / pixel);
                         i > ceilf((end - offset) / pixel);
                         i -= step, cnt++) {
                        if (cnt > lines)
                            break;
                        idx++;
                        map_[i] = true;
                        info << idx << "(" << ray_sensor_->VerticalAngleMin().Degree() + i * pixel << "), ";
                    }

                } else {
                    break;
                }

            }
            cout << info.str() << endl;
        }



        // init gazebo node
        gazebo_node_ = gazebo::transport::NodePtr(new gazebo::transport::Node());
        gazebo_node_->Init();

        // init ros node
        if (!ros::isInitialized()) {
            ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin. "
                                     << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
            return;
        }

//        nh_ = ns == "/" ? new ros::NodeHandle() : new ros::NodeHandle(ns);
        nh_ = new ros::NodeHandle(ns);
        //      Advertise publisher with a custom callback queue
        if (lidar_topic_name_ != "") {
            ros::AdvertiseOptions ao = ros::AdvertiseOptions::create<sensor_msgs::PointCloud2>(
                    lidar_topic_name_, 1,
                    boost::bind(&GazeboRosCustomizedLiDAR::ConnectCb, this),
                    boost::bind(&GazeboRosCustomizedLiDAR::ConnectCb, this),
                    ros::VoidPtr(), &laser_queue_);
            ros_pub_ = nh_->advertise(ao);
        }
        //      set lidar-frame
        std::string prefix;
        nh_->getParam(std::string("tf_prefix"), prefix);
        if (ns != "/") {
            prefix = ns;
        }
        boost::trim_right_if(prefix, boost::is_any_of("/"));
        frame_name_ = tf::resolve(prefix, frame_name_);
        // init PointCloud2 msg
        static uint32_t POINT_STEP = 12;
        {
            int totalLidarPoints = ray_sensor_->VerticalRangeCount() * ray_sensor_->RangeCount();

            msg.header.frame_id = frame_name_;
            msg.point_step = POINT_STEP;
            msg.data.resize(totalLidarPoints * msg.point_step);

            msg.fields.resize(3);
            msg.fields[0].name = "x";
            msg.fields[0].offset = 0;
            msg.fields[0].datatype = sensor_msgs::PointField::FLOAT32;
            msg.fields[0].count = 1;
            msg.fields[1].name = "y";
            msg.fields[1].offset = 4;
            msg.fields[1].datatype = sensor_msgs::PointField::FLOAT32;
            msg.fields[1].count = 1;
            msg.fields[2].name = "z";
            msg.fields[2].offset = 8;
            msg.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
            msg.fields[2].count = 1;
//            msg.fields[3].name = "intensity";
//            msg.fields[3].offset = 12;
//            msg.fields[3].datatype = sensor_msgs::PointField::FLOAT32;
//            msg.fields[3].count = 1;
//            msg.fields[4].name = "ring";
//            msg.fields[4].offset = 16;
//            msg.fields[4].datatype = sensor_msgs::PointField::UINT16;
//            msg.fields[4].count = 1;
//            msg.fields[5].name = "time";
//            msg.fields[5].offset = 18;
//            msg.fields[5].datatype = sensor_msgs::PointField::FLOAT32;
//            msg.fields[5].count = 1;
        }

        // thread
        callback_laser_queue_thread_ = boost::thread(boost::bind(&GazeboRosCustomizedLiDAR::laserQueueThread, this));

        // disable the lidar simulation until topic connect
        this->ray_sensor_->SetActive(false);
        ROS_INFO("%splugin is loaded", STR_GPU_);
    }

    double active_timestamp;

    void GazeboRosCustomizedLiDAR::ConnectCb() {
        boost::lock_guard<boost::mutex> lock(lock_);

        if (ros_pub_.getNumSubscribers()) {
            if (!gazebo_sub_) {
                ROS_INFO("%s has %d subscribers", lidar_topic_name_.c_str(), ros_pub_.getNumSubscribers());
                gazebo_sub_ = gazebo_node_->Subscribe(this->ray_sensor_->Topic(), &GazeboRosCustomizedLiDAR::OnScan,
                                                      this);
            }
            ray_sensor_->SetActive(true);
            active_timestamp = ros::Time::now().toSec();
        } else {
            ROS_INFO("%s is disconnected", lidar_topic_name_.c_str());
            if (gazebo_sub_) {
                gazebo_sub_->Unsubscribe();
                gazebo_sub_.reset();
            }
            ray_sensor_->SetActive(false);
        }
    }


    void GazeboRosCustomizedLiDAR::OnScan(ConstLaserScanStampedPtr &_msg) {
        const double maxRange = ray_sensor_->RangeMax();
        const double minRange = ray_sensor_->RangeMin();

        const ignition::math::Angle maxAngle = ray_sensor_->AngleMax();
        const ignition::math::Angle minAngle = ray_sensor_->AngleMin();
        const ignition::math::Angle verticalMaxAngle = ray_sensor_->VerticalAngleMax();
        const ignition::math::Angle verticalMinAngle = ray_sensor_->VerticalAngleMin();

        const int rayCount = ray_sensor_->RayCount();
        const int rangeCount = ray_sensor_->RangeCount();
        const int verticalRayCount = ray_sensor_->VerticalRayCount();
        const int verticalRangeCount = ray_sensor_->VerticalRangeCount();

        const double yDiff = maxAngle.Radian() - minAngle.Radian();
        const double pDiff = verticalMaxAngle.Radian() - verticalMinAngle.Radian();
        const double yStep = rangeCount == 0 ? 0 : yDiff / (rangeCount - 1);
        const double pStep = verticalRangeCount == 0 ? 0 : pDiff / (verticalRangeCount - 1);


        const double MIN_RANGE = std::max(min_range_, minRange);
        const double MAX_RANGE = std::min(max_range_, maxRange);
        const double MIN_INTENSITY = min_intensity_;

        // convert gazebo msg and clock ros topic
        uint8_t *ptr = msg.data.data();
        for (int i = 0; i < rangeCount; i++) {
            for (int j = 0; j < verticalRangeCount; j++) {
                if (!map_[j])
                    continue;
                double r = _msg->scan().ranges(i + j * rangeCount);

                // Get angles of ray to get xyz for point
                double yAngle = i * yStep + minAngle.Radian();
                double pAngle = j * pStep + verticalMinAngle.Radian();


                // pAngle is rotated by yAngle:
                if ((MIN_RANGE < r) && (r < MAX_RANGE)) {
                    *((float *) (ptr + 0)) = r * cos(pAngle) * cos(yAngle);
                    *((float *) (ptr + 4)) = r * cos(pAngle) * sin(yAngle);
                    *((float *) (ptr + 8)) = r * sin(pAngle);
                    ptr += msg.point_step;
                } else {
                    *((float *) (ptr + 0)) = 0;
                    *((float *) (ptr + 4)) = 0;
                    *((float *) (ptr + 8)) = 0;
                    continue;
                }
            }
        }
        msg.header.stamp = ros::Time(_msg->time().sec(), _msg->time().nsec());
        msg.row_step = ptr - msg.data.data();
        msg.height = 1;
        msg.width = msg.row_step / msg.point_step;
        msg.is_bigendian = false;
        msg.is_dense = true;
        msg.data.resize(msg.row_step);
        ros_pub_.publish(msg);
        static double last_time = 0;
        auto this_time = (ros::Time::now().toSec() - active_timestamp);
        ROS_INFO("%s active time: %4.1lf, %4.1lf", lidar_topic_name_.c_str(), this_time,
                this_time-last_time);
        last_time = this_time;
    }

    void GazeboRosCustomizedLiDAR::laserQueueThread() {
        while (nh_->ok()) {
            laser_queue_.callAvailable(ros::WallDuration(0.01));
        }
    }
}

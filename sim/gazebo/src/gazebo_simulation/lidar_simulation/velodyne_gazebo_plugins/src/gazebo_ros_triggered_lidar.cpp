//
// Created by ou on 2021/6/20.
//
#include "velodyne_gazebo_plugins/gazebo_ros_triggered_lidar.h"

using namespace std;
namespace gazebo {
    GZ_REGISTER_SENSOR_PLUGIN(GazeboRosTriggeredLidar)

    void GazeboRosTriggeredLidar::Load(sensors::SensorPtr _parent, sdf::ElementPtr _sdf) {
        ROS_INFO("-------------------");
        ROS_INFO("%s %splugin is loading ...", _sdf->Get<string>("name").c_str(), STR_GPU_);

        // Make sure the ROS node for Gazebo has already been initialized
        if (!ros::isInitialized()) {
            ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin. "
                                     << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
            return;
        }

        RayPlugin::Load(_parent, _sdf);
        GazeboRosLidarUtils::Load(_parent, _sdf);

        if (!_sdf->HasElement("vDistribution")) {
            ROS_INFO("laser plugin missing <vDistribution>, defaults to whold");
            sectors_.empty();
        } else {
            stringstream info;
            info.precision(2);
            info.setf(ios::right);
            info.width(10);

            auto v_count = this->parentSensor_->VerticalRangeCount();
            double pixel = (this->parentSensor_->VerticalAngleMax().Degree()
                            - this->parentSensor_->VerticalAngleMin().Degree()) / v_count;
            map_ = vector<bool>(v_count, false);

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

                    auto offset = this->parentSensor_->VerticalAngleMin().Degree();
                    int cnt = 1;
                    for (int i = floorf((begin - offset) / pixel);
                         i > ceilf((end - offset) / pixel);
                         i -= step, cnt++) {
                        if (cnt > lines)
                            break;
                        idx++;
                        map_[i] = true;
                        info << idx << "(" << this->parentSensor_->VerticalAngleMin().Degree() + i * pixel << "), ";
                    }
                } else {
                    break;
                }
            }
            cout << info.str() << endl;
        }

        this->SetLidarEnabled(false);
        this->updateConnection =
                event::Events::ConnectBeforePhysicsUpdate(
                        std::bind(&GazeboRosTriggeredLidar::OnUpdate, this));

        ROS_INFO("%s %splugin is loaded", _sdf->Get<string>("name").c_str(), STR_GPU_);
    }

    bool GazeboRosTriggeredLidar::CanTriggerLidar() {
        return true;
    }

    void GazeboRosTriggeredLidar::TriggerLidar() {
        std::lock_guard<std::mutex> lock(this->mutex);
        if (!this->parentSensor_)
            return;
        this->triggered++;
    }

    void GazeboRosTriggeredLidar::OnUpdate() {
        std::lock_guard<std::mutex> lock(this->mutex);
        if (this->triggered > 0) {
            this->SetLidarEnabled(true);
        }
    }

    void GazeboRosTriggeredLidar::SetLidarEnabled(const bool _enabled) {
        this->parentSensor_->SetActive(_enabled);
        this->parentSensor_->SetUpdateRate(_enabled ? 0.0 : DBL_MIN);
    }

    void GazeboRosTriggeredLidar::OnNewLaserScans(){
        /// TODO:或使用Sensor::ConnectUpdated(std::function<void()> _subscriber)
        ROS_INFO("on_new_laser_scan");
//        this->sensor_update_time_ = this->parentSensor_->LastMeasurementTime();
//
//        if ((*this->image_connect_count_) > 0)
//        {
//            this->PutCameraData(_image);
//            this->PublishCameraInfo();
//        }
//        this->SetCameraEnabled(false);
//
//        std::lock_guard<std::mutex> lock(this->mutex);
//        this->triggered = std::max(this->triggered-1, 0);
//        this->parentSensor_->Ranges()
    }
}
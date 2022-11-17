/*
 * Copyright (C) 2016 Open Source Robotics Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
*/
#include <ignition/transport.hh>
#include <ignition/math.hh>
#include <ignition/msgs.hh>
#include <gazebo/common/Time.hh>
#include <ros/node_handle.h>
#include <gazebo_msgs/LinkStates.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <iostream>

/////////////////////////////////////////////////
ignition::transport::Node *n;

class Marker : public ignition::msgs::Marker {
public:
    explicit Marker(const std::string &ns = "default", const unsigned long &id = 0,
                    ::ignition::msgs::Marker_Type t = ::ignition::msgs::Marker_Type::Marker_Type_BOX) {
        set_ns(ns);
        set_id(id);
        set_type(t);
    }

    virtual void set_pose(const ignition::math::Pose3d &pose) {
        ignition::msgs::Set(mutable_pose(), pose);
    }

    virtual void draw(ignition::transport::Node &node) { node.Request("/marker", *this); }
};


class ConeMarker : public Marker {
    float resolution = 0.1;
    float radius = 0.15;
    float length = 0.3;
public:
    explicit ConeMarker(const std::string &ns = "default", const unsigned long &id = 0, float res = 0.1)
            : Marker(ns, id), resolution(res) {
        set_action(ignition::msgs::Marker::ADD_MODIFY);
        set_type(ignition::msgs::Marker::TRIANGLE_FAN);
    }

    void set_shape(float radius = 0.15, float length = 0.3) {
        this->radius = radius;
        this->length = length;
        reset_shape();
    }

    void set_resolution(float res = 0.1) {
        this->resolution = res;
    }

    void reset_shape() {
        /*
         * points: 相对自身坐标系
         * 第一个点为圆心
         * 其余点为圆边点
         */
        clear_point();
        ignition::msgs::Set(add_point(),
                            ignition::math::Vector3d(0, 0, this->length));
        auto res = this->resolution > M_PI_4 ? M_PI_4 : this->resolution;
        for (double t = 0; t <= 2 * M_PI; t += this->resolution) {
            ignition::msgs::Set(add_point(),
                                ignition::math::Vector3d(this->radius * cos(t),
                                                         this->radius * sin(t),
                                                         0));
        }
        ignition::msgs::Set(add_point(),
                            ignition::math::Vector3d(this->radius, 0, 0));
    }
};

class CylinderMarker : public Marker {
    float radius = 0.1, length = 0.6;
public:
    explicit CylinderMarker(const std::string &ns = "default", const unsigned long &id = 0) : Marker(ns, id) {
        set_action(ignition::msgs::Marker::ADD_MODIFY);
        set_type(ignition::msgs::Marker::CYLINDER);
    }

    void set_shape(float radius = 0.15, float length = 0.3) {
        this->radius = radius;
        this->length = length;
        reset_shape();
    }

    void reset_shape() {
        ignition::msgs::Set(mutable_scale(),
                            ignition::math::Vector3d(radius, radius, length));
    }
};

class ArrowMarker : public ConeMarker, public CylinderMarker {
    float resolution = 0.1;
    float head_length = 0.3, shaft_length = 0.6;
public:
    explicit ArrowMarker(const std::string &ns = "default", const unsigned long &id = 0, float res = 0.1) :
            ConeMarker(ns, id, res), CylinderMarker(ns, id + ULONG_LONG_MAX / 2) {

        ignition::msgs::Material *matMsg = ConeMarker::mutable_material();
        matMsg->mutable_script()->set_name("Gazebo/RedLaser");
    }

    void set_head(float radius = 0.15, float length = 0.3) {
        head_length = 0.3;
        ConeMarker::set_shape(radius, length);
    }

    void set_shaft(float radius = 0.1, float length = 0.6) {
        shaft_length = 0.6;
        CylinderMarker::set_shape(radius, length);
    }

    void set_pose(const ignition::math::Pose3d &pose) {
        auto p1 = pose, p2 = pose;
        p1 = ignition::math::Pose3d(this->shaft_length / 2, .0, 0., 0, M_PI_2, 0) * p1;
        ignition::msgs::Set(CylinderMarker::mutable_pose(), p1);
        p2 = ignition::math::Pose3d(this->shaft_length, .0, 0., 0, M_PI_2, 0) * p2;
        ignition::msgs::Set(ConeMarker::mutable_pose(), p2);
    }

    void draw(ignition::transport::Node &node) override {
        ConeMarker::draw(node);
        CylinderMarker::draw(node);
    }

    void set_action(::ignition::msgs::Marker_Action value) {
        ConeMarker::set_action(value);
        CylinderMarker::set_action(value);
    }
};

ignition::math::Pose3d car_pose, lidar_pose(1.5, 0, 1.72, 0, 0, 0);

void set_pose(const gazebo_msgs::LinkStatesConstPtr msg) {
    for (int i = 0; i < msg->name.size(); i++) {
        if (msg->name[i] == "prius_ego::actor_base_link") {
            car_pose = ignition::math::Pose3d(msg->pose[i].position.x,
                                              msg->pose[i].position.y,
                                              msg->pose[i].position.z,
                                              msg->pose[i].orientation.w,
                                              msg->pose[i].orientation.x,
                                              msg->pose[i].orientation.y,
                                              msg->pose[i].orientation.z);
            car_pose = lidar_pose * car_pose;
        }
    }
}

void callback(const visualization_msgs::MarkerArray::ConstPtr &msgs) {
    static int num = 0;
    static auto mk = msgs->markers[0];
    printf("%zu\n", msgs->markers.size());
    for (int i = 0; i < msgs->markers.size(); i++) {
        auto &marker = msgs->markers[i];
        Marker marker_gz(marker.ns, marker.id, ignition::msgs::Marker::LINE_LIST);
        marker_gz.set_action(ignition::msgs::Marker::ADD_MODIFY);
        ignition::msgs::Set(marker_gz.mutable_pose(), car_pose);
        for (auto pt: marker.points) {
            ignition::msgs::Set(marker_gz.add_point(), ignition::math::Vector3d(pt.x, pt.y, pt.z));
        }
        marker_gz.mutable_material()->mutable_script()->set_name("Gazebo/Green");
        marker_gz.draw(*n);
    }
//    for(int i=msgs->markers.size();i<num;i++){
//        Marker marker_gz(mk.ns, i, ignition::msgs::Marker::NONE);
//        marker_gz.set_action(ignition::msgs::Marker::DELETE_MARKER);
//        marker_gz.draw(*n);
//    }

    num = msgs->markers.size();
}

void callback_pts(const sensor_msgs::PointCloud2ConstPtr msg) {
    double max_range = 1.5;
    ignition::math::Pose3d offset(0, 0, 1, 0, 0, 0);

    auto square_dist = [](auto &&a) { return a.x() * a.x() + a.y() * a.y() + a.z() * a.z(); };
    // points marker
    Marker points_gz("lidar_viz", 0, ignition::msgs::Marker::POINTS);
    points_gz.set_action(ignition::msgs::Marker::ADD_MODIFY);
    points_gz.mutable_material()->mutable_script()->set_name("Gazebo/BlueLaser");
    auto p_end = msg->data.data() + msg->data.size();
    for (auto p = msg->data.data(); p < p_end; p += msg->point_step) {
        ignition::msgs::Set(points_gz.add_point(), ignition::math::Vector3d(
                *(float *) (p), *(float *) (p + 4), *(float *) (p + 8)));
    }
    auto &pts = points_gz.point();
    auto max_ele = *std::max_element(pts.begin(), pts.end(), [square_dist](auto &&a, auto &&b) {
        return square_dist(a) < square_dist(b);
    });
    auto max_dist = std::max(sqrt(square_dist(max_ele)), 100.);
    auto scale = max_range/max_dist;
    ignition::msgs::Set(points_gz.mutable_scale(), ignition::math::Vector3d(scale,scale,scale));
    ignition::msgs::Set(points_gz.mutable_pose(), offset * car_pose);
//    points_gz.draw(*n);

    // box
    static const ignition::math::Vector3d box_corner[] = {{-1, -1, 0},
                                                          {1,  -1, 0},
                                                          {1,  1,  0},
                                                          {-1, 1,  0}};
    Marker box_gz("lidar_viz", 1, ignition::msgs::Marker::LINE_STRIP);
    box_gz.set_action(ignition::msgs::Marker::ADD_MODIFY);
    ignition::msgs::Set(box_gz.mutable_pose(), offset * car_pose);
    for (int i = 0; i < 4; i++) {
        ignition::msgs::Set(box_gz.add_point(), ignition::math::Vector3d(0, 0, -offset.Pos().Z()));
        ignition::msgs::Set(box_gz.add_point(), max_range*box_corner[i % 4]);
        ignition::msgs::Set(box_gz.add_point(), max_range*box_corner[i % 4]);
        ignition::msgs::Set(box_gz.add_point(), max_range*box_corner[(i + 1) % 4]);
    }
    ignition::msgs::Set(box_gz.mutable_scale(), ignition::math::Vector3d(1,1,1));
    box_gz.mutable_material()->mutable_script()->set_name("Gazebo/Black");
    box_gz.draw(*n);
}


int main(int _argc, char **_argv) {
    ros::init(_argc, _argv, "gz_marker");
    auto nh = ros::NodeHandle();
    auto sub_marker = nh.subscribe("/marker", 1, callback);
    auto sub_pose = nh.subscribe("/gazebo/link_states", 1, set_pose);
//    auto sub_cloud = nh.subscribe("/prius_ego/hdl_32e", 1, callback_pts);
    ignition::transport::Node node;
    n = &node;
    ros::spin();
}

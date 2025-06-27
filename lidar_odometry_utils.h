#ifndef _LIDAR_ODOMETRY_UTILS_H_
#define _LIDAR_ODOMETRY_UTILS_H_


#include <kiss_icp/pipeline/KissICP.hpp>
#include <portable-file-dialogs.h>

#include "structures.h"

#include <laszip/laszip_api.h>
#include <iostream>

#include <Eigen/Dense>

#include <vector>

#include <map>
#include <execution>

#include <imgui.h>
#include <imgui_impl_glut.h>
#include <imgui_impl_opengl2.h>
#include <ImGuizmo.h>
#include <imgui_internal.h>
#include <GL/glu.h>
#include <GL/gl.h>

#ifdef WITH_ROS2
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <rosbag2_cpp/converter_interfaces/serialization_format_converter.hpp>
#endif



unsigned long long int get_index(const int16_t x, const int16_t y, const int16_t z);
unsigned long long int get_rgd_index(const Eigen::Vector3d p, const Eigen::Vector3d b);
// this function finds interpolated pose between two poses according to query_time
Eigen::Matrix4d getInterpolatedPose(const std::map<double, Eigen::Matrix4d> &trajectory, double query_time);

// this function reduces number of points by preserving only first point for each bucket {bucket_x, bucket_y, bucket_z}
std::vector<Point3Di> decimate(const std::vector<Point3Di> &points, double bucket_x, double bucket_y, double bucket_z);

//! This function load point cloud from LAS/LAZ file.
//! Optionally it can apply extrinsic calibration to each point.
//! The calibration is stored in a map, where key is laser scanner id.
//! The id of the laser scanner is stored in LAS/LAZ file as `user_data` field.
//! @param lazFile - path to file with point cloud
//! @param ommit_points_with_timestamp_equals_zero - if true, points with timestamp == 0 will be omited
//! @param filter_threshold_xy - threshold for filtering points in xy plane
//! @param calibrations - map of calibrations for each scanner key is scanner id.
//! @return vector of points of @ref Point3Di type
std::vector<Point3Di> load_point_cloud(const std::string &lazFile, bool ommit_points_with_timestamp_equals_zero, double filter_threshold_xy,
                                       const std::unordered_map<int, Eigen::Affine3d>& calibrations);

#ifdef WITH_ROS2
//! Load point cloud data from ROS 2 rosbag files
//! @param rosbag_path - path to rosbag directory
//! @param topic_filters - list of PointCloud2 topics to extract (empty = all topics)
//! @param start_time_sec - start time in seconds (0 = from beginning)  
//! @param end_time_sec - end time in seconds (0 = to end)
//! @param filter_threshold_xy - threshold for filtering points in xy plane
//! @return vector of points grouped by topic/timestamp
std::vector<std::vector<Point3Di>> load_rosbag_pointclouds(
    const std::string& rosbag_path,
    const std::vector<std::string>& topic_filters = {},
    double start_time_sec = 0.0,
    double end_time_sec = 0.0,
    double filter_threshold_xy = 0.0);

//! Convert ROS PointCloud2 message to Point3Di vector
//! @param cloud_msg - ROS PointCloud2 message
//! @param lidar_id - ID to assign to all points (default 0)
//! @param filter_threshold_xy - threshold for filtering points in xy plane
//! @return vector of Point3Di structures
std::vector<Point3Di> pointcloud2_to_point3di(
    const sensor_msgs::msg::PointCloud2& cloud_msg,
    uint8_t lidar_id = 0,
    double filter_threshold_xy = 0.0);

//! Get available PointCloud2 topics from rosbag
//! @param rosbag_path - path to rosbag directory
//! @return vector of topic names with PointCloud2 message type
std::vector<std::string> get_pointcloud2_topics(const std::string& rosbag_path);
#endif

bool saveLaz(const std::string &filename, const std::vector<Point3Di> &points_global);
bool save_poses(const std::string file_name, std::vector<Eigen::Affine3d> m_poses, std::vector<std::string> filenames);


// this function draws ellipse for each bucket
void draw_ellipse(const Eigen::Matrix3d &covar, const Eigen::Vector3d &mean, Eigen::Vector3f color, float nstd = 3);


//! This namespace contains functions for loading calibration file (.json and .sn).
//!
//! Calibration file is a json file with the following format:
//!{```json
//!    "calibration": {
//!      "47MDL9T0020193": {
//!        "identity" : "true"
//!      },
//!      "47MDL9S0020300":
//!          {
//!            "order" : "ROW",
//!            "inverted" : "TRUE",
//!            "data":[
//!             0.999824, 0.00466397, -0.0181595, -0.00425984,
//!             -0.0181478, -0.00254457, -0.999832, -0.151599,
//!              -0.0047094,0.999986, -0.00245948, -0.146408,
//!              0, 0, 0, 1
//!            ]
//!          }
//!    },
//!                    "imuToUse": "47MDL9T0020193"
//!}```
//! Json object `calibration` contains a map of calibration for each sensor.
//! The key of the map is serial number of the sensor.
//! The value is a json object with the following fields:
//! - `identity` - if true, the calibration is identity matrix
//!  - `order` - order of the matrix, can be `ROW` or `COLUMN`
//!  - `inverted` - if true, the calibration matrix is inverted
//!  - `data` - calibration matrix in given order
//! Json object `imuToUse` contains serial number of the sensor to use for IMU.
//! The JSON file contains mapping from sensor id to serial number to calibration.
//! The sensor id is the id of the point in LAS/LAZ file.
//!
//! The MANDEYE_CONTROLLER saves the sensor id in `user_data` field of LAZ file,
//! and also saves the serial number of the sensors in .sn file.
//! The .sn file is a text file with the following format:
//! ```text
//! 0 47MDL9T0020193
//! 1 47MDL9S0020300
//! ```
//! The first column is sensor id, the second column is serial number.
//! It is a mapping from lidarid (used in LAZ file) to serial number.
//! Those two files allows to apply calibration to each point in LAZ file.
namespace MLvxCalib {

//! Parse the calibration file and return a map from sensor id to serial number.
//! Sensor id is the id is id of the point in laz file.
//! Serial number is the serial number of the Livox.
//! @param filename calibration file
//! @return map of serial number, where key is sensor id.
std::unordered_map<int, std::string> GetIdToSnMapping(const std::string& filename);

//! Parse the calibration file and return a map from serial number to calibration.
//! @param filename calibration file
//! @return map of extrinsic calibration, where key is serial number of the lidar.
std::unordered_map<std::string, Eigen::Affine3d> GetCalibrationFromFile(const std::string& filename);

//! Parse the calibration file and return a serial number of the Livox to use for IMU.
//! @param filename calibration file
//! @return serial number of the Livox to use for IMU
std::string GetImuSnToUse(const std::string& filename);

//! Combine the id to serial number mapping and the calibration into a single map.
//! The single map is from sensor id to calibration.
//! @param idToSn mapping from serial number to Id number in pointcloud or IMU CSV
//! @param calibration map of extrinsic calibration, where key is serial number of the lidar.
//! @return map from sensor id to extrinsic calibration
std::unordered_map<int, Eigen::Affine3d> CombineIntoCalibration(const std::unordered_map<int, std::string>& idToSn,
                                                                const std::unordered_map<std::string, Eigen::Affine3d>& calibration);
//! Get the id of the IMU to use.
//! @param idToSn mapping from serial number to Id number in pointcloud or IMU CSV
//! @param snToUse serial number of the Livox to use for IMU
//! @return id of the IMU to use
int GetImuIdToUse(const std::unordered_map<int, std::string>& idToSn, const std::string& snToUse );

}
#endif

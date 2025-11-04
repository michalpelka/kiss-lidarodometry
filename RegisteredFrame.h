#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <set>
struct TrajectoryNode
{
    Eigen::Affine3d pose;
    double hardware_timestamp;
    double unix_timestamp;

};

struct RegisteredFrame
{
    static bool CacheEnabled;
    static std::set<std::string> CachedFiles;
    static void ClearCache();
    std::vector<Eigen::Vector3d> points;
    std::vector<float> intensities;
    std::vector<double> timestamps_offset;
    std::vector<double> timestamp_hardware;
    Eigen::Affine3d pose;
    std::vector<TrajectoryNode> trajectory;
    int id = 0;
    static int idCounter;

    // path to the cached binary file (empty if not cached)
    std::string cacheFilePath;

    RegisteredFrame();

    //! Remove heavy data
    void RelaseData();
    void Cache();
    void UnCache();

    // cereal serialize - defined inline so it can be instantiated where needed
    template <class Archive>
    void serialize(Archive &ar)
    {
        ar(points, intensities, timestamps_offset, timestamp_hardware, pose, trajectory, id);
    }

};

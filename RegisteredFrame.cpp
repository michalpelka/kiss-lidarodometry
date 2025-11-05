#include "RegisteredFrame.h"

#include "cereal/cereal.hpp"
#include <array>
#include <atomic>
#include <cereal/archives/binary.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/vector.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

// single definition of the static id counter
int RegisteredFrame::idCounter = 0;
bool RegisteredFrame::CacheEnabled = false;
std::set<std::string> RegisteredFrame::CachedFiles;
RegisteredFrame::RegisteredFrame()
{
    id = idCounter++;
    cacheFilePath = (std::filesystem::temp_directory_path() / "KissLidarOdometry" / ("registered_frame_cache_" + std::to_string(id) + ".bin")).string();

}

void RegisteredFrame::RelaseData()
{
    if (!CacheEnabled) return;
    points.clear();
    intensities.clear();
    timestamp_hardware.clear();
    timestamps_offset.clear();
    trajectory.clear();
}

// Save this RegisteredFrame to a binary file using cereal
void RegisteredFrame::Cache()
{

    if (!CacheEnabled) return;
    if(cacheFilePath.empty()) return;
    // create parent dir
    const auto parentPath = std::filesystem::path(cacheFilePath).parent_path();
    if(!std::filesystem::exists(parentPath))
    {
        std::filesystem::create_directories(parentPath);
    }
    std::ofstream os(cacheFilePath, std::ios::binary);
    if(!os) return;
    cereal::BinaryOutputArchive archive(os);
    archive(*this);
    CachedFiles.insert(cacheFilePath);
}

// Load a RegisteredFrame from a binary file using cereal
void RegisteredFrame::UnCache()
{

    if (!CacheEnabled) return;
    std::ifstream is(cacheFilePath, std::ios::binary);
    if(!is) return;
    cereal::BinaryInputArchive archive(is);
    archive(*this);
}

void RegisteredFrame::ClearCache()
{
    for (auto& fn : CachedFiles)
    {
        std::filesystem::remove(fn);
    }

}


// Non-intrusive cereal support for Eigen::Vector3d and Eigen::Affine3d and TrajectoryNode
namespace cereal {

template <class Archive>
void save(Archive &ar, const Eigen::Vector3d &v)
{
    ar(v.x(), v.y(), v.z());
}

template <class Archive>
void load(Archive &ar, Eigen::Vector3d &v)
{
    double x, y, z;
    ar(x, y, z);
    v.x() = x; v.y() = y; v.z() = z;
}

template <class Archive>
void save(Archive &ar, const Eigen::Affine3d &a)
{
    // serialize as a flat array of 16 doubles (row-major)
    std::array<double, 16> arr;
    Eigen::Matrix4d m = a.matrix();
    for(int r = 0; r < 4; ++r)
        for(int c = 0; c < 4; ++c)
            arr[r * 4 + c] = m(r, c);
    ar(arr);
}

template <class Archive>
void load(Archive &ar, Eigen::Affine3d &a)
{
    std::array<double, 16> arr;
    ar(arr);
    Eigen::Matrix4d m;
    for(int r = 0; r < 4; ++r)
        for(int c = 0; c < 4; ++c)
            m(r, c) = arr[r * 4 + c];
    a = Eigen::Affine3d(m);
}

template <class Archive>
void serialize(Archive &ar, TrajectoryNode &t)
{
    ar(t.pose, t.hardware_timestamp, t.unix_timestamp);
}



} // namespace cereal

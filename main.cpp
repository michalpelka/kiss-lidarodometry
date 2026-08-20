#include <kiss_icp/pipeline/KissICP.hpp>
#include "lidar_odometry_utils.h"
#include <portable-file-dialogs.h>

#include <imgui.h>
#include <imgui_impl_glut.h>
#include <imgui_impl_opengl2.h>
#include <imgui_internal.h>

#include <GL/freeglut.h>

#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <Fusion/FusionAhrs.h>
#include <nlohmann/json.hpp>
#include "nmeaParser.h"
#include "Params.h"
#include "RegisteredFrame.h"

namespace fs = std::filesystem;

void SaveTrj(const std::string& pathtrj, const std::vector<TrajectoryNode>& trajectory)
{
    std::ofstream outfile;
    outfile.open(pathtrj);
    if (!outfile.good())
    {
        std::cout << "can not save file: " << pathtrj << std::endl;
        return;
    }

    outfile << "timestamp_nanoseconds pose00 pose01 pose02 pose03 pose10 pose11 pose12 pose13 pose20 pose21 pose22 pose23 timestampUnix_nanoseconds" << std::endl;
    for (int j = 0; j < trajectory.size(); j++)
    {
        const auto& trjNode = trajectory[j];
        auto pose = trjNode.pose.matrix();
        outfile << std::setprecision(20) << trjNode.hardware_timestamp * 1e9 << " " << std::setprecision(10) << pose(0, 0) << " "
                << pose(0, 1) << " " << pose(0, 2) << " " << pose(0, 3) << " " << pose(1, 0) << " " << pose(1, 1) << " " << pose(1, 2)
                << " " << pose(1, 3) << " " << pose(2, 0) << " " << pose(2, 1) << " " << pose(2, 2) << " " << pose(2, 3) << " "
                << std::setprecision(20) << trjNode.hardware_timestamp * 1e9 << std::endl;
    }
    outfile.close();
}

std::vector<Point3Di> CreateMetascan(std::vector<RegisteredFrame>& frames)
{
    std::vector<Point3Di> result;
    for (auto& currentOriginalFrame : frames)
    {
        currentOriginalFrame.UnCache();
        for (size_t i = 0; i < currentOriginalFrame.points.size(); ++i)
        {
           const auto& originalPoint = currentOriginalFrame.points[i];
           const auto transformedPoint = currentOriginalFrame.pose * originalPoint;
            Point3Di p;
            p.point = transformedPoint;
            p.timestamp = currentOriginalFrame.timestamps_offset[i];
            p.intensity = currentOriginalFrame.intensities[i];;
            result.push_back(p);
        }
        currentOriginalFrame.RelaseData();
    }
    return result;
}
std::vector<RegisteredFrame> ConcatenateFrames( std::vector<RegisteredFrame>& frames, int maxNumberOfPoints = 200000)
{
    std::vector<RegisteredFrame> result;
    result.resize(1);
    result.back().pose = frames.front().pose;
    Eigen::Affine3d currentIncrement = Eigen::Affine3d::Identity();
    for (auto& currentOriginalFrame : frames)
    {
        currentOriginalFrame.UnCache();
        TrajectoryNode currentTrajectoryNode;
        currentIncrement = result.back().pose.inverse() * currentOriginalFrame.pose;
        currentTrajectoryNode.hardware_timestamp = currentOriginalFrame.timestamp_hardware.front();
        currentTrajectoryNode.pose = currentIncrement;
        result.back().trajectory.push_back(currentTrajectoryNode);
        std::cout << "Frame at " << currentTrajectoryNode.hardware_timestamp << " has " << currentOriginalFrame.points.size() << " points" << std::endl;
        for (size_t i = 0; i < currentOriginalFrame.points.size(); ++i)
        {
            assert(currentOriginalFrame.points.size() == currentOriginalFrame.intensities.size());
            assert(currentOriginalFrame.points.size() == currentOriginalFrame.timestamps_offset.size());
            auto & buildFrame = result.back();
            buildFrame.points.push_back(currentIncrement * currentOriginalFrame.points[i]);
            buildFrame.intensities.push_back(currentOriginalFrame.intensities[i]);
            buildFrame.timestamps_offset.push_back(currentOriginalFrame.timestamps_offset[i]);
            buildFrame.timestamp_hardware.push_back(currentOriginalFrame.timestamp_hardware[i]);
            if (buildFrame.points.size() >= maxNumberOfPoints)
            {
                result.resize(result.size() + 1);
                result.back().pose = currentOriginalFrame.pose;
                currentIncrement = result.back().pose.inverse() * currentOriginalFrame.pose;
            }
        }
        currentOriginalFrame.RelaseData();
    }
    std::cout << "Concatenated " << frames.size() << " frames into " << result.size() << " frames" << std::endl;
    return result;
}

std::vector<Point3Di> toPoint3DiV(const std::vector<Eigen::Vector3d>& points, const std::vector<float>& intensities, const std::vector<double>& timestamps)
{
    assert(points.size() == intensities.size());
    assert(points.size() == timestamps.size());
    std::vector<Point3Di> result;
    result.reserve(points.size());
    for (int i = 0; i < points.size(); ++i)
    {
        result.push_back({ points[i], timestamps[i], intensities[i], i, 0 });
    }
    return result;
}


class DelayedPoints
{
    std::vector<Point3Di> m_data;
    std::string m_filename;
    std::unordered_map<std::string, Eigen::Affine3d> m_calibration;
    float m_filter_threshold_xy = 0.0;
public:
    DelayedPoints() = default;
    DelayedPoints(const std::string& filename, const std::unordered_map<std::string, Eigen::Affine3d> & calibration, float filter_threshold_xy):
        m_filename(filename), m_calibration(calibration), m_filter_threshold_xy(filter_threshold_xy)
    {
    }

    void LoadData()
    {
        if (m_data.size() > 0)
        {
            return;
        }

        // Load mapping from id to sn
        fs::path fnSn(m_filename);
        fnSn.replace_extension(".sn");


        const auto idToSn = MLvxCalib::GetIdToSnMapping(fnSn.string());
        auto calibration = MLvxCalib::CombineIntoCalibration(idToSn, m_calibration);
        auto data = load_point_cloud(m_filename.c_str(), true, m_filter_threshold_xy, calibration);
        m_data = data;
    }

    void ReleaseData()
    {
        m_data.clear();
    }

    const std::vector<Point3Di>& GetData() const
    {
        return m_data;
    }

    std::vector<Point3Di>& GetData()
    {
        return m_data;
    }

};


namespace globals
{
    float rotate_x = 0.0, rotate_y = 0.0;
    float translate_x, translate_y = 0.0;
    float translate_z = -50.0;
    const unsigned int window_width = 800;
    const unsigned int window_height = 600;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    std::string working_directory = "";
    int mouse_buttons = 0;
    int mouse_old_x, mouse_old_y;
    bool gui_mouse_down{ false };
    float mouse_sensitivity = 1.0;

    Params::Params params;

    std::mutex mtx;
    std::vector<RegisteredFrame> registeredFrames;
    std::vector<DelayedPoints> pointsPerFile;
    std::vector<Eigen::Vector3d> localMap;
    std::thread icpThread;
    std::atomic<bool> icpRunning{ false };
    std::atomic<float> currentGnssSpeed { 0.0 };
    std::atomic<float> icpProgress{ 0.0 };
    std::map<double, Eigen::Matrix4d> imuTrajectory;
    std::map<double, float> gnssSpeed;

} // namespace globals


bool LoadData(std::vector<std::string> input_file_names)
{
    std::sort(std::begin(input_file_names), std::end(input_file_names));

    std::vector<std::string> laz_files;
    std::vector<std::string> csv_files;
    std::vector<std::string> nmea_files;
    std::for_each(
        std::begin(input_file_names),
        std::end(input_file_names),
        [&](const std::string& fileName)
        {
            if (fileName.ends_with(".laz") || fileName.ends_with(".las"))
            {
                laz_files.push_back(fileName);
            }
            else if (fileName.ends_with(".csv"))
            {
                csv_files.push_back(fileName);
            }
            else if (fileName.ends_with(".nmea"))
            {
                nmea_files.push_back(fileName);
            }
        });

    if (input_file_names.size() > 0)
    {
        globals::working_directory = fs::path(input_file_names.front()).parent_path().string();

        const auto calibrationFile = (fs::path(globals::working_directory) / "calibration.json").string();
        const auto preloadedCalibration = MLvxCalib::GetCalibrationFromFile(calibrationFile);
        const std::string imuSnToUse = MLvxCalib::GetImuSnToUse(calibrationFile);

        fs::path wdp = fs::path(input_file_names[0]).parent_path();
        wdp /= "preview";
        if (!fs::exists(wdp))
        {
            fs::create_directory(wdp);
        }
        globals::pointsPerFile.resize(laz_files.size());

        // nmea
        for (const auto& nmeaFile : nmea_files)
        {
            auto speeds = mandeye::GetSpeedFromNMEA(nmeaFile);
            for (const auto& [timestamp, speed] : speeds)
            {
                globals::gnssSpeed[timestamp] = speed;
            }
        }

        std::transform(
#ifndef __APPLE__
            std::execution::par,
#endif
            std::begin(laz_files),
            std::end(laz_files),
            std::begin(globals::pointsPerFile),
            [&](const std::string& fn)
            {
                auto d= DelayedPoints(fn, preloadedCalibration, globals::params.filter_threshold_xy);
                if (!globals::params.loadDataDuringICP)
                {
                    d.LoadData();
                }
                return d;
            });

        std::cout << "std::transform finished" << std::endl;

        // load IMU data
        std::vector<std::tuple<std::pair<double, double>, FusionVector, FusionVector>> imu_data;

        for (size_t fileNo = 0; fileNo < csv_files.size(); fileNo++)
        {
            const std::string &imufn = csv_files.at(fileNo);
            fs::path fnSn(imufn);
            fnSn.replace_extension(".sn");

            // GetId of Imu to use
            const auto idToSn = MLvxCalib::GetIdToSnMapping(fnSn.string());
            // GetId of Imu to use
            int imuNumberToUse = MLvxCalib::GetImuIdToUse(idToSn, imuSnToUse);
            std::cout << "imuNumberToUse  " << imuNumberToUse << " at: '" << imufn << "'" << std::endl;
            auto imu = load_imu(imufn.c_str(), imuNumberToUse);
            std::cout << imufn << " with mapping " << fnSn << std::endl;
            imu_data.insert(std::end(imu_data), std::begin(imu), std::end(imu));
        }
        std::sort(
            imu_data.begin(),
            imu_data.end(),
            [](const std::tuple<std::pair<double, double>, FusionVector, FusionVector>& a,
               const std::tuple<std::pair<double, double>, FusionVector, FusionVector>& b)
            {
                return std::get<0>(a).first < std::get<0>(b).first;
            });

        FusionAhrs ahrs;
        FusionAhrsInitialise(&ahrs);


        int counter =0;
        for (const auto &[timestamp_pair, gyr, acc] : imu_data)
        {
            const FusionVector gyroscope = {static_cast<float>(gyr.axis.x * 180.0 / M_PI), static_cast<float>(gyr.axis.y * 180.0 / M_PI), static_cast<float>(gyr.axis.z * 180.0 / M_PI)};
            // const FusionVector gyroscope = {static_cast<float>(gyr.axis.x), static_cast<float>(gyr.axis.y), static_cast<float>(gyr.axis.z)};
            const FusionVector accelerometer = {acc.axis.x, acc.axis.y, acc.axis.z};
            static bool first = true;

            static double last_ts;
            if (first)
            {
                FusionAhrsUpdateNoMagnetometer(&ahrs, gyroscope, accelerometer, 1/200.0); // initial update with 100ms
                first = false;
                // last_ts = timestamp_pair.first;
            }
            else
            {
                double curr_ts = timestamp_pair.first;

                double ts_diff = curr_ts - last_ts;

                FusionAhrsUpdateNoMagnetometer(&ahrs, gyroscope, accelerometer, ts_diff);
            }

            last_ts = timestamp_pair.first;
            //

            FusionQuaternion quat = FusionAhrsGetQuaternion(&ahrs);

            Eigen::Quaterniond d{quat.element.w, quat.element.x, quat.element.y, quat.element.z};
            Eigen::Affine3d t{Eigen::Matrix4d::Identity()};
            t.rotate(d);

            globals::imuTrajectory[timestamp_pair.first] = t.matrix();
        }

        return true;
    }

    return false;
}

// mandeye_controller datasets keep extra GNSS receiver logs in an
// EXTRA_GNSS/ subfolder next to the lidar/imu files; pull its *.nmea
// files in since the main directory scan is not recursive.
std::vector<std::string> CollectExtraGnssFiles(const fs::path& dir)
{
    std::vector<std::string> files;
    const auto extraGnssDir = dir / "EXTRA_GNSS";
    if (fs::exists(extraGnssDir) && fs::is_directory(extraGnssDir))
    {
        for (const auto& entry : fs::directory_iterator(extraGnssDir))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".nmea")
            {
                files.push_back(entry.path().string());
            }
        }
    }
    return files;
}

void LoadDataButton()
{
    static std::shared_ptr<pfd::open_file> open_file;
    std::vector<std::string> input_file_names;
    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, (bool)open_file);
    const auto t = [&]()
    {
        std::vector<std::string> filters;
        auto sel = pfd::open_file("Load las files", fs::current_path().string(), filters, true).result();
        for (int i = 0; i < sel.size(); i++)
        {
            input_file_names.push_back(sel[i]);
        }
    };
    std::thread t1(t);
    t1.join();

    if (!input_file_names.empty())
    {
        // ok, let try to scan directory
        const auto firstFilename = input_file_names.front();

        auto dir = fs::path(firstFilename).parent_path();
        std::cout << "Trying to scan directory: " << dir << std::endl;
        std::vector<std::string> allFiles;
        for (const auto& entry : fs::directory_iterator(dir))
        {
            if (entry.is_regular_file())
            {
                const auto fn = entry.path().string();
                if (fn.ends_with(".laz") || fn.ends_with(".las") || fn.ends_with(".csv") || fn.ends_with(".nmea"))
                {
                    allFiles.push_back(fn);
                }
            }
        }
        const auto extraGnssFiles = CollectExtraGnssFiles(dir);
        allFiles.insert(allFiles.end(), extraGnssFiles.begin(), extraGnssFiles.end());
        if (LoadData(allFiles))
        {
            return;
        }
    }


    pfd::message("Error", "please select files correctly", pfd::choice::ok);
    std::cout << "please select files correctly" << std::endl;

}

std::tuple<kiss_icp::pipeline::KissICP::Vector3dVector, kiss_icp::pipeline::KissICP::Vector3dVector, Sophus::SE3d> PerformICPOnFrame(const RegisteredFrame& frame, kiss_icp::pipeline::KissICP & icp)
{
        Sophus::SE3d imuUpdate;
          const double query_time1 = frame.timestamp_hardware.back();
            const double query_time2 = query_time1 + globals::params.timestamp_per_icp;
            const auto m1 = getInterpolatedPose(globals::imuTrajectory, query_time1);
            const auto m2 = getInterpolatedPose(globals::imuTrajectory, query_time2);
            if (!m1.isZero() && !m2.isZero())
            {
                const Eigen::Affine3d imuPose1 {m1};
                const Eigen::Affine3d imuPose2 {m2};

                const Eigen::Affine3d imuPose = imuPose1.inverse() * imuPose2;
                imuUpdate = Sophus::SE3d::fitToSE3(imuPose.matrix());
                //rotationIMU << std::setprecision(20) << frame.timestamp_hardware.front()  << "," << imuUpdate.inverse().log().transpose().format(CSVFormat)<< std::endl;
                if (globals::params.useImu)
                {
                    auto tangentImu = imuUpdate.log();
                    auto tangentDelta = icp.delta().log();
                    tangentDelta[3] = tangentImu[3]; // keep z rotation
                    tangentDelta[4] = tangentImu[4]; // keep x rotation
                    tangentDelta[5] = tangentImu[5]; // keep y rotation
                    icp.delta() =  Sophus::SE3d::exp(tangentDelta);
                }

            }

            if (globals::params.useGNSSSpeed)
            {
                // get speed at query_time2
                auto it = std::lower_bound(globals::gnssSpeed.begin(), globals::gnssSpeed.end(), query_time2,
                    [](const auto& pair, double time) { return pair.first < time; });

                if ( it != globals::gnssSpeed.end())
                {
                    const double gnssSpeed = it->second;

                    globals::currentGnssSpeed.store(gnssSpeed);
                    if (gnssSpeed > globals::params.minGNSSSpeed)
                    {
                        auto tangent = icp.delta().log();
                        tangent[0] = it->second  * globals::params.timestamp_per_icp;
                        icp.delta() = Sophus::SE3d::exp(tangent);
                    }
                }

            }
            const auto [a, b] = icp.RegisterFrame(frame.points, frame.timestamps_offset);
            return std::tuple(a,b,imuUpdate);
}


void IcpButton()
{
    if (globals::icpRunning)
    {
        return;
    }
    if (globals::pointsPerFile.size() == 0)
    {
        pfd::message("Error", "please load data first", pfd::choice::ok);
        return;
    }
    globals::icpRunning.store(true);
    std::thread icpThread(
        []()
        {
            const auto startTime = std::chrono::high_resolution_clock::now();

            using namespace kiss_icp::pipeline;
            KissICP icp(globals::params.icp_config);

           //  std::ofstream gnssSpeeds;
           //  gnssSpeeds.open("gnss_speeds.csv");
           // // gnssSpeeds << "timestamp,speed" << std::endl;
           //  for (auto & [timestamp, speed] : globals::gnssSpeed)
           //  {
           //      gnssSpeeds << std::setprecision(20) <<  double(timestamp) << "," << speed * globals::params.timestamp_per_icp << std::endl;
           //  }
            // std::ofstream rotationICP;
            // rotationICP.open("rotation_icp.csv");
            // rotationICP << "timestamp,delta_x,delta_y,delta_z,delta_qx,delta_qy,delta_qz" << std::endl;
            // std::ofstream rotationIMU;
            // rotationIMU.open("rotation_imu.csv");
            // rotationICP << "timestamp,delta_x,delta_y,delta_z,delta_qx,delta_qy,delta_qz" << std::endl;

            const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");


            globals::registeredFrames.resize(1);

            double last_timestamp = 0;

            const double experimentStartTime = globals::imuTrajectory.begin()->first;
            const double experimentEndTime = globals::imuTrajectory.rbegin()->first;
            const double experimentDuration = experimentEndTime - experimentStartTime;

            for (auto& file : globals::pointsPerFile)
            {
                file.LoadData();
                for (const auto& point : file.GetData())
                {
                    const double timestamp = point.timestamp;
                    if (timestamp > 0)
                    {
                        if (last_timestamp == 0)
                        {
                            last_timestamp = timestamp;
                        }
                        auto& lastFrame = globals::registeredFrames.back();
                        lastFrame.points.emplace_back(point.point);
                        lastFrame.intensities.emplace_back(point.intensity);
                        double deltaTime = point.timestamp - last_timestamp;
                        lastFrame.timestamps_offset.emplace_back(deltaTime);
                        lastFrame.timestamp_hardware.emplace_back(point.timestamp);
                        if (deltaTime > globals::params.timestamp_per_icp)
                        {
                            auto& frame = globals::registeredFrames.back();

                            auto [registered_frame, registered_frame_timestamps, imuUpdate] = PerformICPOnFrame(frame, icp);
                            //rotationICP << std::setprecision(20) << frame.timestamp_hardware.front()  << "," << icp.delta().log().transpose().format(CSVFormat)<< std::endl;
                            //rotationIMU << std::setprecision(20) << frame.timestamp_hardware.front()  << "," << imuUpdate.inverse().log().transpose().format(CSVFormat)<< std::endl;
                            std::unique_lock lck(globals::mtx);
                            frame.pose = Eigen::Affine3d(icp.pose().matrix());
                            globals::localMap = icp.LocalMap();
                            const double durIcp = timestamp - experimentStartTime;
                            globals::icpProgress.store(durIcp / experimentDuration);

                            // cache data and release for save RAM
                            frame.Cache();
                            frame.RelaseData();
                            last_timestamp = timestamp;
                            globals::registeredFrames.emplace_back();
                        }
                    }
                }
                file.ReleaseData();
            }

            globals::icpRunning.store(false);
            const auto endTime = std::chrono::high_resolution_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
            std::cout << "computation completed, took " << elapsed.count() << " seconds" << std::endl;

        });
    icpThread.detach();
}

void SaveMetascan(const std::string& metcanPath)
{
    std::cout << "Saving metascan to: " << metcanPath << std::endl;
    auto metascan = CreateMetascan(globals::registeredFrames);
    const fs::path lazFilename = metcanPath;
    saveLaz(lazFilename.string(), metascan);
}



void SaveSession(const std::string& resultName = "lidar_odometry_result_kiss_0")
{
    const fs::path resultDir = fs::path(globals::working_directory) / resultName;
    std::cout << "Saving session to: " << resultDir << std::endl;
    fs::create_directory(resultDir);
    std::vector<Eigen::Affine3d> poses;
    std::vector<std::string> lioLazFiles;
    auto concatframes = ConcatenateFrames(globals::registeredFrames);
    for (size_t i = 0; i < concatframes.size(); ++i)
    {
        auto& frame = concatframes[i];
        auto vecPoints = toPoint3DiV(frame.points, frame.intensities, frame.timestamps_offset);
        const auto fn = ("scan_lio_" + std::to_string(i) + ".laz");
        lioLazFiles.push_back(fn);
        poses.push_back(frame.pose);
        const fs::path lazFilename = resultDir / fs::path(fn);
        const fs::path trjFileName = resultDir / ("trajectory_lio_" + std::to_string(i) + ".csv");
        saveLaz(lazFilename.string(), vecPoints);
        SaveTrj(trjFileName.string(), frame.trajectory);

    }
    const fs::path path(resultDir);
    const fs::path pathReg = path / "poses.reg";
    const fs::path pathRegInitial = path / "lio_initial_poses.reg";

    const fs::path pathSession = path / "session.json";


    save_poses(pathReg.string(), poses, lioLazFiles);
    save_poses(pathRegInitial.string(), poses, lioLazFiles);

    nlohmann::json jj;
    nlohmann::json j;
    j["offset_x"] = 0.0;
    j["offset_y"] = 0.0;
    j["offset_z"] = 0.0;
    j["lidar_odometry_version"] = "v0.72.0";
    j["folder_name"] =resultDir;
    j["out_folder_name"] =resultDir;
    j["poses_file_name"] = pathReg.string();
    j["initial_poses_file_name"] = pathRegInitial.string();
    j["out_poses_file_name"] = pathReg.string();
    jj["Session Settings"] = j;
    nlohmann::json jlaz_file_names;
    for (const auto& lioFileName : lioLazFiles)
    {
        auto filename = path / lioFileName;
        std::cout << "adding file: " << filename << std::endl;

        nlohmann::json jfn{
            {"file_name", filename.string()}};
        jlaz_file_names.push_back(jfn);
    }
    jj["laz_file_names"] = jlaz_file_names;
    std::ofstream o(pathSession);
    o << std::setw(4) << jj << std::endl;
}

void lidar_odometry_gui()
{
    if (ImGui::Begin("lidar_odometry_step_1-kiss-icp"))
    {
        ImGui::Text("This program is first step in MANDEYE process.");
        ImGui::Text("It results trajectory and point clouds as single session for "
                    "'multi_view_tls_registration_step_2' program.");
        ImGui::Text("It saves session.json file in 'Working directory'.");
        ImGui::Text("Next step will be to load session.json file with "
                    "'multi_view_tls_registration_step_2' program.");
        ImGui::SameLine();
        ImGui::Text("Select all imu *.csv and lidar *.laz files produced by "
                    "MANDEYE saved in 'continousScanning_*' folder");
        ImGui::Separator();
        if (ImGui::Button("load data"))
        {
            LoadDataButton();
        }

        if (ImGui::Button("icp"))
        {
            IcpButton();
        }
        if (ImGui::Button("save session"))
        {
            if (globals::pointsPerFile.size() == 0)
            {
                pfd::message("Error", "please load data first", pfd::choice::ok);
                return;
            }
            SaveSession();
        }
        if (ImGui::Button("save metascan"))
        {
            if (globals::pointsPerFile.size() == 0)
            {
                pfd::message("Error", "please load data first", pfd::choice::ok);
                return;
            }
            SaveMetascan("metascan.laz");
        }
        if (globals::icpRunning)
        {
            ImGui::ProgressBar(globals::icpProgress);
            ImGui::Text("Speed from GNSS: %.2f m/s (%.2f km/h)", globals::currentGnssSpeed.load(), globals::currentGnssSpeed.load() * 3.6);
        }

        ImGui::Separator();
        ImGui::Text("Parameters");
        ImGui::InputDouble("filter_threshold_xy", &globals::params.filter_threshold_xy);
        ImGui::InputDouble("timestamp_per_icp", &globals::params.timestamp_per_icp);
        //ImGui::InputInt("decimation", &globals::params.decimation);
        ImGui::Separator();
        ImGui::Text("ICP Parameters");
        ImGui::InputDouble("voxel_size", &globals::params.icp_config.voxel_size);
        ImGui::InputDouble("max_range", &globals::params.icp_config.max_range);
        ImGui::InputDouble("min_range", &globals::params.icp_config.min_range);
        ImGui::InputInt("max_points_per_voxel", &globals::params.icp_config.max_points_per_voxel);
        ImGui::InputDouble("min_motion_th", &globals::params.icp_config.min_motion_th);
        ImGui::InputDouble("initial_threshold", &globals::params.icp_config.initial_threshold);
        ImGui::InputInt("max_num_iterations", &globals::params.icp_config.max_num_iterations);
        ImGui::InputDouble("convergence_criterion", &globals::params.icp_config.convergence_criterion);
        ImGui::InputInt("max_num_threads", &globals::params.icp_config.max_num_threads);
        ImGui::Checkbox("deskew", &globals::params.icp_config.deskew);
        ImGui::Checkbox("use imu", &globals::params.useImu);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("If checked, IMU data will be used for initial guess. ");
        ImGui::Checkbox("use GNSS speed", &globals::params.useGNSSSpeed);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("If checked, GNSS speed will be used for guess. ");
        ImGui::InputFloat("minimum GNSS speed (m/s)", &globals::params.minGNSSSpeed);
        ImGui::Checkbox("Load Data during ICP", &globals::params.loadDataDuringICP);
        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip("If checked, data will be loaded during ICP. "
                              "This reduces memory consumption, but increases processing time."
                              "It also cache intermediatate data. ");
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Mimimum GNSS speed to use for guess. ");

        if (ImGui::Button("saveParams"))
        {
            std::ofstream files("params.json");
            auto j = Params::ParamsToJson(globals::params);
            files << j.dump(10);
        }
        RegisteredFrame::CacheEnabled = globals::params.loadDataDuringICP;
    }

    ImGui::End();
}

void mouse(int glut_button, int state, int x, int y)
{
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2((float)x, (float)y);
    int button = -1;
    if (glut_button == GLUT_LEFT_BUTTON)
        button = 0;
    if (glut_button == GLUT_RIGHT_BUTTON)
        button = 1;
    if (glut_button == GLUT_MIDDLE_BUTTON)
        button = 2;
    if (button != -1 && state == GLUT_DOWN)
        io.MouseDown[button] = true;
    if (button != -1 && state == GLUT_UP)
        io.MouseDown[button] = false;


    if (!io.WantCaptureMouse)
    {
        if (state == GLUT_DOWN)
        {
            globals::mouse_buttons |= 1 << glut_button;
        }
        else if (state == GLUT_UP)
        {
            globals::mouse_buttons = 0;
        }
        globals::mouse_old_x = x;
        globals::mouse_old_y = y;
    }
}

void wheel(int button, int dir, int x, int y)
{
    // Use reciprocal zoom-in/zoom-out factors so that an equal number of
    // in/out events exactly cancels out, regardless of order. Trackpads
    // (macOS in particular) can deliver many jittery, alternating-direction
    // wheel events per gesture; non-reciprocal factors (e.g. 0.95 in /
    // 1.05 out) compound into a net drift on every such gesture, which
    // shows up as the view being unable to zoom out and "jumping" as the
    // jitter is applied.
    constexpr float zoomFactor = 1.05f;
    if (dir > 0)
    {
        globals::translate_z /= zoomFactor;
    }
    else
    {
        globals::translate_z *= zoomFactor;
    }
}

void reshape(int w, int h)
{
    glViewport(0, 0, (GLsizei)w, (GLsizei)h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)w / (GLfloat)h, 0.01, 10000.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void motion(int x, int y)
{
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2((float)x, (float)y);
    using namespace globals;

    if (!io.WantCaptureMouse)
    {
        float dx, dy;
        dx = (float)(x - mouse_old_x);
        dy = (float)(y - mouse_old_y);

        gui_mouse_down = mouse_buttons > 0;
        if (mouse_buttons & 1)
        {
            rotate_x += dy * 0.2f;
            rotate_y += dx * 0.2f;
        }
        if (mouse_buttons & 4)
        {
            translate_x += dx * 0.5f * mouse_sensitivity;
            translate_y -= dy * 0.5f * mouse_sensitivity;
        }

        mouse_old_x = x;
        mouse_old_y = y;
    }
    glutPostRedisplay();
}

void coordinateSystem(float s)
{
    glBegin(GL_LINES);

    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(s, 0.0f, 0.0f);

    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, s, 0.0f);

    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, s);
    glEnd();
}
void display()
{
    ImGuiIO& io = ImGui::GetIO();
    glViewport(0, 0, (GLsizei)io.DisplaySize.x, (GLsizei)io.DisplaySize.y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float ratio = float(io.DisplaySize.x) / float(io.DisplaySize.y);

    glClearColor(
        globals::clear_color.x * globals::clear_color.w,
        globals::clear_color.y * globals::clear_color.w,
        globals::clear_color.z * globals::clear_color.w,
        globals::clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    reshape((GLsizei)io.DisplaySize.x, (GLsizei)io.DisplaySize.y);

    {
        using namespace globals;
        glTranslatef(translate_x, translate_y, translate_z);
        glRotatef(rotate_x, 1.0, 0.0, 0.0);
        glRotatef(rotate_y, 0.0, 0.0, 1.0);
    }

    coordinateSystem(100);

    {
        std::unique_lock lck(globals::mtx);
        for (int i = 0; i < globals::registeredFrames.size(); ++i)
        {
            if (i == globals::registeredFrames.size() - 1)
            {
                continue;
            }
            const auto& frame = globals::registeredFrames[i];
            glPushMatrix();
            Eigen::Matrix4d mat = frame.pose.matrix();
            glMultMatrixd(mat.data());
            coordinateSystem(1);
            glPopMatrix();
        }
    }
    glBegin(GL_POINTS);
    std::vector<Eigen::Vector3d> points;
    {
        std::unique_lock lck(globals::mtx);
        points = globals::localMap;
    }
    for (const auto& point : points)
    {
        glVertex3dv(point.data());
    }

    glEnd();
    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGLUT_NewFrame();

    lidar_odometry_gui();

    ImGui::Render();
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

    glutSwapBuffers();
    glutPostRedisplay();
}

bool initGL(int* argc, char** argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(globals::window_width, globals::window_height);
    glutCreateWindow("lidar_odometry");
    glutDisplayFunc(display);
    glutMotionFunc(motion);

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glEnable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, globals::window_width, globals::window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)globals::window_width / (GLfloat)globals::window_height, 0.01, 10000.0);
    glutReshapeFunc(reshape);
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable
    // Keyboard Controls

    ImGui::StyleColorsDark();
    ImGui_ImplGLUT_Init();
    ImGui_ImplGLUT_InstallFuncs();
    ImGui_ImplOpenGL2_Init();
    return true;
}

int main(int argc, char* argv[])
{
    std::optional<std::string> configFn;
    std::optional<std::string> dataSetToProcess;
    std::optional<std::string> resultName;
    std::optional<std::string> metaScanPath;
    bool gui = true;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            configFn = argv[i + 1];
            break;
        }
        if (arg == "--nogui")
        {
            gui = false;
        }
        if (arg == "--process" && i + 1 < argc)
        {
            dataSetToProcess = argv[i + 1];
        }
        if (arg == "--metascan" && i + 1 < argc)
        {
            metaScanPath = argv[i + 1];
        }
        if (arg == "--resultName" && i + 1 < argc)
        {
            resultName = argv[i + 1];
        }
    }

    if (configFn.has_value())
    {
        // load json
        std::ifstream file(*configFn);
        using json = nlohmann::json;
        json jsonData = json::parse(file);
        globals::params = Params::LoadParamFromJson(jsonData, globals::params);
    }
    else
    {
        std::ifstream file("params.json");;
        if (file.is_open())
        {
            using json = nlohmann::json;
            json jsonData = json::parse(file);
            globals::params = Params::LoadParamFromJson(jsonData, globals::params);
        }
    }

    if (dataSetToProcess.has_value())
    {
        std::vector<std::string> files;
        for (const auto & entry : fs::directory_iterator(*dataSetToProcess))
        {
            if (entry.is_regular_file())
            {
                files.push_back(entry.path().string());
            }
        }
        const auto extraGnssFiles = CollectExtraGnssFiles(*dataSetToProcess);
        files.insert(files.end(), extraGnssFiles.begin(), extraGnssFiles.end());

        const bool isLoadOk = LoadData(files);
        if (isLoadOk)
        {
            std::cout << "LoadData complete" << std::endl;
        }
        IcpButton();


    }

    if (gui)
    {
        initGL(&argc, argv);
        glutDisplayFunc(display);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        glutMouseWheelFunc(wheel);
        glutMainLoop();

        ImGui_ImplOpenGL2_Shutdown();
        ImGui_ImplGLUT_Shutdown();

        ImGui::DestroyContext();
    }
    else
    {
        while (globals::icpRunning){
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cout << "ICP in progress : " << 100.0 *globals::icpProgress << " % "<< std::endl;
        }
        if (dataSetToProcess.has_value())
        {
            SaveSession(resultName.value_or("lidar_odometry_result_kiss_0"));
        }
        if (metaScanPath.has_value())
        {
            SaveMetascan(*metaScanPath);
        }
    }

    // clear cached data
    RegisteredFrame::ClearCache();

    return 0;
}

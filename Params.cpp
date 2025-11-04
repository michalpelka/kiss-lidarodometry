#include "Params.h"
namespace Params
{

    nlohmann::json ParamsToJson(const Params& params)
    {
        nlohmann::json j;
        j["filter_threshold_xy"] = params.filter_threshold_xy;
        j["timestamp_per_icp"] = params.timestamp_per_icp;
        j["decimation"] =  params.decimation;
        j["icp_config"]["voxel_size"] = params.icp_config.voxel_size;
        j["icp_config"]["max_range"] = params.icp_config.max_range;
        j["icp_config"]["min_range"] = params.icp_config.min_range;
        j["icp_config"]["max_points_per_voxel"] = params.icp_config.max_points_per_voxel;
        j["icp_config"]["min_motion_th"] = params.icp_config.min_motion_th;
        j["icp_config"]["initial_threshold"] = params.icp_config.initial_threshold;
        j["icp_config"]["max_num_iterations"] = params.icp_config.max_num_iterations;
        j["icp_config"]["convergence_criterion"] = params.icp_config.convergence_criterion;
        j["icp_config"]["max_num_threads"] = params.icp_config.max_num_threads;
        j["icp_config"]["deskew"] = params.icp_config.deskew;
        j["useImu"] = params.useImu;
        j["useGNSSSpeed"] = params.useGNSSSpeed;
        j["minGNSSSpeed"] = params.minGNSSSpeed;
        j["loadDataDuringICP"] = params.loadDataDuringICP;

        return j;
    }

    Params LoadParamFromJson(const nlohmann::json &j, const Params& defaultParams)
    {
        Params params;
        LoadIfExists(j, "filter_threshold_xy", params.filter_threshold_xy);
        LoadIfExists(j, "timestamp_per_icp", params.timestamp_per_icp);
        LoadIfExists(j, "decimation", params.decimation);

        if (j.contains("icp_config") && j["icp_config"].is_object()) {
            const auto& icp = j["icp_config"];
            LoadIfExists(icp, "voxel_size", params.icp_config.voxel_size);
            LoadIfExists(icp, "max_range", params.icp_config.max_range);
            LoadIfExists(icp, "min_range", params.icp_config.min_range);
            LoadIfExists(icp, "max_points_per_voxel", params.icp_config.max_points_per_voxel);
            LoadIfExists(icp, "min_motion_th", params.icp_config.min_motion_th);
            LoadIfExists(icp, "initial_threshold", params.icp_config.initial_threshold);
            LoadIfExists(icp, "max_num_iterations", params.icp_config.max_num_iterations);
            LoadIfExists(icp, "convergence_criterion", params.icp_config.convergence_criterion);
            LoadIfExists(icp, "max_num_threads", params.icp_config.max_num_threads);
            LoadIfExists(icp, "deskew", params.icp_config.deskew);
        } else {
            std::cerr << "WARN: 'icp_config' missing or not an object. Using default ICP config.\n";
        }

        LoadIfExists(j, "useImu", params.useImu);
        LoadIfExists(j, "useGNSSSpeed", params.useGNSSSpeed);
        LoadIfExists(j, "minGNSSSpeed", params.minGNSSSpeed);
        LoadIfExists(j,"loadDataDuringICP", params.loadDataDuringICP);
        return params;
    }

}
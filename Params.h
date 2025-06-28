#pragma once
#include <nlohmann/json.hpp>
#include <kiss_icp/pipeline/KissICP.hpp>
#include <iostream>
namespace Params
{
  struct Params
  {
    double filter_threshold_xy = 0.0;
    double timestamp_per_icp = 0.05;
    kiss_icp::pipeline::KISSConfig icp_config;
    int decimation = 10;
    bool useImu = true;
    bool useGNSSSpeed = true; // use GNSS speed for initial guess
    float minGNSSSpeed = 5.0f; // minimum speed in m/s to use GNSS speed for registration
  };

  template<typename T>
  void LoadIfExists(const nlohmann::json& j, const std::string& key, T& out)
  {
    try {
      if (j.contains(key) && !j.at(key).is_null()) {
        out = j.at(key).get<T>();
      } else {
        std::cerr << "WARN: '" << key << "' missing or null. Using default.\n";
      }
    } catch (const nlohmann::json::exception& e) {
      std::cerr << "ERROR: Failed to load '" << key << "': " << e.what() << std::endl;
    }
  };

  nlohmann::json ParamsToJson(const Params& params);

  Params LoadParamFromJson(const nlohmann::json &j, const Params& defaultParams );
}
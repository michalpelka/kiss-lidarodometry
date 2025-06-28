#include <iostream>
#include <string>
#include <optional>
#include <sstream>
#include <vector>
#include <variant>
namespace mandeye
{
    struct GNRMCData {
        std::string time_utc;
        char status;
        double latitude;
        char lat_dir;
        double longitude;
        char lon_dir;
        double speed_knots;
        double track_angle;
        std::string date;
        double magnetic_variation;
        char mag_var_dir;
    };


    std::tuple<double,double,std::string> BreakLineFromNMEAFile(const std::string& line) {
        std::istringstream iss(line);
        double timestampLidar;
        double timestampUnix;
        std::string nmeaSentence;
        iss >> timestampLidar >> timestampUnix;
        char data;
        iss.read(&data, 1);
        std::getline(iss, nmeaSentence);
        return std::make_tuple(timestampLidar/1e9, timestampUnix/1e9, nmeaSentence);
    }

    inline bool validateNMEAChecksum(const std::string& nmea) {
        if (nmea.empty() || nmea[0] != '$')
        {
            return false;
        }
        auto asterisk = nmea.find('*');
        if (asterisk == std::string::npos || asterisk + 3 > nmea.size()) return false;
        unsigned char checksum = 0;
        for (size_t i = 1; i < asterisk; ++i) {
            checksum ^= nmea[i];
        }
        unsigned int expected;
        std::istringstream iss(nmea.substr(asterisk + 1, 2));
        iss >> std::hex >> expected;
        return checksum == expected;
    }


    inline std::optional<GNRMCData> parseGNRMC(const std::string& nmea) {
        if (nmea.find("$GNRMC") != 0 && nmea.find("$GPRMC") != 0)
            return std::nullopt;

        std::vector<std::string> fields;
        std::stringstream ss(nmea);
        std::string item;
        while (std::getline(ss, item, ',')) {
            fields.push_back(item);
        }
        if (fields.size() < 12) return std::nullopt;

        GNRMCData data;
        data.time_utc = fields[1];
        data.status = fields[2].empty() ? 'V' : fields[2][0];
        data.latitude = fields[3].empty() ? 0.0 : std::stod(fields[3]);
        data.lat_dir = fields[4].empty() ? 'N' : fields[4][0];
        data.longitude = fields[5].empty() ? 0.0 : std::stod(fields[5]);
        data.lon_dir = fields[6].empty() ? 'E' : fields[6][0];
        data.speed_knots = fields[7].empty() ? 0.0 : std::stod(fields[7]);
        data.track_angle = fields[8].empty() ? 0.0 : std::stod(fields[8]);
        data.date = fields[9];
        data.magnetic_variation = fields[10].empty() ? 0.0 : std::stod(fields[10]);
        data.mag_var_dir = fields[11].empty() ? 'E' : fields[11][0];

        return data;
    }

    std::map<double, double> GetSpeedFromNMEA(const std::string& nmeaFile) {
        std::map<double, double> gnssSpeed;

        // open file
        std::ifstream file(nmeaFile);
        if (!file.is_open()) {
            std::cerr << "Could not open NMEA file: " << nmeaFile << std::endl;
            return gnssSpeed;
        }
        std::string line;
        while (std::getline(file, line)) {
            const auto [timestampLidar, timestampUnix, nmeaSentence] = BreakLineFromNMEAFile(line);
            if (!validateNMEAChecksum(nmeaSentence)) {
                std::cerr << "Invalid NMEA sentence: " << nmeaSentence << std::endl;
                continue; // skip invalid sentences
            }
            auto data = parseGNRMC(nmeaSentence);
            if (!data.has_value()) {
                    continue; // skip non-GNRMC sentences
            }
            if (data->status == 'A') { // 'A' for active, 'V' for void
                    double speed = data->speed_knots * 0.514444; // convert knots to m/s
                    gnssSpeed[timestampLidar] = speed;
            }
        }
        return gnssSpeed;
    }

}

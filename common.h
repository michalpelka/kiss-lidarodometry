#pragma once

namespace common
{
//! \brief GetInterpolatedTimstampForLidarPoint
//! \param frameRate - frame rate of the lidar in seconds
//! \param startTs - timestamp of the first point in the point cloud
//! \param numPoints - number of points in the point cloud
//! \param pointNumber - number of the point to get the timestamp for
//! \return interpolated timestamp for the point in seconds
double GetInterpolatedTimstampForLidarPoint(const double frameRate, const double startTs, const int numPoints, const int pointNumber)
{
  return startTs + (pointNumber * frameRate) / numPoints;
}

}
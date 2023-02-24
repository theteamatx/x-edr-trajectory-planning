// Copyright 2023 Google LLC

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     https://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TRAJECTORY_PLANNING_TRAJECTORY_PLANNING_PATH_TOOLS_H_
#define TRAJECTORY_PLANNING_TRAJECTORY_PLANNING_PATH_TOOLS_H_

#include <type_traits>

#include "eigenmath/interpolation.h"
#include "eigenmath/types.h"
#include "eigenmath/utils.h"
#include "absl/status/statusor.h"

namespace trajectory_planning {
template <typename Scalar, int N>
struct ProjectedPointResult {
  static_assert(std::is_floating_point_v<Scalar>,
                "Scalar must be a floating point type.");

  int waypoint_index = 0;
  Scalar distance_to_path = std::numeric_limits<Scalar>::max();
  Scalar line_parameter = 0;
  eigenmath::Vector<Scalar, N> projected_point;
};

// Returns ProjectedPointResult for `point` projected on straight line segements
// between `waypoints` for which the distance is the smallest.
template <typename Scalar, int N>
absl::StatusOr<ProjectedPointResult<Scalar, N>> ProjectPointOnPath(
    absl::Span<const eigenmath::Vector<Scalar, N>> waypoints,
    const eigenmath::Vector<Scalar, N>& point);

// Returns a point computed from the location where the robot could come to a
// stop with linear motion when starting at `position` with `velocity` and not
// exceeding the given `acceleration_limit`.
// The returned point is shifted further along the offset from `position` to
// the stopping location to account for `corner_rounding`, which
// ensures that if the returned point is used as a waypoint the path timing
// problem is feasible within the acceleration limits.
absl::StatusOr<eigenmath::VectorXd> ComputeStoppingPoint(
    absl::Span<const double> position, absl::Span<const double> velocity,
    absl::Span<const double> acceleration_limit, double corner_rounding);

// Implementation details below.
template <typename Scalar, int N>
absl::StatusOr<ProjectedPointResult<Scalar, N>> ProjectPointOnPath(
    absl::Span<const eigenmath::Vector<Scalar, N>> waypoints,
    const eigenmath::Vector<Scalar, N>& point) {
  static_assert(std::is_floating_point_v<Scalar>,
                "Scalar must be a floating point type.");
  if (waypoints.empty()) {
    return absl::InvalidArgumentError("No waypoints.");
  }
  if (waypoints.size() == 1) {
    return ProjectedPointResult<Scalar, N>{
        .waypoint_index = 0,
        .distance_to_path = (waypoints.front() - point).norm(),
        .line_parameter = 0.0,
        .projected_point = waypoints.front()};
  }

  const int dim = point.size();
  ProjectedPointResult<Scalar, N> result;
  for (const auto& wp : waypoints) {
    if (wp.size() != dim) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid number of joints: point.size()= ", point.size(),
                       ", waypoint.size: ", wp.size()));
    }
  }
  // result.distance_to_path is initialized to numeric_limits::max().
  for (int i = 0; i < waypoints.size() - 1; ++i) {
    Scalar distance = 0.0;
    Scalar line_parameter = 0.0;
    eigenmath::DistanceFromLineSegment(waypoints[i], waypoints[i + 1], point,
                                       &distance, &line_parameter);
    if (distance < result.distance_to_path) {
      result.distance_to_path = distance;
      result.line_parameter = line_parameter;
      result.waypoint_index = i;
    }
  }

  result.projected_point = eigenmath::InterpolateLinear(
      result.line_parameter, waypoints[result.waypoint_index],
      waypoints[result.waypoint_index + 1]);

  return result;
}

inline absl::StatusOr<ProjectedPointResult<double, Eigen::Dynamic>>
ProjectPointOnPath(
    absl::Span<const eigenmath::Vector<double, Eigen::Dynamic>> waypoints,
    const eigenmath::Vector<double, Eigen::Dynamic>& point) {
  return ProjectPointOnPath<double, Eigen::Dynamic>(waypoints, point);
}

inline absl::StatusOr<ProjectedPointResult<double, 3>> ProjectPointOnPath(
    absl::Span<const eigenmath::Vector<double, 3>> waypoints,
    const eigenmath::Vector<double, 3>& point) {
  return ProjectPointOnPath<double, 3>(waypoints, point);
}

}  // namespace trajectory_planning

#endif  // TRAJECTORY_PLANNING_TRAJECTORY_PLANNING_PATH_TOOLS_H_

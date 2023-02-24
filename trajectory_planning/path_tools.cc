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

#include "trajectory_planning/path_tools.h"

#include <limits>

#include "trajectory_planning/splines/spline_utils.h"
#include "eigenmath/utils.h"
#include "absl/status/status.h"

namespace trajectory_planning {

absl::StatusOr<eigenmath::VectorXd> ComputeStoppingPoint(
    absl::Span<const double> position, absl::Span<const double> velocity,
    absl::Span<const double> acceleration_limit, double corner_rounding) {
  if (position.size() != velocity.size()) {
    return absl::InvalidArgumentError("position.size != velocity.size.");
  }
  if (position.size() != acceleration_limit.size()) {
    return absl::InvalidArgumentError("position.size != acceleration.size.");
  }

  Eigen::Map<const eigenmath::VectorXd> pos(position.data(), position.size());
  Eigen::Map<const eigenmath::VectorXd> vel(velocity.data(), velocity.size());
  Eigen::Map<const eigenmath::VectorXd> acc(acceleration_limit.data(),
                                            acceleration_limit.size());
  if (acc.minCoeff() <= 0.0) {
    return absl::InvalidArgumentError(
        "Acceleration limits must be strictly positive.");
  }
  if (vel.cwiseAbs().maxCoeff() <
      std::numeric_limits<double>::epsilon() * 100) {
    return pos;
  }

  // Compute maximum stopping desceleration that is collinear to `velocity` and
  // within `acceleration_limits`.
  const eigenmath::VectorXd stopping_deceleration =
      eigenmath::ScaleDownToLimits((-vel.normalized() * acc.maxCoeff()).eval(),
                                   acc.eval());
  // The velocity trajectory is: vel+time*stopping_desceleration.
  // As vel and stopping_desceleration are parallel, this is essentially a
  // scalar equation in the 'vel' direction.
  // All velocity components are zero at the same time.
  // As |velocity| > 0, but some components might be 0, the time is
  // computed after projecting onto the equations onto the velocity direction
  // via the scalar product.
  const double stop_time = -vel.squaredNorm() / vel.dot(stopping_deceleration);
  // Offset from `position` at which we could stop.
  const eigenmath::VectorXd stopping_delta =
      stop_time * (vel + 0.5 * stop_time * stopping_deceleration);
  // An additional offset to account for spline corner rounding, such that if
  // the returned point is added as a waypoint, the resulting spline will have
  // a straight line from `position` until `position+stopping_delta`.
  // This is highly conservative and probably not necessary, but should
  // guarantee that maximum accelerations are not violated due to corner
  // rounding. Consider removing this.
  const eigenmath::VectorXd corner_rounding_distance =
      PolyLineToBspline3WaypointsCornerOffset(stopping_delta,
                                                       corner_rounding);
  return pos + stopping_delta + corner_rounding_distance;
}

}  // namespace trajectory_planning

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

#include "trajectory_planning/splines/spline_utils.h"

#include <algorithm>
#include <limits>
#include <vector>

#include "absl/log/log.h"


namespace trajectory_planning {
eigenmath::VectorXd PolyLineToBspline3WaypointsCornerOffset(
    const eigenmath::VectorXd& delta, const double radius) {
  eigenmath::VectorXd offset = eigenmath::VectorXd::Zero(delta.size());
  constexpr double kMinNorm = 1e-6;
  const double norm = delta.norm();
  if (norm > kMinNorm) {
    offset = delta / norm;
  } else {
    offset.setZero();
  }
  // Make sure inner points for rounding corners maintain ordering along path
  // and do not introduce duplicate control points, as those reduce the
  // continuity of the spline.
  if (norm > spline_utils_details::kMinWaypointSpacingFactor * radius) {
    offset = offset * radius;
  } else {
    offset =
        offset * (1.0 / spline_utils_details::kMinWaypointSpacingFactor) * norm;
  }
  return offset;
}

template <>
void PolyLineToBspline3Waypoints(
    const std::vector<eigenmath::VectorXd>& corners, const double radius,
    std::vector<eigenmath::VectorXd>* output) {
  CHECK(nullptr != output);

  eigenmath::VectorXd::Index ndof{corners.front().rows()};

  // Special case: one point.
  if (1 == corners.size()) {
    *output = {corners.front(), corners.front(), corners.front(),
               corners.front()};
    return;
  }

  // Check for consistently sized points.
  for (auto& corn : corners) {
    CHECK(ndof == corn.rows());
  }

  // Construct bspline path by inserting one point before and one after each
  // corner of the piecewise-linear path. This ensures straight lines between
  // inner points and rounded corners.
  output->resize(3 * corners.size() - 2);
  // copy corners as spline control points
  for (size_t idx = 0; idx < corners.size(); idx++) {
    (*output)[3 * idx] = corners[idx];
  }
  // Add remaining points at radius-offset from corners on straight line
  // connecting corners.
  eigenmath::VectorXd offset(ndof);
  for (size_t idx = 1; idx < corners.size() - 1; idx++) {
    const size_t k = 3 * idx;
    const size_t knext = 3 * (idx + 1);
    const size_t klast = 3 * (idx - 1);
    const size_t kp = k + 1;
    const size_t km = k - 1;

    offset = PolyLineToBspline3WaypointsCornerOffset(
        (*output)[knext] - (*output)[k], radius);
    (*output)[kp] = (*output)[k] + offset;

    offset = PolyLineToBspline3WaypointsCornerOffset(
        (*output)[klast] - (*output)[k], radius);
    (*output)[km] = (*output)[k] + offset;
  }
  // Special-case first and last inner points.
  offset = PolyLineToBspline3WaypointsCornerOffset((*output)[3] - (*output)[0],
                                                   radius);
  (*output)[1] = (*output)[0] + offset;

  const std::size_t sz = output->size();
  offset = PolyLineToBspline3WaypointsCornerOffset(
      (*output)[sz - 4] - (*output)[sz - 1], radius);
  (*output)[sz - 2] = (*output)[sz - 1] + offset;
}

namespace {
eigenmath::Pose3d CornerOffset(const eigenmath::Pose3d& delta,
                               const double translation_radius,
                               const double rotation_radius) {
  eigenmath::Pose3d offset;

  static constexpr double kMinRadius = 1e-6;
  if (translation_radius < kMinRadius || rotation_radius < kMinRadius) {
    // Zero radius -> extra control point will be added on top of existing
    // point.
    return offset;
  }

  // Calculate translation norm and rotation angle.
  const double translation_norm = delta.translation().norm();
  Eigen::AngleAxis<double> delta_rotation(delta.quaternion());
  const double rotation_angle = delta_rotation.angle();

  // Use most conservative offset percentage to not exceed either radius.
  double offset_pct_trans = translation_norm == 0.0
                                ? std::numeric_limits<double>::infinity()
                                : translation_radius / translation_norm;
  double offset_pct_rot = rotation_angle == 0.0
                              ? std::numeric_limits<double>::infinity()
                              : rotation_radius / rotation_angle;
  double offset_pct = std::min(offset_pct_trans, offset_pct_rot);

  // Make sure inner points for rounding corners maintain ordering along path
  // and do not introduce duplicate control points, as those reduce the
  // continuity of the spline.
  if (offset_pct > (1.0 / spline_utils_details::kMinWaypointSpacingFactor)) {
    offset_pct = (1.0 / spline_utils_details::kMinWaypointSpacingFactor);
  }

  // Calculate offset as percentage of delta
  offset.translation() = delta.translation() * offset_pct;
  delta_rotation.angle() *= offset_pct;
  offset.setQuaternion(eigenmath::Quaterniond(delta_rotation));

  return offset;
}
}  // namespace

void PolyLineToBspline3Waypoints(const std::vector<eigenmath::Pose3d>& corners,
                                 const double translation_radius,
                                 const double rotational_radius,
                                 std::vector<eigenmath::Pose3d>* output) {
  CHECK(output != nullptr);

  // special case: one point
  if (corners.size() == 1) {
    *output = {corners.front(), corners.front(), corners.front(),
               corners.front()};
    return;
  }

  // Construct bspline path by inserting one point before and one after each
  // corner of the piecewise-linear path. This ensures straight lines between
  // inner points and rounded corners.
  output->resize(3 * corners.size() - 2);
  // copy corners as spline control points
  for (size_t idx = 0; idx < corners.size(); idx++) {
    (*output)[3 * idx] = corners[idx];
  }
  // add remaining points at radius-offset from corners on straight line
  // connecting corners
  for (size_t idx = 1; idx < corners.size() - 1; idx++) {
    const size_t k = 3 * idx;
    const size_t knext = 3 * (idx + 1);  // next corner
    const size_t klast = 3 * (idx - 1);  // previous corner
    const size_t kp = k + 1;             // inner point after corner
    const size_t km = k - 1;             // inner point before corner

    eigenmath::Pose3d current_t_next_corner =
        (*output)[k].inverse() * (*output)[knext];
    eigenmath::Pose3d current_t_next_inner_point = CornerOffset(
        current_t_next_corner, translation_radius, rotational_radius);
    (*output)[kp] = (*output)[k] * current_t_next_inner_point;

    eigenmath::Pose3d current_pose_previous_corner =
        (*output)[k].inverse() * (*output)[klast];
    eigenmath::Pose3d current_pose_previous_inner_point = CornerOffset(
        current_pose_previous_corner, translation_radius, rotational_radius);
    (*output)[km] = (*output)[k] * current_pose_previous_inner_point;
  }

  // Special-case: first inner point.
  eigenmath::Pose3d first_t_next_corner = (*output)[0].inverse() * (*output)[3];
  eigenmath::Pose3d first_t_first_inner_point =
      CornerOffset(first_t_next_corner, translation_radius, rotational_radius);
  (*output)[1] = (*output)[0] * first_t_first_inner_point;

  // Special_case: last inner point.
  const std::size_t sz = output->size();
  eigenmath::Pose3d last_t_previous_corner =
      (*output)[sz - 1].inverse() * (*output)[sz - 4];
  eigenmath::Pose3d last_t_last_inner_point = CornerOffset(
      last_t_previous_corner, translation_radius, rotational_radius);
  (*output)[sz - 2] = (*output)[sz - 1] * last_t_last_inner_point;
}
}  // namespace trajectory_planning

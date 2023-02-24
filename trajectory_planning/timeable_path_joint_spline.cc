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

#include "trajectory_planning/timeable_path_joint_spline.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "eigenmath/types.h"
#include "trajectory_planning/path_tools.h"
#include "trajectory_planning/splines/spline_utils.h"

namespace trajectory_planning {

namespace {
constexpr double kSmall = 1e-4;

#define CHECK_WITH_MSG(condition, error_message, args...) \
  char buffer[256];                                       \
  snprintf(buffer, 256, error_message, args);             \
  ABSL_CHECK_IMPL((condition), buffer);

}  // namespace

constexpr int TimeableJointSplinePath::kSplineOrder;

TimeableJointSplinePath::TimeableJointSplinePath(
    const JointPathOptions& options)
    : options_(options),
      // one velocity constraint per dof, one acceleration constraint per dof.
      num_constraints_(options_.num_dofs() * 2),
      path_state_(State::kNoPath),
      initial_velocity_(eigenmath::VectorXd::Zero(options_.num_dofs())) {
  CHECK(options_.num_path_samples() >= 3);
  spline_sampled_.resize(3);
  for (auto& s : spline_sampled_) {
    s.resize(options.num_dofs());
  }
  path_position_.resize(options_.num_path_samples());
  first_path_derivative_.resize(options_.num_path_samples());
  second_path_derivative_.resize(options_.num_path_samples());
  for (int idx = 0; idx < options_.num_path_samples(); idx++) {
    path_position_[idx].resize(options_.num_dofs());
    path_position_[idx].setZero();
    first_path_derivative_[idx].resize(options_.num_dofs());
    first_path_derivative_[idx].setZero();
    second_path_derivative_[idx].resize(options_.num_dofs());
    second_path_derivative_[idx].setZero();
  }

  constraints_.resize(options_.num_path_samples());
  for (auto& constraint : constraints_) {
    constraint.resize(num_constraints_);
  }
  max_joint_velocity_.resize(options_.num_dofs());
  max_joint_acceleration_.resize(options_.num_dofs());
  initial_velocity_.resize(options_.num_dofs());
  max_joint_acceleration_.setZero();
  max_joint_velocity_.setZero();
  initial_velocity_.setZero();
  Reset();
}

void TimeableJointSplinePath::Reset() {
  waypoints_.clear();
  control_points_.clear();
  path_state_ = State::kNoPath;
  parameter_start_ = std::numeric_limits<double>::quiet_NaN();
  parameter_end_ = std::numeric_limits<double>::quiet_NaN();
  initial_velocity_.setZero();
}

absl::Status TimeableJointSplinePath::SetMaxJointVelocity(
    absl::Span<const double> max_velocity) {
  if (max_velocity.size() != options_.num_dofs()) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("Dimension error, need ", options_.num_dofs(),
                     " joint values but max_velocity has ",
                     max_velocity.size()));
  }
  max_joint_velocity_ = FromSpan(max_velocity);
  return absl::OkStatus();
}

absl::Status TimeableJointSplinePath::SetMaxJointAcceleration(
    absl::Span<const double> max_acceleration) {
  if (max_acceleration.size() != options_.num_dofs()) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("Dimension error, need ", options_.num_dofs(),
                     " joint values but max_velocity has ",
                     max_acceleration.size()));
  }
  max_joint_acceleration_ = FromSpan(max_acceleration);
  return absl::OkStatus();
}
const eigenmath::VectorXd& TimeableJointSplinePath::GetMaxJointVelocity()
    const {
  return max_joint_velocity_;
}
const eigenmath::VectorXd& TimeableJointSplinePath::GetMaxJointAcceleration()
    const {
  return max_joint_acceleration_;
}

absl::Status TimeableJointSplinePath::SetInitialVelocity(
    absl::Span<const double> velocity) {
  if (velocity.size() != NumDofs()) {
    return absl::InvalidArgumentError(
        "Velocity dimension doesn't match number of dofs.");
  }
  initial_velocity_ =
      Eigen::Map<const eigenmath::VectorXd>(velocity.data(), velocity.size());
  return absl::OkStatus();
}

const eigenmath::VectorXd& TimeableJointSplinePath::GetInitialVelocity() const {
  return initial_velocity_;
}

size_t TimeableJointSplinePath::NumConstraints() const {
  return num_constraints_;
}
size_t TimeableJointSplinePath::NumDofs() const { return options_.num_dofs(); }
size_t TimeableJointSplinePath::NumPathSamples() const {
  return options_.num_path_samples();
}

bool TimeableJointSplinePath::CloseToEnd(double parameter) const {
  return knots_.empty() || parameter >= knots_.back() - kSmall;
}

TimeablePath::State TimeableJointSplinePath::GetState() const {
  return path_state_;
}

const eigenmath::VectorXd& TimeableJointSplinePath::GetPathStart() const {
  CHECK(!path_position_.empty());
  return path_position_.front();
}
const eigenmath::VectorXd& TimeableJointSplinePath::GetPathEnd() const {
  CHECK(!path_position_.empty());
  return path_position_.back();
}

double TimeableJointSplinePath::GetParameterStart() const {
  return parameter_start_;
}

double TimeableJointSplinePath::GetParameterEnd() const {
  return parameter_end_;
}

const std::vector<eigenmath::VectorXd>& TimeableJointSplinePath::GetWaypoints()
    const {
  return waypoints_;
}

const eigenmath::VectorXd& TimeableJointSplinePath::GetPathPositionAt(
    size_t n) const {
  CHECK_WITH_MSG(n < path_position_.size(), "Have %zu samples, but n=%zu.",
                 path_position_.size(), n);
  return path_position_[n];
}
const eigenmath::VectorXd& TimeableJointSplinePath::GetFirstPathDerivativeAt(
    size_t n) const {
  CHECK_WITH_MSG(n < first_path_derivative_.size(),
                 "Have %zu samples, but n=%zu.", first_path_derivative_.size(),
                 n);
  return first_path_derivative_[n];
}
const eigenmath::VectorXd& TimeableJointSplinePath::GetSecondPathDerivativeAt(
    size_t n) const {
  CHECK_WITH_MSG(n < second_path_derivative_.size(),
                 "Have %zu samples, but n=%zu.", second_path_derivative_.size(),
                 n);
  return second_path_derivative_[n];
}

absl::Status TimeableJointSplinePath::CheckWaypoints(
    absl::Span<const eigenmath::VectorXd> waypoints) const {
  for (const auto& wp : waypoints) {
    if (wp.size() != options_.num_dofs()) {
      return absl::Status(
          absl::StatusCode::kInvalidArgument,
          absl::StrCat("Dimension error, need ", options_.num_dofs(),
                       " joint values but waypoint has ", wp.size()));
    }
  }
  return absl::OkStatus();
}

absl::Status TimeableJointSplinePath::SetWaypoints(
    absl::Span<const eigenmath::VectorXd> waypoints) {
  RETURN_IF_ERROR(CheckWaypoints(waypoints));
  waypoints_ = {waypoints.begin(), waypoints.end()};
  path_state_ = State::kNewPath;

  return FitSplineToWaypoints();
}

absl::Status TimeableJointSplinePath::SwitchToWaypointPath(
    const double keep_path_until,
    absl::Span<const eigenmath::VectorXd> waypoints) {
  path_state_ = State::kModifiedPath;
  // Truncate the current spline at `keep_path_until`.
  RETURN_IF_ERROR(spline_.TruncateSplineAt(keep_path_until));

  eigenmath::VectorXd switch_position(NumDofs());
  RETURN_IF_ERROR(spline_.EvalCurve(keep_path_until, switch_position));
  const auto projection_result_status =
      ProjectPointOnPath(waypoints, switch_position);
  if (!projection_result_status.ok()) {
    return projection_result_status.status();
  }
  const auto projection_result = *projection_result_status;

  std::vector<eigenmath::VectorXd> new_waypoints;
  new_waypoints.reserve(waypoints.size() + 1);
  // Use the projected point as first waypoint if it is not too close to the
  // switching position.
  constexpr double kEpsilon = 1e-3;
  if ((switch_position - projection_result.projected_point)
          .lpNorm<Eigen::Infinity>() > kEpsilon) {
    new_waypoints.push_back(projection_result.projected_point);
  }
  // If the line parameter for the closest line segment is negative, the
  // projected point is on the line before the first  point.
  // In that case use all waypoints starting at the first line segment
  // waypoint. Otherwise, omit the first waypoint.
  const int first_waypoint = projection_result.line_parameter >= 0
                                 ? projection_result.waypoint_index + 1
                                 : projection_result.waypoint_index;
  new_waypoints.insert(new_waypoints.end(), waypoints.begin() + first_waypoint,
                       waypoints.end());

  PolyLineToBspline3Waypoints(new_waypoints, options_.rounding(),
                              &control_points_);
  RETURN_IF_ERROR(spline_.ExtendWithControlPoints(control_points_));
  knots_ = {spline_.GetKnotVector().begin(), spline_.GetKnotVector().end()};

  return absl::OkStatus();
}

absl::Status TimeableJointSplinePath::FitSplineToWaypoints() {
  PolyLineToBspline3Waypoints(waypoints_, options_.rounding(),
                              &control_points_);

  knots_.resize(BSplineXd::NumKnots(control_points_.size(), kSplineOrder));
  // Allocate extra capacity for knots, in case it is needed later for
  // replanning, plus a minimum number.
  // Consider putting this in Options if we have different use cases.
  // Alternatively, consider enabling optional dynamic resizing in BSpline.
  constexpr int kNotCapacitySafety = 2;
  constexpr int kMinKnotCapacity = 100;
  const int knot_capacity = std::max(
      static_cast<int>(kNotCapacitySafety * knots_.size()), kMinKnotCapacity);
  RETURN_IF_ERROR(
      spline_.Init(kSplineOrder, knot_capacity, options_.num_dofs()));

  RETURN_IF_ERROR(BSplineXd::MakeUniformKnotVector(control_points_.size(),
                                                   &knots_, kSplineOrder));

  // Scale knot values to account for the total length of the control polygon.
  // This is to make the problem less dependent on the density of waypoints.
  double control_polygon_length = 0.0;
  for (int i = 0; i < control_points_.size() - 1; ++i) {
    control_polygon_length +=
        (control_points_[i + 1] - control_points_[i]).norm();
  }
  constexpr double kMinimumFinalKnotValue = 0.1;
  constexpr double kPathParameterPerPolygonLength = 1.0;
  const double weighted_length =
      std::max(control_polygon_length * kPathParameterPerPolygonLength,
               kMinimumFinalKnotValue);

  // Scale knot values to account for the total length of the control polygon.
  Eigen::Map<Eigen::VectorXd>(knots_.data(), knots_.size()) *= weighted_length;

  RETURN_IF_ERROR(spline_.SetKnotVector(knots_));

  RETURN_IF_ERROR(spline_.SetControlPoints(control_points_));

  return absl::OkStatus();
}

absl::Status TimeableJointSplinePath::SamplePath(const double path_start) {
  parameter_start_ = path_start;
  parameter_end_ = options_.num_path_samples() * options_.delta_parameter();
  for (int idx = 0; idx < options_.num_path_samples(); idx++) {
    // TODO Consider adjusting sampling strategy so samples have
    // a more uniform spatial distribution.
    const double parameter = path_start + idx * options_.delta_parameter();
    if (parameter < knots_.back() + options_.delta_parameter()) {
      // TODO: Change interface to avoid copy below.
      RETURN_IF_ERROR(spline_.EvalCurveAndDerivatives(
          std::clamp(parameter, knots_.front(), knots_.back()),
          absl::MakeSpan(spline_sampled_)));
      path_position_[idx] = spline_sampled_[0];
      first_path_derivative_[idx] = spline_sampled_[1];
      second_path_derivative_[idx] = spline_sampled_[2];
    } else {
      path_position_[idx] = control_points_.back();
      first_path_derivative_[idx].setZero();
      second_path_derivative_[idx].setZero();
    }
  }
  path_state_ = State::kPathWasSampled;

  return absl::OkStatus();
}

absl::Status TimeableJointSplinePath::ConstraintSetup() {
  // Setup constraints: one constraint for velocity & acceleration per dof.
  for (int idx = 0; idx < options_.num_path_samples(); idx++) {
    for (size_t dof = 0; dof < options_.num_dofs(); dof++) {
      // Acceleration constraints. The timing algorithm could handle
      // asymmetrical limits, but we only apply symmetrical limits here.
      constraints_[idx].a_coefficient(dof) = first_path_derivative_[idx][dof];
      constraints_[idx].b_coefficient(dof) = second_path_derivative_[idx][dof];
      constraints_[idx].upper(dof) =
          max_joint_acceleration_[dof] * options_.constraint_safety();
      constraints_[idx].lower(dof) =
          -max_joint_acceleration_[dof] * options_.constraint_safety();
      // Velocity constraints (must be symmetrical).
      constraints_[idx].a_coefficient(options_.num_dofs() + dof) = 0.0;
      constraints_[idx].b_coefficient(options_.num_dofs() + dof) =
          std::pow(first_path_derivative_[idx][dof], 2);
      constraints_[idx].upper(options_.num_dofs() + dof) =
          std::pow(max_joint_velocity_[dof] * options_.constraint_safety(), 2);
      constraints_[idx].lower(options_.num_dofs() + dof) = 0.0;
    }
  }

  return absl::OkStatus();
}

const std::vector<TimeOptimalPathProfile::Constraint>&
TimeableJointSplinePath::GetConstraints() const {
  return constraints_;
}
}  // namespace trajectory_planning

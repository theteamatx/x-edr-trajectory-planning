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

#include "trajectory_planning/timeable_path_cartesian_spline.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <vector>

#include "absl/log/log.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "eigenmath/pose3_utils.h"
#include "eigenmath/types.h"
#include "trajectory_planning/path_tools.h"
#include "trajectory_planning/splines/spline_utils.h"

namespace trajectory_planning {

namespace {
constexpr double kSmall = 1e-4;

void ComputePathDerivatives(
    const std::vector<eigenmath::VectorXd>& path, double delta_parameter,
    std::vector<eigenmath::VectorXd>* first_derivative_arg,
    std::vector<eigenmath::VectorXd>* second_derivative_arg) {
  const int path_size = path.size();
  CHECK(path_size > 1);
  CHECK_NE(first_derivative_arg, nullptr);
  CHECK_NE(second_derivative_arg, nullptr);
  std::vector<eigenmath::VectorXd>& first_derivative = *first_derivative_arg;
  std::vector<eigenmath::VectorXd>& second_derivative = *second_derivative_arg;
  CHECK_EQ(first_derivative.size(), path_size);
  CHECK_EQ(second_derivative.size(), path_size);

  const double inv_delta_parameter = 1.0 / delta_parameter;

  // Compute path derivatives by backward finite differences.
  for (int idx = 0; idx < path_size - 1; idx++) {
    first_derivative[idx] = inv_delta_parameter * (path[idx + 1] - path[idx]);
  }
  first_derivative.back().setZero();
  for (int idx = 1; idx < path_size - 1; idx++) {
    second_derivative[idx] = inv_delta_parameter * (first_derivative[idx + 1] -
                                                    first_derivative[idx]);
  }

  // Set second derivatives to zero at the edges to minimize numerical errors
  // due to transients in the numerical path ik dynamics.
  second_derivative[0].setZero();
  second_derivative[path_size - 1].setZero();
}
}  // namespace

double CartesianPathOptions::translation_rounding() const {
  return translational_rounding_;
}

CartesianPathOptions& CartesianPathOptions::set_translation_rounding(
    double rounding) {
  translational_rounding_ = rounding;
  return *this;
}

CartesianPathOptions::PathIKFunc CartesianPathOptions::GetPathIKFunc() const {
  return path_ik_func_;
}

CartesianPathOptions& CartesianPathOptions::set_path_ik_func(
    PathIKFunc path_ik) {
  path_ik_func_ = path_ik;
  return *this;
}

CartesianPathOptions::JacobianFunc CartesianPathOptions::GetJacobianFunc()
    const {
  return jacobian_func_;
}

CartesianPathOptions& CartesianPathOptions::set_jacobian_func(
    JacobianFunc jacobian) {
  jacobian_func_ = jacobian;
  return *this;
}

constexpr int TimeableCartesianSplinePath::kSplineOrder;

TimeableCartesianSplinePath::TimeableCartesianSplinePath(
    const CartesianPathOptions& options)
    : options_(options),
      // one velocity constraint per dof, one acceleration constraint per dof +
      // 2 for Cartesian velocity.
      num_constraints_(options_.num_dofs() * 2 + 2),
      path_state_(State::kNoPath),
      initial_velocity_(eigenmath::VectorXd::Zero(options_.num_dofs())) {
  // TODO: refactor to private ctor + factory method to avoid
  // asserts.
  CHECK(options_.num_path_samples() >= 3);
  CHECK(options.GetJacobianFunc() != nullptr)
      << "Need a Jacobian function set in Options.";
  CHECK(options.GetPathIKFunc() != nullptr)
      << "Need a path IK function set in Options.";

  path_ik_func_ = options.GetPathIKFunc();
  jacobian_func_ = options.GetJacobianFunc();

  path_position_.resize(options_.num_path_samples());
  first_path_derivative_.resize(options_.num_path_samples());
  second_path_derivative_.resize(options_.num_path_samples());
  sampled_pose_targets_.resize(options_.num_path_samples());
  sampled_joint_targets_.resize(options_.num_path_samples());
  for (int idx = 0; idx < options_.num_path_samples(); idx++) {
    path_position_[idx].resize(options_.num_dofs());
    path_position_[idx].setZero();
    sampled_joint_targets_[idx].resize(options_.num_dofs());
    sampled_joint_targets_[idx].setZero();
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
  max_joint_acceleration_.setZero();
  max_joint_velocity_.setZero();
  max_translational_velocity_ = 0.0;
  max_rotational_velocity_ = 0.0;
  Reset();
}

const std::vector<eigenmath::VectorXd>&
TimeableCartesianSplinePath::GetJointWaypoints() const {
  return joint_waypoints_;
}
const std::vector<eigenmath::Pose3d>&
TimeableCartesianSplinePath::GetPoseWaypoints() const {
  return pose_waypoints_;
}

absl::Status TimeableCartesianSplinePath::SetWaypoints(
    absl::Span<const eigenmath::Pose3d> pose_waypoints,
    absl::Span<const eigenmath::VectorXd> joint_waypoints) {
  path_state_ = State::kNewPath;
  if (joint_waypoints.size() != pose_waypoints.size()) {
    return absl::InvalidArgumentError(
        "'joint_waypoints' and 'pose_waypoints' have different sizes.");
  }

  for (const auto& joint_waypoint : joint_waypoints) {
    if (joint_waypoint.size() != options_.num_dofs()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Dimension error, need ", options_.num_dofs(),
          " joint values but waypoint has ", joint_waypoint.size()));
    }
  }
  joint_waypoints_ = {joint_waypoints.begin(), joint_waypoints.end()};
  pose_waypoints_ = {pose_waypoints.begin(), pose_waypoints.end()};
  path_ik_positions_.clear();
  return FitSplineToWaypoints();
}

absl::Status TimeableCartesianSplinePath::SwitchToWaypointPath(
    const double keep_path_until,
    absl::Span<const eigenmath::Pose3d> pose_waypoints,
    absl::Span<const eigenmath::VectorXd> joint_waypoints) {
  // Erase obsolete part of the ik solution.
  if (const int ik_index = PathIkIndex(keep_path_until);
      ik_index < path_ik_positions_.size()) {
    path_ik_positions_.resize(ik_index + 1);
  }

  if (joint_waypoints.size() != pose_waypoints.size()) {
    return absl::InvalidArgumentError(
        "'joint_waypoints' and 'pose_waypoints' have different sizes.");
  }
  path_state_ = State::kModifiedPath;
  // Truncate the current spline at `keep_path_until`.
  RETURN_IF_ERROR(joint_spline_.TruncateSplineAt(keep_path_until));
  RETURN_IF_ERROR(rotation_spline_.TruncateSplineAt(keep_path_until));
  RETURN_IF_ERROR(translation_spline_.TruncateSplineAt(keep_path_until));

  eigenmath::VectorXd joint_switch_position(NumDofs());
  eigenmath::Quaterniond switch_rotation = eigenmath::Quaterniond::Identity();
  eigenmath::Vector3d switch_translation = eigenmath::Vector3d::Zero();
  RETURN_IF_ERROR(
      joint_spline_.EvalCurve(keep_path_until, joint_switch_position));

  RETURN_IF_ERROR(rotation_spline_.EvalCurve(keep_path_until, switch_rotation));
  RETURN_IF_ERROR(
      translation_spline_.EvalCurve(keep_path_until, switch_translation));

  const auto joint_projection_status =
      ProjectPointOnPath(joint_waypoints, joint_switch_position);
  if (!joint_projection_status.ok()) {
    return joint_projection_status.status();
  }
  const auto joint_projection_result = *joint_projection_status;

  const auto kExtractTranslation =
      [](const eigenmath::Pose3d& pose) -> const eigenmath::Vector3d& {
    return pose.translation();
  };

  std::vector<eigenmath::Vector3d> translation_waypoints;
  translation_waypoints.reserve(pose_waypoints.size());
  std::transform(pose_waypoints.begin(), pose_waypoints.end(),
                 std::back_inserter(translation_waypoints),
                 kExtractTranslation);
  const auto translation_projection_status = ProjectPointOnPath(
      absl::Span<const eigenmath::Vector3d>(translation_waypoints),
      switch_translation);
  if (!translation_projection_status.ok()) {
    return translation_projection_status.status();
  }
  const auto translation_projection_result = *translation_projection_status;

  std::vector<eigenmath::VectorXd> new_joint_waypoints;
  std::vector<eigenmath::Pose3d> new_pose_waypoints;
  new_joint_waypoints.reserve(joint_waypoints.size() + 1);
  new_pose_waypoints.reserve(joint_waypoints.size() + 1);
  // Use the projected point as first waypoint if it is not too close to the
  // switching position.
  constexpr double kEpsilon = 1e-3;
  if ((switch_translation - translation_projection_result.projected_point)
          .lpNorm<Eigen::Infinity>() > kEpsilon) {
    new_joint_waypoints.push_back(joint_projection_result.projected_point);
    // Get rotation position at same path location as projected translation.
    eigenmath::Quaterniond rotation_at_projection =
        pose_waypoints[translation_projection_result.waypoint_index]
            .quaternion()
            .slerp(
                translation_projection_result.line_parameter,
                pose_waypoints[translation_projection_result.waypoint_index + 1]
                    .quaternion());
    new_pose_waypoints.push_back(eigenmath::Pose3d(
        rotation_at_projection, translation_projection_result.projected_point));
  }
  // If the line parameter for the closest line segment is negative, the
  // projected point is on the line before the first  point.
  // In that case use all waypoints starting at the first line segment
  // waypoint. Otherwise, omit the first waypoint.
  const int first_waypoint =
      translation_projection_result.line_parameter >= 0
          ? translation_projection_result.waypoint_index + 1
          : translation_projection_result.waypoint_index;
  new_joint_waypoints.insert(new_joint_waypoints.end(),
                             joint_waypoints.begin() + first_waypoint,
                             joint_waypoints.end());
  new_pose_waypoints.insert(new_pose_waypoints.end(),
                            pose_waypoints.begin() + first_waypoint,
                            pose_waypoints.end());

  PolyLineToBspline3Waypoints(new_joint_waypoints, options_.rounding(),
                              &joint_control_points_);
  PolyLineToBspline3Waypoints(new_pose_waypoints,
                              options_.translation_rounding(),
                              options_.rounding(), &pose_control_points_);
  std::vector<eigenmath::Vector3d> translation_points(
      pose_control_points_.size());
  std::vector<eigenmath::Quaterniond> rotation_points(
      pose_control_points_.size());
  for (int idx = 0; idx < pose_control_points_.size(); idx++) {
    translation_points[idx] = pose_control_points_[idx].translation();
    rotation_points[idx] = pose_control_points_[idx].quaternion();
  }

  RETURN_IF_ERROR(joint_spline_.ExtendWithControlPoints(joint_control_points_));
  RETURN_IF_ERROR(
      translation_spline_.ExtendWithControlPoints(translation_points));
  RETURN_IF_ERROR(rotation_spline_.ExtendWithControlPoints(rotation_points));

  knots_ = {joint_spline_.GetKnotVector().begin(),
            joint_spline_.GetKnotVector().end()};

  return absl::OkStatus();
}

absl::Status TimeableCartesianSplinePath::SetMaxJointVelocity(
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

absl::Status TimeableCartesianSplinePath::SetMaxJointAcceleration(
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

const eigenmath::VectorXd& TimeableCartesianSplinePath::GetMaxJointVelocity()
    const {
  return max_joint_velocity_;
}

const eigenmath::VectorXd&
TimeableCartesianSplinePath::GetMaxJointAcceleration() const {
  return max_joint_acceleration_;
}

absl::Status TimeableCartesianSplinePath::SetInitialVelocity(
    absl::Span<const double> velocity) {
  if (velocity.size() != NumDofs()) {
    return absl::InvalidArgumentError(
        "Velocity dimension doesn't match number of dofs.");
  }
  initial_velocity_ =
      Eigen::Map<const eigenmath::VectorXd>(velocity.data(), velocity.size());
  return absl::OkStatus();
}

const eigenmath::VectorXd& TimeableCartesianSplinePath::GetInitialVelocity()
    const {
  return initial_velocity_;
}

absl::Status TimeableCartesianSplinePath::SetMaxCartesianVelocity(
    double max_translational_velocity, double max_rotational_velocity) {
  if (max_translational_velocity <= 0 || max_rotational_velocity <= 0) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::Substitute(
            "Velocity limits must be positive, but got "
            "max_translational_velocity= $0, max_rotational_velocity= $1",
            max_translational_velocity, max_rotational_velocity));
  }
  max_rotational_velocity_ = max_rotational_velocity;
  max_translational_velocity_ = max_translational_velocity;
  return absl::OkStatus();
}

absl::Status TimeableCartesianSplinePath::SetTranslationRounding(
    double translation_rounding) {
  if (translation_rounding <= 0.0) {
    return absl::InvalidArgumentError(absl::Substitute(
        "translation_rounding needs to be greater than zero, got $0",
        translation_rounding));
  }
  options_.set_translation_rounding(translation_rounding);
  return absl::OkStatus();
}

absl::Status TimeableCartesianSplinePath::SetRotationRounding(
    double rotation_rounding) {
  if (rotation_rounding <= 0.0) {
    return absl::InvalidArgumentError(absl::Substitute(
        "rotation_rounding needs to be greater than zero, got $0",
        rotation_rounding));
  }
  options_.set_rounding(rotation_rounding);
  return absl::OkStatus();
}

bool TimeableCartesianSplinePath::CloseToEnd(double parameter) const {
  return knots_.empty() || parameter >= knots_.back() - kSmall;
}

TimeablePath::State TimeableCartesianSplinePath::GetState() const {
  return path_state_;
}

absl::Status TimeableCartesianSplinePath::FitSplineToWaypoints() {
  PolyLineToBspline3Waypoints(joint_waypoints_, options_.rounding(),
                              &joint_control_points_);
  PolyLineToBspline3Waypoints(pose_waypoints_, options_.translation_rounding(),
                              options_.rounding(), &pose_control_points_);

  const int num_control_points = joint_control_points_.size();

  knots_.resize(BSplineBase::NumKnots(num_control_points, kSplineOrder));

  // Allocate extra capacity for knots, in case it is needed later for
  // replanning, plus a minimum number.
  // Consider putting this in Options if we have different use cases.
  // Alternatively, consider enabling optional dynamic resizing in BSpline.
  constexpr int kNotCapacitySafety = 2;
  constexpr int kMinKnotCapacity = 100;
  const int knot_capacity = std::max(
      static_cast<int>(kNotCapacitySafety * knots_.size()), kMinKnotCapacity);

  RETURN_IF_ERROR(
      joint_spline_.Init(kSplineOrder, knot_capacity, options_.num_dofs()));
  RETURN_IF_ERROR(translation_spline_.Init(kSplineOrder, knot_capacity));
  RETURN_IF_ERROR(rotation_spline_.Init(kSplineOrder, knot_capacity));

  RETURN_IF_ERROR(BSplineBase::MakeUniformKnotVector(num_control_points,
                                                     &knots_, kSplineOrder));

  // Scale knot values to account for the total length of the control polygon.
  // This is to make the problem less dependent on the density of waypoints.
  eigenmath::PoseError control_polygon_length = {0.0, 0.0};
  for (int i = 0; i < pose_control_points_.size() - 1; ++i) {
    const eigenmath::PoseError delta =
        PoseErrorBetween(pose_control_points_[i], pose_control_points_[i + 1]);
    control_polygon_length.rotation += delta.rotation;
    control_polygon_length.translation += delta.translation;
  }
  // Equally weigh translation and rotation, which seems reasonable for
  // manipulator ranges of motion.
  constexpr double kMinimumFinalKnotValue = 0.1;
  constexpr double kPathParameterPerPolygonLength = 10.0;
  const double weighted_length = std::max(
      control_polygon_length.translation + control_polygon_length.translation,
      kMinimumFinalKnotValue);

  // Scale knot values to account for the total length of the control polygon.
  Eigen::Map<Eigen::VectorXd>(knots_.data(), knots_.size()) *=
      weighted_length * kPathParameterPerPolygonLength;

  RETURN_IF_ERROR(joint_spline_.SetKnotVector(knots_));
  RETURN_IF_ERROR(translation_spline_.SetKnotVector(knots_));
  RETURN_IF_ERROR(rotation_spline_.SetKnotVector(knots_));

  std::vector<eigenmath::Vector3d> translation_points(
      pose_control_points_.size());
  std::vector<eigenmath::Quaterniond> rotation_points(
      pose_control_points_.size());
  for (int idx = 0; idx < pose_control_points_.size(); idx++) {
    translation_points[idx] = pose_control_points_[idx].translation();
    rotation_points[idx] = pose_control_points_[idx].quaternion();
  }

  RETURN_IF_ERROR(joint_spline_.SetControlPoints(joint_control_points_));
  RETURN_IF_ERROR(translation_spline_.SetControlPoints(translation_points));
  RETURN_IF_ERROR(rotation_spline_.SetControlPoints(rotation_points));

  return absl::OkStatus();
}

absl::Status TimeableCartesianSplinePath::SamplePath(const double path_start) {
  const double path_horizon =
      path_start +
      options_.delta_parameter() * (options_.num_path_samples() - 1);
  // Compute the range of ik solution indices needed for path_position_.
  const int horizon_ik_upper_index = PathIkIndex(path_horizon);
  const int current_ik_upper_index = path_ik_positions_.size() - 1;
  // If necessary, sample the splines and compute more ik solutions.
  if (horizon_ik_upper_index >= current_ik_upper_index) {
    // Evaluate spline and compute IK for segment that hasn't been evaluated
    // yet, re-evaluating the first sample.
    const int num_new_samples =
        horizon_ik_upper_index - current_ik_upper_index + 1;
    sampled_pose_targets_.resize(num_new_samples);
    sampled_joint_targets_.resize(num_new_samples);
    for (int i = current_ik_upper_index; i <= horizon_ik_upper_index; ++i) {
      // NOTE: if we go past the valid spline range, we repeat the last sample,
      // possibly many times.
      // For the first segment, the parameter 0.0 is evaluated twice.
      // This is done to make the code for copying ik solution segments uniform.
      const double parameter = i < 0 ? 0.0 : PathIkParameter(i);
      const int new_sample_index = i - current_ik_upper_index;
      CHECK_LT(new_sample_index, num_new_samples);
      sampled_joint_targets_[new_sample_index].resize(options_.num_dofs());
      if (parameter < knots_.back() - options_.delta_parameter()) {
        RETURN_IF_ERROR(joint_spline_.EvalCurve(
            parameter, sampled_joint_targets_[new_sample_index]));
        eigenmath::Vector3d translation = eigenmath::Vector3d::Zero();
        eigenmath::Quaterniond rotation = eigenmath::Quaterniond::Identity();
        RETURN_IF_ERROR(translation_spline_.EvalCurve(parameter, translation));
        RETURN_IF_ERROR(rotation_spline_.EvalCurve(parameter, rotation));
        sampled_pose_targets_[new_sample_index].setQuaternion(rotation);
        sampled_pose_targets_[new_sample_index].translation() = translation;
      } else {
        sampled_pose_targets_[new_sample_index] = pose_control_points_.back();
        sampled_joint_targets_[new_sample_index] = joint_control_points_.back();
      }
    }

    // Initial value: one before the first ik solution.
    const eigenmath::VectorXd initial_value =
        current_ik_upper_index > 0 ? path_ik_positions_[current_ik_upper_index]
                                   : sampled_joint_targets_.front();
    new_ik_path_.clear();
    const absl::Status ik_status =
        path_ik_func_(initial_value, sampled_pose_targets_,
                      sampled_joint_targets_, &new_ik_path_);

    // Append the new IK solution unconditionally, so partial solutions are
    // available for debugging on failure.
    // Append starting from the second sample, as the first one matches the
    // initial conditions, which are taken from the previous segment.
    path_ik_positions_.insert(path_ik_positions_.end(),
                              new_ik_path_.begin() + 1, new_ik_path_.end());
    if (!ik_status.ok()) {
      LOG(ERROR) << absl::Substitute(
          "PathIK status: '$0', solution samples: $1, target samples: $2",
          ik_status.ToString(), new_ik_path_.size(),
          sampled_pose_targets_.size());
      return ik_status;
    }
  }

  // Copy the required segment from path_ik_positions_ to path_position_.
  const int path_start_index = PathIkIndex(path_start);
  CHECK_EQ(horizon_ik_upper_index - path_start_index,
           options_.num_path_samples() - 1)
      << absl::Substitute(
             "horizon_ik_upper_index $0, path_start_index: $1, delta: $2, "
             "#samples: $3",
             horizon_ik_upper_index, path_start_index,
             horizon_ik_upper_index - path_start_index,
             options_.num_path_samples());
  std::copy(path_ik_positions_.begin() + path_start_index,
            path_ik_positions_.begin() + horizon_ik_upper_index + 1,
            path_position_.begin());

  // Approximate derivatives by finite differences.
  ComputePathDerivatives(path_position_, options_.delta_parameter(),
                         &first_path_derivative_, &second_path_derivative_);

  path_state_ = State::kPathWasSampled;
  parameter_start_ = path_start;
  parameter_end_ = path_horizon;
  return absl::OkStatus();
}

absl::Status TimeableCartesianSplinePath::ConstraintSetup() {
  eigenmath::Matrix6Xd jacobian(6, options_.num_dofs());
  jacobian.setZero();
  // Derivatives of velocity (linear;angular) w.r.t. the path parameter.
  eigenmath::Vector6d velocity_derivative = eigenmath::Vector6d::Zero();

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
    RETURN_IF_ERROR(jacobian_func_(path_position_[idx], &jacobian));
    velocity_derivative = jacobian * first_path_derivative_[idx];
    constraints_[idx].a_coefficient(2 * options_.num_dofs()) = 0.0;
    constraints_[idx].b_coefficient(2 * options_.num_dofs()) =
        velocity_derivative.head(3).squaredNorm();
    constraints_[idx].upper(2 * options_.num_dofs()) =
        std::pow(max_translational_velocity_, 2);
    constraints_[idx].lower(2 * options_.num_dofs()) =
        -constraints_[idx].upper(2 * options_.num_dofs());

    constraints_[idx].a_coefficient(2 * options_.num_dofs() + 1) = 0.0;
    constraints_[idx].b_coefficient(2 * options_.num_dofs() + 1) =
        velocity_derivative.tail(3).squaredNorm();
    constraints_[idx].upper(2 * options_.num_dofs() + 1) =
        std::pow(max_rotational_velocity_, 2);
    constraints_[idx].lower(2 * options_.num_dofs() + 1) =
        -constraints_[idx].upper(2 * options_.num_dofs() + 1);
  }
  return absl::OkStatus();
}

const std::vector<TimeOptimalPathProfile::Constraint>&
TimeableCartesianSplinePath::GetConstraints() const {
  return constraints_;
}

size_t TimeableCartesianSplinePath::NumConstraints() const {
  return num_constraints_;
}

size_t TimeableCartesianSplinePath::NumDofs() const {
  return options_.num_dofs();
}

size_t TimeableCartesianSplinePath::NumPathSamples() const {
  return options_.num_path_samples();
}

void TimeableCartesianSplinePath::Reset() {
  joint_waypoints_.clear();
  pose_waypoints_.clear();
  joint_control_points_.clear();
  pose_control_points_.clear();
  path_ik_positions_.clear();
  path_state_ = State::kNoPath;
  parameter_start_ = -1.0;
  parameter_end_ = -1.0;
}

const eigenmath::VectorXd& TimeableCartesianSplinePath::GetPathStart() const {
  CHECK(!joint_waypoints_.empty());
  return joint_waypoints_.front();
}

const eigenmath::VectorXd& TimeableCartesianSplinePath::GetPathEnd() const {
  CHECK(!joint_waypoints_.empty());
  return path_position_.back();
}

double TimeableCartesianSplinePath::GetParameterStart() const {
  return parameter_start_;
}
double TimeableCartesianSplinePath::GetParameterEnd() const {
  return parameter_end_;
}

const eigenmath::VectorXd& TimeableCartesianSplinePath::GetPathPositionAt(
    size_t n) const {
  ABSL_CHECK_LT(n, path_position_.size());
  return path_position_[n];
}

const eigenmath::VectorXd&
TimeableCartesianSplinePath::GetFirstPathDerivativeAt(size_t n) const {
  ABSL_CHECK_LT(n, first_path_derivative_.size());
  return first_path_derivative_[n];
}

const eigenmath::VectorXd&
TimeableCartesianSplinePath::GetSecondPathDerivativeAt(size_t n) const {
  ABSL_CHECK_LT(n, second_path_derivative_.size());
  return second_path_derivative_[n];
}

const std::vector<eigenmath::VectorXd>&
TimeableCartesianSplinePath::GetSplineIKPosition() const {
  return path_ik_positions_;
}
int TimeableCartesianSplinePath::GetNumPathSamples() const {
  return options_.num_path_samples();
}
double TimeableCartesianSplinePath::GetPathSamplingDistance() const {
  return options_.delta_parameter();
}

int TimeableCartesianSplinePath::PathIkIndex(
    const double path_parameter) const {
  return std::round(path_parameter / options_.delta_parameter());
}

double TimeableCartesianSplinePath::PathIkParameter(const int index) const {
  return index * options_.delta_parameter();
}

}  // namespace trajectory_planning

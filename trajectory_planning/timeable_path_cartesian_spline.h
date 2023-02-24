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

#ifndef TRAJECTORY_PLANNING_TIMEABLE_PATH_CARTESIAN_SPLINE_H_
#define TRAJECTORY_PLANNING_TIMEABLE_PATH_CARTESIAN_SPLINE_H_

#include <functional>
#include <vector>

#include "absl/status/status.h"
#include "eigenmath/pose3.h"
#include "eigenmath/types.h"
#include "trajectory_planning/splines/bspline.h"
#include "trajectory_planning/splines/bsplineq.h"
#include "trajectory_planning/time_optimal_path_timing.h"
#include "trajectory_planning/timeable_path.h"

namespace trajectory_planning {

class CartesianPathOptions : public PathOptions<CartesianPathOptions> {
 public:
  // Path IK function, taking (initial_value, pose_targets, joint_targets,
  // ik_result).
  typedef std::function<absl::Status(const eigenmath::VectorXd&,
                                     const std::vector<eigenmath::Pose3d>&,
                                     const std::vector<eigenmath::VectorXd>&,
                                     std::vector<eigenmath::VectorXd>*)>
      PathIKFunc;
  typedef std::function<absl::Status(const eigenmath::VectorXd&,
                                     eigenmath::Matrix6Xd*)>
      JacobianFunc;
  double translation_rounding() const;
  CartesianPathOptions& set_translation_rounding(double rounding);
  PathIKFunc GetPathIKFunc() const;
  CartesianPathOptions& set_path_ik_func(PathIKFunc path_ik);
  JacobianFunc GetJacobianFunc() const;
  CartesianPathOptions& set_jacobian_func(JacobianFunc jacobian);

 private:
  // Rounding of corners for translational component when converting waypoints
  // to splines.
  double translational_rounding_ = 0.05;
  // Path inverse kinematics function.
  PathIKFunc path_ik_func_;
  // Jacobian function.
  JacobianFunc jacobian_func_;
};

// A Cartesian path with constraint computation to generate velocity &
// accleration-limited trajectories along it.
class TimeableCartesianSplinePath : public TimeablePath {
 public:
  explicit TimeableCartesianSplinePath(const CartesianPathOptions& options);

  // Add a waypoint to the queue, consisting of the Cartesian target pose and
  // the ideal joint space solution for it (to guide redundancy resolution,
  // etc.).
  absl::Status SetWaypoints(
      absl::Span<const eigenmath::Pose3d> pose_waypoints,
      absl::Span<const eigenmath::VectorXd> joint_waypoints);

  // Truncates the current path at `keep_path_until` and adds spline
  // control points corresponding to `*_waypoints` starting at waypoints
  // following the projection of the position at `path_parameter` onto the
  // polyline defined by `waypoints`. Returns a non-ok status on error.
  absl::Status SwitchToWaypointPath(
      double keep_path_until,
      absl::Span<const eigenmath::Pose3d> pose_waypoints,
      absl::Span<const eigenmath::VectorXd> joint_waypoints);

  // Set Cartesian velocity limits for path following.
  // max_translational_velocity is imposed on the Euclidean norm of the
  // translational component of motion, max_rotational_velocity on the Euclidean
  // norm of the angular velocity.
  absl::Status SetMaxCartesianVelocity(double max_translational_velocity,
                                       double max_rotational_velocity);

  // Gets the currently configured corner rounding parameters.
  double GetTranslationRounding() const {
    return options_.translation_rounding();
  }
  double GetRotationRounding() const { return options_.rounding(); }

  // Sets the Cartesian b-spline rounding parameters at the path waypoints.
  // These will control how closely the b-spline tries to reach the given
  // waypoints. Needs to be called before 'SetWaypoints', which is where the
  // b-spline is generated.
  // Returns an error if the given value is invalid (zero or negative).
  absl::Status SetTranslationRounding(double translation_rounding);
  absl::Status SetRotationRounding(double rotation_rounding);

  // TimeablePath interface implementation.
  // See base class for documentation of these methods.
  absl::Status SetMaxJointVelocity(
      absl::Span<const double> max_velocity) override;
  absl::Status SetMaxJointAcceleration(
      absl::Span<const double> max_acceleration) override;
  const eigenmath::VectorXd& GetMaxJointVelocity() const override;
  const eigenmath::VectorXd& GetMaxJointAcceleration() const override;
  absl::Status SetInitialVelocity(absl::Span<const double> velocity) override;
  const eigenmath::VectorXd& GetInitialVelocity() const override;

  bool CloseToEnd(double parameter) const override;
  State GetState() const override;
  absl::Status SamplePath(double path_start) override;
  absl::Status ConstraintSetup() override;
  const std::vector<TimeOptimalPathProfile::Constraint>& GetConstraints()
      const override;
  size_t NumConstraints() const override;
  size_t NumDofs() const override;
  size_t NumPathSamples() const override;
  void Reset() override;
  const eigenmath::VectorXd& GetPathStart() const override;
  const eigenmath::VectorXd& GetPathEnd() const override;
  const std::vector<eigenmath::VectorXd>& GetJointWaypoints() const;
  const std::vector<eigenmath::Pose3d>& GetPoseWaypoints() const;
  double GetParameterStart() const override;
  double GetParameterEnd() const override;
  const eigenmath::VectorXd& GetPathPositionAt(size_t n) const override;
  const eigenmath::VectorXd& GetFirstPathDerivativeAt(size_t n) const override;
  const eigenmath::VectorXd& GetSecondPathDerivativeAt(size_t n) const override;
  const std::vector<eigenmath::VectorXd>& GetSplineIKPosition() const;
  int GetNumPathSamples() const override;
  double GetPathSamplingDistance() const override;

  // Returns the index into the IK positions vector (see GetSplineIKPosition)
  // for the given path_parameter. Does not check the size of the vector.
  int PathIkIndex(double path_parameter) const;

  // Returns the path parameter corresponding to a given `index` in the IK
  // positions vector (see GetSplineIKPosition). Does not check the size of the
  // vector.
  double PathIkParameter(int index) const;

 private:
  absl::Status FitSplineToWaypoints();

  static constexpr int kSplineOrder = 2;

  CartesianPathOptions options_;
  const size_t num_constraints_;
  State path_state_ = State::kNoPath;
  BSpline3d translation_spline_;
  BSplineQ rotation_spline_;
  BSplineXd joint_spline_;
  std::vector<eigenmath::VectorXd> joint_waypoints_;
  std::vector<eigenmath::Pose3d> pose_waypoints_;
  std::vector<eigenmath::VectorXd> joint_control_points_;
  std::vector<eigenmath::Pose3d> pose_control_points_;

  std::vector<eigenmath::Pose3d> sampled_pose_targets_;
  std::vector<eigenmath::VectorXd> sampled_joint_targets_;

  std::vector<eigenmath::VectorXd> path_position_;
  std::vector<eigenmath::VectorXd> first_path_derivative_;
  std::vector<eigenmath::VectorXd> second_path_derivative_;
  std::vector<double> knots_;

  // Inverse kinematics solution for the path.
  // Samples are equidistant in the path coordinate. The vector grows with calls
  // to SamplePath and is cleared in SetWaypoints and Reset.
  // The `PathIkIndex` and `PathIkParameter` functions convert between indices
  // into this vector and the corresponsing path/spline parameter.
  std::vector<eigenmath::VectorXd> path_ik_positions_;
  std::vector<TimeOptimalPathProfile::Constraint> constraints_;
  eigenmath::VectorXd max_joint_velocity_;
  eigenmath::VectorXd max_joint_acceleration_;
  double max_translational_velocity_;
  double max_rotational_velocity_;
  CartesianPathOptions::PathIKFunc path_ik_func_;
  CartesianPathOptions::JacobianFunc jacobian_func_;
  // Buffer for new IK solution samples.
  std::vector<eigenmath::VectorXd> new_ik_path_;
  double parameter_start_;
  double parameter_end_;

  eigenmath::VectorXd initial_velocity_;
};

}  // namespace planning
#endif  // TRAJECTORY_PLANNING_TIMEABLE_PATH_CARTESIAN_SPLINE_H_

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

#ifndef TRAJECTORY_PLANNING_TIMEABLE_PATH_JOINT_SPLINE_H_
#define TRAJECTORY_PLANNING_TIMEABLE_PATH_JOINT_SPLINE_H_

#include <vector>

#include "trajectory_planning/splines/bspline.h"
#include "eigenmath/types.h"
#include "trajectory_planning/time_optimal_path_timing.h"
#include "trajectory_planning/timeable_path.h"
#include "absl/status/status.h"

namespace trajectory_planning {

class JointPathOptions : public PathOptions<JointPathOptions> {
 public:
  JointPathOptions() = default;
};

// A joint-space path with constraint computation to generate velocity &
// acceleration-limited trajectories along it.
class TimeableJointSplinePath : public TimeablePath {
 public:
  explicit TimeableJointSplinePath(const JointPathOptions& options);

  // Unconditionally sets waypoints, discarding any previous path.
  absl::Status SetWaypoints(absl::Span<const eigenmath::VectorXd> waypoints);

  // Truncates the current path at `keep_path_until` and adds spline
  // control points corresponding to `waypoints` starting at waypoints following
  // the projection of the position at `path_parameter` onto the polyline
  // defined by `waypoints`. Returns a non-ok status on error.
  absl::Status SwitchToWaypointPath(
      double keep_path_until, absl::Span<const eigenmath::VectorXd> waypoints);

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
  absl::Status SamplePath(const double path_start) override;
  absl::Status ConstraintSetup() override;
  const std::vector<TimeOptimalPathProfile::Constraint>& GetConstraints()
      const override;
  size_t NumConstraints() const override;
  size_t NumDofs() const override;
  size_t NumPathSamples() const override;
  void Reset() override;
  const eigenmath::VectorXd& GetPathStart() const override;
  const eigenmath::VectorXd& GetPathEnd() const override;
  const std::vector<eigenmath::VectorXd>& GetWaypoints() const;
  double GetParameterStart() const override;
  double GetParameterEnd() const override;
  const eigenmath::VectorXd& GetPathPositionAt(size_t n) const override;
  const eigenmath::VectorXd& GetFirstPathDerivativeAt(size_t n) const override;
  const eigenmath::VectorXd& GetSecondPathDerivativeAt(size_t n) const override;
  int GetNumPathSamples() const override { return options_.num_path_samples(); }
  double GetPathSamplingDistance() const override {
    return options_.delta_parameter();
  }

 private:
  absl::Status FitSplineToWaypoints();

  absl::Status CheckWaypoints(
      absl::Span<const eigenmath::VectorXd> waypoints) const;

  // The order of the used b-spline.
  static constexpr int kSplineOrder = 2;
  const JointPathOptions options_;
  const size_t num_constraints_;

  State path_state_ = State::kNoPath;
  BSplineXd spline_;
  std::vector<eigenmath::VectorXd> waypoints_;
  std::vector<eigenmath::VectorXd> control_points_;
  std::vector<eigenmath::VectorXd> spline_sampled_;
  std::vector<eigenmath::VectorXd> path_position_;
  std::vector<eigenmath::VectorXd> first_path_derivative_;
  std::vector<eigenmath::VectorXd> second_path_derivative_;
  std::vector<double> knots_;
  std::vector<TimeOptimalPathProfile::Constraint> constraints_;
  eigenmath::VectorXd max_joint_velocity_;
  eigenmath::VectorXd max_joint_acceleration_;
  double parameter_start_;
  double parameter_end_;
  eigenmath::VectorXd initial_velocity_;
};

}  // namespace trajectory_planning

#endif  // TRAJECTORY_PLANNING_TIMEABLE_PATH_JOINT_SPLINE_H_

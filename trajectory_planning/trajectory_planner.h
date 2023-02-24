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

#ifndef TRAJECTORY_PLANNING_TRAJECTORY_PLANNER_H_
#define TRAJECTORY_PLANNING_TRAJECTORY_PLANNER_H_

#include <memory>
#include <vector>

#include "eigenmath/types.h"
#include "trajectory_planning/timeable_path.h"
#include "absl/status/status.h"
#include "absl/time/time.h"

namespace trajectory_planning {

// Common options for path timing trajectory optimization.
template <typename DerivedOptions>
class TrajectoryPlannerOptions {
 public:
  TrajectoryPlannerOptions() {
    static_assert(std::is_base_of<TrajectoryPlannerOptions<DerivedOptions>,
                                  TrajectoryPlannerOptions>::value,
                  "Template parameter for TrajectoryPlannerOptions must be a "
                  "derived class.");
  }
  size_t GetNumDofs() const { return num_dofs_; }
  absl::Duration GetTimeStep() const { return time_step_; }

  DerivedOptions& SetNumDofs(size_t num_dofs) {
    num_dofs_ = num_dofs;
    return static_cast<DerivedOptions&>(*this);
  }
  DerivedOptions& SetTimeStep(absl::Duration time_step) {
    time_step_ = time_step;
    return static_cast<DerivedOptions&>(*this);
  }

 protected:
  // The time step at which solutions will be sampled.
  absl::Duration time_step_;
  // Number of degrees of freedom for the problem.
  size_t num_dofs_ = 0;
};

// A trajectory planning class that takes waypoints and generates a dense
// vector of time sampled points representing a trajectory.
class TrajectoryPlanner {
 public:
  TrajectoryPlanner() = default;
  virtual ~TrajectoryPlanner() = default;

  // Resets the generator and clears waypoint queue.
  // Trajectories planned after calling this will start at the first
  // waypoint added after the call.
  // Will call Reset() on the path associated with *this.
  void Reset() {
    ResetBase();
    ResetDerived();
  }

  // Plan a trajectory segment from time start until about time+time_horizon.
  // If the end of the trajectory is reached earlier, the trajectory will be
  // shorter. Otherwise, the trajectory may also be longer due to
  // implementation details. The trajectory always ends with zero velocity. To
  // compute a solution over a moving horizon, repeatedly call Plan(start,
  // time_horizon), with start shifted accordingly.
  virtual absl::Status Plan(absl::Time start, absl::Duration time_horizon) = 0;
  // Returns the number of time samples.
  size_t GetNumTimeSamples() const { return time_.size(); }
  // Returns starting time of most recent plan.
  absl::Time GetStartTime() const { return start_time_; }
  // Returns end time of most recent plan.
  absl::Time GetEndTime() const { return end_time_; }
  // Returns the time pointes for the trajectory.
  const std::vector<double>& GetTime() const { return time_; }
  // Returns the time series of positions for the trajectory.
  const std::vector<eigenmath::VectorXd>& GetPositions() const {
    return positions_;
  }
  // Returns the time series of velocities for the trajectory.
  const std::vector<eigenmath::VectorXd>& GetVelocities() const {
    return velocities_;
  }
  // Returns the time series of accelerations for the trajectory.
  const std::vector<eigenmath::VectorXd>& GetAccelerations() const {
    return accelerations_;
  }

  // Returns the path parameter values.
  const std::vector<double>& GetPathParameters() const {
    return path_parameter_;
  }
  const std::vector<double>& GetPathParameterDerivatives() const {
    return path_parameter_derivative_;
  }
  const std::vector<double>& GetSecondPathParameterDerivatives() const {
    return second_path_parameter_derivative_;
  }

  // Returns true if the most recent plan reached the end of the path, and the
  // path has not changed since.
  virtual bool IsTrajectoryAtEnd() const {
    const bool path_unchanged =
        path_ == nullptr ||
        (path_->GetState() != TimeablePath::State::kModifiedPath &&
         path_->GetState() != TimeablePath::State::kNewPath);
    return path_unchanged && target_reached_;
  }

  virtual absl::Status SetPath(std::shared_ptr<TimeablePath> path) = 0;

 protected:
  // Reset functionality specific to the derived class.
  // Called from Reset() after ResetBase().
  virtual void ResetDerived() = 0;
  // Reset functionality specific to this base class.
  // Called from Reset() before ResetDerived().
  void ResetBase() {
    if (path_ != nullptr) {
      path_->Reset();
    }
    start_time_ = absl::FromUnixSeconds(0.0);
    end_time_ = absl::FromUnixSeconds(0.0);
    time_.clear();
    path_parameter_.clear();
    path_parameter_derivative_.clear();
    second_path_parameter_derivative_.clear();
    positions_.clear();
    velocities_.clear();
    accelerations_.clear();
    target_reached_ = false;
  }

  std::shared_ptr<TimeablePath> path_;
  absl::Time start_time_;
  absl::Time end_time_;
  // The trajectory, equidistantly sampled in time.
  std::vector<double> time_;
  std::vector<double> path_parameter_;
  std::vector<double> path_parameter_derivative_;
  std::vector<double> second_path_parameter_derivative_;
  std::vector<eigenmath::VectorXd> positions_;
  std::vector<eigenmath::VectorXd> velocities_;
  std::vector<eigenmath::VectorXd> accelerations_;
  bool target_reached_ = false;
};

}  // namespace trajectory_planning

#endif  // TRAJECTORY_PLANNING_TRAJECTORY_PLANNER_H_

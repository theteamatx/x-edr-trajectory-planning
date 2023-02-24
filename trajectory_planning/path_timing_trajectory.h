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

#ifndef TRAJECTORY_PLANNING_PATH_TIMING_TRAJECTORY_H_
#define TRAJECTORY_PLANNING_PATH_TIMING_TRAJECTORY_H_

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "eigenmath/types.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "trajectory_planning/time_optimal_path_timing.h"
#include "trajectory_planning/timeable_path.h"
#include "trajectory_planning/trajectory_planner.h"

namespace trajectory_planning {

// Options for path timing trajectory optimization.
class PathTimingTrajectoryOptions
    : public TrajectoryPlannerOptions<PathTimingTrajectoryOptions> {
 public:
  // Indicates the method for generating time samples from the output of the
  // path timing algorithm, which is uniformly sampled in the path parameter
  // space.
  enum class TimeSamplingMethod {
    // Resample the path timer output uniformly in time.
    kUniformlyInTime,
    // Keep the path sampler output as is, but discard samples closer than
    // TrajectoryPlannerOptions::GetTimeStep().
    kSkipSamplesCloserThanTimeStep
  };
  // Returns the number of path samples used in the TimeOptimalPathTiming
  // algorithm.
  size_t GetNumPathSamples() const;
  // Sets the number of path samples used in the TimeOptimalPathTiming
  // algorithm.
  PathTimingTrajectoryOptions& SetNumPathSamples(size_t num_samples);
  // Returns the maximum acceptable error in the velocity initial condition.
  double GetMaxInitialVelocityError() const;
  // Sets the maximum acceptable error in the velocity initial condition.
  PathTimingTrajectoryOptions& SetMaxInitialVelocityError(double error);
  // Set the maximum number of internal planning iterations.
  PathTimingTrajectoryOptions& SetMaxPlanningLoops(int max_planning_iterations);
  // Returns the maximum number of internal planning iterations.
  int GetMaxPlanningIterations() const;
  // Sets the sampling policy.
  PathTimingTrajectoryOptions& SetTimeSamplingMethod(
      TimeSamplingMethod method) {
    time_sampling_method_ = method;
    return *this;
  }
  // Returns the sampling policy.
  TimeSamplingMethod GetTimeSamplingMethod() const {
    return time_sampling_method_;
  }

 private:
  size_t num_path_samples_ = 1000;
  double max_initial_velocity_error_ = 1e-2;
  int max_planning_iterations_ = 200;
  TimeSamplingMethod time_sampling_method_ =
      TimeSamplingMethod::kUniformlyInTime;
};

// A trajectory planning class that takes waypoints, fits a b-spline to follow
// the piecewise-linear path with deviations at the corners and computes a
// trajectory tracking the b-spline by computing a timing profile along it.
//
// For examples of intended use, see path_timing_trajectory_planner_test.
//
// Limitations:Adding waypoints after starting a plan will make the trajectory
// fully stop at the last waypoint before the newly inserted one.
//
// TODO: Consider making a run time interface for all
// trajectory planners and deriving from it in order simplify switching
// between various implementations in the module.
class PathTimingTrajectory : public TrajectoryPlanner {
 public:
  explicit PathTimingTrajectory(const PathTimingTrajectoryOptions& options);
  // Plan a trajectory segment from time start until about time+time_horizon.
  // If the end of the trajectory is reached earlier, the trajectory will be
  // shorter. Otherwise, the trajectory may also be longer due to
  // implementation details. The trajectory always ends with zero velocity. To
  // compute a solution over a moving horizon, repeatedly call Plan(start,
  // time_horizon), with start shifted accordingly.
  absl::Status Plan(absl::Time start, absl::Duration time_horizon) override;
  // Returns the number of time samples.
  size_t NumTimeSamples() const;
  // Returns the time at which the final deceleration phase begins.
  absl::Time GetFinalDecelStart() const;
  // Returns the next viable start time for a given target starting time.
  // absl::Time will advance by replan from the last start time if possible.
  // If crossing a spline boundary, the duration will be increased, if the
  // path end has been reached, it will be smaller.
  absl::Time GetNextPlanStartTime(absl::Time target_time);

  // Sets the path to compute the trajectory from.
  absl::Status SetPath(std::shared_ptr<TimeablePath> path) override;

  // Sets the debug verbosity level in the time optimal profile generator.
  void SetProfileDebugVerbosity(int level);

  // Returns the path parameter at which zero velocity can be reached,
  // or a status if an error occurred.
  absl::StatusOr<double> GetPathStopParameter(absl::Time time) const;

  // Returns the Options for *this.
  const PathTimingTrajectoryOptions& GetOptions() const { return options_; }

  // Test-only functions.
  void TestOnlySetTimeSamples(absl::Span<const double> time);

 protected:
  void ResetDerived() override;

 private:
  struct InterpolationResult {
    // The lower index of samples used for interplation.
    int lower_index = 0;
    eigenmath::VectorXd position;
    eigenmath::VectorXd velocity;
    eigenmath::VectorXd acceleration;
    double path_parameter = 0.0;
    double path_parameter_derivative = 0.0;
    double second_path_parameter_derivative = 0.0;
  };

  void UpdatePathTrackingStatus();
  absl::Status HandleTimeArguments(absl::Time start);
  absl::Status ComputeTimingProfile(absl::Time start,
                                    absl::Duration target_duration);
  void ClampToTimeStepMultiple(absl::Time* time);
  absl::StatusOr<int> GetTimeOffsetAfter(absl::Time time) const;
  void ResampleTrajectory(double start_sec);
  void ResampleEquidistantlyInTime(double start_sec);
  void ResampleSkippingSamplesCloserThanTimeStep(double start_sec);
  int TimeAtPathSamplesLowerIndex(int starting_index, double time) const;
  void EraseTrajectoryBefore(absl::Time time);

  // Returns an `InterpolationResult` computed by linear interpolation at
  // `time_sec`. Search for samples to use in the interpolation starts at
  // `lower_index`.
  InterpolationResult InterpolateAtTime(double time_sec, int lower_index) const;

  // Erases (resampled) trajectory samples in the range of [begin,
  // begin+offsset).
  void EraseSamplesUntil(int offset);

  double GetMinTimeDeltaToKeep() const;

  const PathTimingTrajectoryOptions options_;
  const double time_step_sec_;
  absl::Time final_decel_start_;
  double path_horizon_;
  double path_time_start_;
  double path_start_;
  double path_start_velocity_;
  double path_start_acceleration_;
  TimeOptimalPathProfile profile_;
  bool initial_plan_ = false;
  bool planned_to_end_ = false;

  // Path values sampled equidistantly along the path coordinate.
  std::vector<double> time_at_path_samples_;
  std::vector<double> path_parameter_at_path_samples_;
  std::vector<double> path_velocity_at_path_samples_;
  std::vector<double> path_acceleration_at_path_samples_;
  // Trajectory values sampled equidistantly along the path coordinate.
  std::vector<eigenmath::VectorXd> position_at_path_samples_;
  std::vector<eigenmath::VectorXd> velocity_at_path_samples_;
  std::vector<eigenmath::VectorXd> acceleration_at_path_samples_;
};

}  // namespace trajectory_planning

#endif  // TRAJECTORY_PLANNING_PATH_TIMING_TRAJECTORY_H_

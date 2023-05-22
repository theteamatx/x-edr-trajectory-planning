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

#include "trajectory_planning/path_timing_trajectory.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <limits>
#include <memory>
#include <tuple>

#include "absl/algorithm/container.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "eigenmath/interpolation.h"
#include "eigenmath/scalar_utils.h"
#include "eigenmath/types.h"
#include "trajectory_planning/timeable_path.h"
#include "trajectory_planning/time.h"

namespace trajectory_planning {

namespace {
int GetSampleCountUntil(absl::Span<const double> samples, const double time) {
  auto it = absl::c_lower_bound(samples, time);
  return std::distance(samples.begin(), it);
}

// A view of a trajectory for use in ComputeFastestStop.
struct FastestStopTrajectoryView {
  // The number samples for this trajectory.
  int sample_count = 0;
  // An invocable that returns the time for sample i.
  absl::AnyInvocable<double(int) const> time;
  // An invocable that returns the joint velocities for sample i.
  absl::AnyInvocable<absl::Span<const double>(int) const> velocities;
  // An invocable that returns the joint accelerations for sample i.
  absl::AnyInvocable<absl::Span<const double>(int) const> accelerations;
};

// A view of a stopping trajectory output for use in ComputeFastestStop.
struct FastestStopOutputView {
  // The last sample of the trajectory used while stopping is written to this
  // value.
  int& trajectory_stop_index;
  // The total stopping duration is written to this value.
  double& total_duration;
  // `append_time(ti)` is called for all stopping trajectory time samples.
  absl::AnyInvocable<void(double)> append_time;
  // `append_rate_squared(ti)` is called for all samples of the squared time
  // scaling rates.
  absl::AnyInvocable<void(double)> append_rate_squared;
  // `append_rate_squared(ti)` is called for all samples of the squared time
  // scaling rate differential.
  absl::AnyInvocable<void(double)> append_diff_rate_squared;
};

void ComputeFastestStop(const FastestStopTrajectoryView& trajectory,
                        absl::Span<const double> stop_acceleration_span,
                        FastestStopOutputView& output) {
  const int dof_count = stop_acceleration_span.size();

  // Compute stopping trajectory by timescaling along the path traced by
  // states_.
  // In the following "pos" = position, "vel" = velocity, "acc" =
  // acceleration and "traj" = trajectory, and "t" = time are used.
  // Time-scaling yields the trajectory:
  // (1) stop_pos(t) = traj_pos(s(t)), so
  // (2) stop_vel(t) = traj_vel(s(t))*(ds/dt)
  // (3) stop_acc(t) = traj_vel(s)*(d^2s/dt^2) + traj_acc(s)*(ds/dt)^2
  //                 = traj_vel(s)*1/2*d((ds/dt)^2)/ds + traj_acc(s)*(ds/dt)^2
  //
  // The following code integrates d((ds/dt)^2)/ds over s to get the time
  // scaling trajectory by computing the maximum admissible scaling rate
  // from (3). This results in new times for the original trajectory
  // samples, plus re-scaled velocities and accelerations.
  // TODO: Reconsider time-scaling approach.
  double& total_duration = output.total_duration;
  int& path_index = output.trajectory_stop_index;
  total_duration = 0.0;

  // Initialize time scaling with scaling rate = 1.0.
  double rate_squared = 1.0;
  double diff_rate_squared = 0.0;
  const double first_time_point = trajectory.time(0);

  const Eigen::Map<const eigenmath::VectorNd> stop_acceleration =
      Eigen::Map<const eigenmath::VectorNd>(stop_acceleration_span.data(),
                                            stop_acceleration_span.size());

  for (path_index = 0;
       (path_index < trajectory.sample_count - 1) && (rate_squared > 0.0);
       path_index++) {
    // Find maximum scaling rate differential by checking all acceleration
    // limits.
    absl::Span<const double> velocity_span = trajectory.velocities(path_index);
    absl::Span<const double> acceleration_span =
        trajectory.accelerations(path_index);
    double diff_rate_squared_min = 0.0;
    const eigenmath::VectorNd acc_bias =
        Eigen::Map<const eigenmath::VectorNd>(acceleration_span.data(),
                                              acceleration_span.size()) *
        rate_squared;
    const eigenmath::VectorNd path_velocity =
        Eigen::Map<const eigenmath::VectorNd>(velocity_span.data(),
                                              velocity_span.size());
    for (int dof = 0; dof < dof_count; dof++) {
      constexpr double kEpsilon = 1e-6;
      if (std::abs(velocity_span[dof]) < kEpsilon) {
        continue;
      }
      double diff_rate_squared_i =
          2.0 * (-acc_bias[dof] - stop_acceleration[dof]) / velocity_span[dof];
      eigenmath::VectorNd acc(acc_bias +
                              0.5 * path_velocity * diff_rate_squared_i);

      constexpr double kTinyAcceleration = 1e-10;
      bool acc_valid =
          (stop_acceleration - acc).minCoeff() >= -kTinyAcceleration &&
          (-stop_acceleration - acc).maxCoeff() <= kTinyAcceleration;
      if (acc_valid && diff_rate_squared_i < diff_rate_squared_min) {
        diff_rate_squared_min = diff_rate_squared_i;
      }
      diff_rate_squared_i =
          2.0 * (-acc_bias[dof] + stop_acceleration[dof]) / path_velocity[dof];
      acc = acc_bias + 0.5 * path_velocity * diff_rate_squared_i;
      acc_valid = (stop_acceleration - acc).minCoeff() >= -1e-10 &&
                  (-stop_acceleration - acc).maxCoeff() <= 1e-10;
      if (acc_valid && diff_rate_squared_i < diff_rate_squared_min) {
        diff_rate_squared_min = diff_rate_squared_i;
      }
    }
    diff_rate_squared = std::min(diff_rate_squared_min, 0.0);
    output.append_time(first_time_point + total_duration);
    output.append_rate_squared(rate_squared);
    output.append_diff_rate_squared(diff_rate_squared);

    // Compute next rate^2 by forward Euler integration.
    const double unscaled_dt =
        trajectory.time(path_index + 1) - trajectory.time(path_index);
    double next_rate_squared =
        std::max(0.0, rate_squared + unscaled_dt * diff_rate_squared);
    const double dt = 2.0 * unscaled_dt /
                      (std::sqrt(rate_squared) + std::sqrt(next_rate_squared));
    total_duration += dt;
    rate_squared = next_rate_squared;
  }
  output.append_time(first_time_point + total_duration);
  output.append_rate_squared(rate_squared);
  output.append_diff_rate_squared(diff_rate_squared);
  // In case we reached the end of the path, ensure path_index valid.
  if (path_index >= trajectory.sample_count) {
    path_index = trajectory.sample_count - 1;
  }
}

}  // namespace

size_t PathTimingTrajectoryOptions::GetNumPathSamples() const {
  return num_path_samples_;
}

PathTimingTrajectoryOptions& PathTimingTrajectoryOptions::SetNumPathSamples(
    size_t num_samples) {
  num_path_samples_ = num_samples;
  return (*this);
}

double PathTimingTrajectoryOptions::GetMaxInitialVelocityError() const {
  return max_initial_velocity_error_;
}

PathTimingTrajectoryOptions&
PathTimingTrajectoryOptions::SetMaxInitialVelocityError(double error) {
  max_initial_velocity_error_ = error;
  return *this;
}

PathTimingTrajectoryOptions& PathTimingTrajectoryOptions::SetMaxPlanningLoops(
    int max_planning_iterations) {
  max_planning_iterations_ = max_planning_iterations;
  return *this;
}

int PathTimingTrajectoryOptions::GetMaxPlanningIterations() const {
  return max_planning_iterations_;
}

PathTimingTrajectory::PathTimingTrajectory(
    const PathTimingTrajectoryOptions& options)
    : options_(options),
      time_step_sec_(TimeToSec(absl::time_internal::FromUnixDuration(
          options.GetTimeStep()))) {
  Reset();
}

void PathTimingTrajectory::ResetDerived() {
  initial_plan_ = false;
  path_time_start_ = 0.0;
  path_start_ = 0.0;
  path_start_velocity_ = 0.0;
  path_start_acceleration_ = 0.0;
  path_horizon_ = 0.0;
  planned_to_end_ = true;
  final_decel_start_ = TimeFromSec(0.0);

  time_at_path_samples_.clear();
  path_parameter_at_path_samples_.clear();
  path_velocity_at_path_samples_.clear();
}

void PathTimingTrajectory::ClampToTimeStepMultiple(absl::Time* time) {
  const int64_t loop_multiple =
      std::round(TimeToSec(*time) / time_step_sec_);
  *time = TimeFromSec(loop_multiple * time_step_sec_);
}

absl::StatusOr<double> PathTimingTrajectory::GetPathStopParameter(
    absl::Time time) const {
  // Get sampled trajectory index for time.
  const double double_time = TimeToSec(time);
  if (!initial_plan_) {
    // No plan yet, return 0.0 (beginning of path).
    return 0.0;
  }
  const auto time_it = absl::c_lower_bound(time_, double_time);
  if (time_it == time_.end()) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("Time ", double_time, " not in timed path range"));
  }
  if (time_it == std::prev(time_.end())) {
    return path_parameter_.back();
  }
  const int time_index_offset = time_it - time_.begin();
  if (time_index_offset < 0) {
    return absl::InternalError(
        absl::StrCat("Got time_offset (", time_index_offset, ") < 0"));
  }
  // Compute path index at which we could stop.
  FastestStopTrajectoryView trajectory_view{
      .sample_count = static_cast<int>(time_.size() - time_index_offset),
      .time = [this, time_index_offset](
                  int i) { return time_[i + time_index_offset]; },
      .velocities =
          [this, time_index_offset](int i) {
            return absl::MakeConstSpan(velocities_[i + time_index_offset]);
          },
      .accelerations =
          [this, time_index_offset](int i) {
            return absl::MakeConstSpan(accelerations_[i + time_index_offset]);
          },
  };
  double total_duration = 0;
  int path_stop_index = 0;
  FastestStopOutputView stop_output{.trajectory_stop_index = path_stop_index,
                                    .total_duration = total_duration,
                                    .append_time = [](double) {},
                                    .append_rate_squared = [](double) {},
                                    .append_diff_rate_squared = [](double) {}};

  ComputeFastestStop(trajectory_view, path_->GetMaxJointAcceleration(),
                     stop_output);
  path_stop_index += time_index_offset;
  // Path parameter for path index.
  CHECK_GE(path_stop_index, 0);
  CHECK(path_stop_index < time_.size()) << "path_stop_index=" << path_stop_index
                                        << " time_.size()=" << time_.size();
  return path_parameter_[path_stop_index];
}

absl::StatusOr<int> PathTimingTrajectory::GetTimeOffsetAfter(
    absl::Time time) const {
  const double time_sec = TimeToSec(time);
  if (time_.empty()) {
    return absl::FailedPreconditionError("No samples yet.");
  }
  if (time_sec < time_.front()) {
    return absl::OutOfRangeError("time < start_time.");
  }
  const auto time_sample_it = absl::c_upper_bound(time_, time_sec);
  if (time_sample_it == time_.end()) {
    return absl::InternalError(
        absl::StrCat("time (", time_sec, ") >= end_time_ (",
                     TimeToSec(end_time_), ")."));
  }
  return time_sample_it - time_.begin();
}

absl::Status PathTimingTrajectory::ComputeTimingProfile(
    absl::Time start, absl::Duration target_duration) {
  const double start_sec = TimeToSec(start);
  if (path_ == nullptr) {
    return absl::Status(absl::StatusCode::kFailedPrecondition, "No path set");
  }
  if (target_duration <= absl::Seconds(0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("absl::Duration must be positive (got ",
                     target_duration / absl::Seconds(1), ")."));
  }
  // If this solution is connecting to a previous segment, ensure we use a
  // discrete sample for the initial conditions.
  const TimeablePath::State old_path_state = path_->GetState();
  int path_samples_offset = 0;
  if (old_path_state == TimeablePath::State::kNewPath) {
    path_start_ = 0.0;
    path_start_velocity_ = 0.0;
    path_start_acceleration_ = 0.0;
    path_time_start_ = start_sec;
  } else {
    // Set the path start to the sample before `start`.
    const int num_nonuniform_samples = time_at_path_samples_.size();
    CHECK(!time_at_path_samples_.empty());
    CHECK_EQ(path_parameter_at_path_samples_.size(), num_nonuniform_samples);
    CHECK_EQ(path_velocity_at_path_samples_.size(), num_nonuniform_samples);
    path_samples_offset =
        std::clamp<int>(absl::c_lower_bound(time_at_path_samples_, start_sec) -
                            time_at_path_samples_.begin() - 1,
                        0, num_nonuniform_samples - 1);
    path_start_ = path_parameter_at_path_samples_[path_samples_offset];
    path_start_velocity_ = path_velocity_at_path_samples_[path_samples_offset];
    path_time_start_ = time_at_path_samples_[path_samples_offset];
  }
  path_horizon_ = path_start_ + path_->GetPathSamplingDistance() *
                                    (path_->GetNumPathSamples() - 1);
  absl::Status sample_path_status = path_->SamplePath(path_start_);
  if (!sample_path_status.ok()) {
    return sample_path_status;
  }
  absl::Status constraint_setup_status = path_->ConstraintSetup();
  if (!constraint_setup_status.ok()) {
    return constraint_setup_status;
  }

  if (old_path_state == TimeablePath::State::kModifiedPath ||
      old_path_state == TimeablePath::State::kNewPath) {
    // Compute the path_start_velocity_ that minimizes the squared norm of
    // the initial velocity and path_->GetInitialVelocity():
    // (initial_velocity -
    // path_->GetFirstPathDerivativeAt(0)*path_start_velocity_)^2 -> min!
    // If the path derivative at the start zero / very small, use
    // path_start_velocity_ = 0 set above.
    const double path_derivative_squared_norm =
        path_->GetFirstPathDerivativeAt(0).squaredNorm();
    if (path_derivative_squared_norm >
        100 * std::numeric_limits<double>::epsilon()) {
      // If the initial velocity condition is antiparallel to the initial path
      // tangent, the minimal solution for path_start_velocity_ is negative.
      // Clamp it to positive values, as the path derivative must be strictly
      // positive for the timing algorithm to work.
      path_start_velocity_ = std::max(
          path_->GetInitialVelocity().dot(path_->GetFirstPathDerivativeAt(0)) /
              path_derivative_squared_norm,
          0.0);
    }
    // Compute the error in the initial velocity condition and return an error
    // if the configured threshold is violated.
    const double max_velocity_error =
        (path_->GetFirstPathDerivativeAt(0) * path_start_velocity_ -
         path_->GetInitialVelocity())
            .lpNorm<Eigen::Infinity>();
    if (max_velocity_error > options_.GetMaxInitialVelocityError()) {
      return absl::InvalidArgumentError(absl::Substitute(
          "Could not satisfy initial velocity (probably not parallel to "
          "initial tangent): error = $0, threshold= $1.",
          max_velocity_error, options_.GetMaxInitialVelocityError()));
    }
  }

  if (!profile_.InitSolver(options_.GetNumPathSamples(),
                           path_->NumConstraints())) {
    return absl::Status(absl::StatusCode::kInternal,
                        "Error initializing solver.");
  }

  // TimeOptimalPathTiming has a default of 100 solver loops.
  // If there are a large number of samples, increase this, as the maximum
  // required of loops will also go up for complicated path shapes
  // (this is related to the number of extremals used for assembling the
  // solution).
  const int max_solver_loops =
      std::max(size_t{100}, 10 * options_.GetNumPathSamples());
  profile_.SetMaxNumSolverLoops(max_solver_loops);

  if (!profile_.SetupProblem(path_->GetConstraints(), path_start_,
                             path_horizon_, path_start_velocity_,
                             path_start_acceleration_, path_time_start_)) {
    return absl::Status(absl::StatusCode::kInternal,
                        "Error setting up optimization problem");
  }

  // Solve the optimization problem.
  if (!profile_.OptimizePathParameter()) {
    // If this happens, it is likely due to a degenerate problem because of
    // too few samples or a bad choice of horizon length.
    return absl::Status(absl::StatusCode::kInternal,
                        "Error optimizing path parameter");
  }

  // Erase now obsolete data that was replanned.
  time_at_path_samples_.erase(
      time_at_path_samples_.begin() + path_samples_offset,
      time_at_path_samples_.end());
  path_parameter_at_path_samples_.erase(
      path_parameter_at_path_samples_.begin() + path_samples_offset,
      path_parameter_at_path_samples_.end());
  path_velocity_at_path_samples_.erase(
      path_velocity_at_path_samples_.begin() + path_samples_offset,
      path_velocity_at_path_samples_.end());
  path_acceleration_at_path_samples_.erase(
      path_acceleration_at_path_samples_.begin() + path_samples_offset,
      path_acceleration_at_path_samples_.end());
  position_at_path_samples_.erase(
      position_at_path_samples_.begin() + path_samples_offset,
      position_at_path_samples_.end());
  velocity_at_path_samples_.erase(
      velocity_at_path_samples_.begin() + path_samples_offset,
      velocity_at_path_samples_.end());
  acceleration_at_path_samples_.erase(
      acceleration_at_path_samples_.begin() + path_samples_offset,
      acceleration_at_path_samples_.end());

  // Append the new profile.
  time_at_path_samples_.insert(time_at_path_samples_.end(),
                               profile_.GetTimeSamples().begin(),
                               profile_.GetTimeSamples().end());
  path_parameter_at_path_samples_.insert(path_parameter_at_path_samples_.end(),
                                         profile_.GetPathParameter().begin(),
                                         profile_.GetPathParameter().end());
  path_velocity_at_path_samples_.insert(path_velocity_at_path_samples_.end(),
                                        profile_.GetPathVelocity().begin(),
                                        profile_.GetPathVelocity().end());
  path_acceleration_at_path_samples_.insert(
      path_acceleration_at_path_samples_.end(),
      profile_.GetPathAcceleration().begin(),
      profile_.GetPathAcceleration().end());

  position_at_path_samples_.reserve(time_at_path_samples_.size());
  velocity_at_path_samples_.reserve(time_at_path_samples_.size());
  acceleration_at_path_samples_.reserve(time_at_path_samples_.size());
  for (int i = 0; i < path_->GetNumPathSamples(); ++i) {
    const double path_velocity = profile_.GetPathVelocity()[i];
    const double path_acceleration = profile_.GetPathAcceleration()[i];
    position_at_path_samples_.push_back(path_->GetPathPositionAt(i));
    velocity_at_path_samples_.push_back(path_->GetFirstPathDerivativeAt(i) *
                                        path_velocity);

    // TODO: Remove clipping after fixing acceleration
    // violations.
    acceleration_at_path_samples_.push_back(
        (path_->GetFirstPathDerivativeAt(i) * path_acceleration +
         path_->GetSecondPathDerivativeAt(i) * eigenmath::Square(path_velocity))
            .cwiseMax(-path_->GetMaxJointAcceleration())
            .cwiseMin(path_->GetMaxJointAcceleration()));
  }

  return absl::OkStatus();
}

void PathTimingTrajectory::UpdatePathTrackingStatus() {
  target_reached_ = false;
  planned_to_end_ = false;
  if (!initial_plan_) {
    path_horizon_ = 0;
    path_start_ = 0;
    return;
  }

  planned_to_end_ = path_->CloseToEnd(path_horizon_);
  if (planned_to_end_) {
    if (path_->GetState() != TimeablePath::State::kNewPath &&
        path_->GetState() != TimeablePath::State::kModifiedPath) {
      target_reached_ = true;
    } else {
      path_horizon_ = 0.0;
      path_time_start_ = 0.0;
      path_start_ = 0.0;
      path_start_velocity_ = 0.0;
      path_start_acceleration_ = 0.0;
      planned_to_end_ = false;
    }
  }
}

absl::Status PathTimingTrajectory::HandleTimeArguments(absl::Time start) {
  if (initial_plan_ && start > end_time_ + absl::Seconds(time_step_sec_)) {
    return absl::Status(
        absl::StatusCode::kOutOfRange,
        absl::Substitute("start $0 > end $1 of previous plan, diff: $2",
                         TimeToSec(start),
                         TimeToSec(end_time_),
                         (end_time_ - start) / absl::Seconds(1)));
  }

  if (initial_plan_ == false) {
    start_time_ = start;
    end_time_ = start;
    path_start_ = 0.0;
  } else {
    if (start > end_time_) {
      return absl::Status(
          absl::StatusCode::kInvalidArgument,
          absl::Substitute("Start time must be < end time, but end_time_ ($0) "
                           "- start ($1) = $2 [ns]",
                           absl::ToUnixNanos(end_time_),
                           absl::ToUnixNanos(start),
                           (end_time_ - start) / absl::Nanoseconds(1)));
    }
    if (start < start_time_) {
      return absl::Status(
          absl::StatusCode::kInvalidArgument,
          absl::Substitute("Start time must be >= previous start time, but new "
                           "($0) - old ($1) = $2 [ns]",
                           absl::ToUnixNanos(start),
                           absl::ToUnixNanos(start_time_),
                           (start - start_time_) / absl::Nanoseconds(1)));
    }
    start_time_ = start;
  }
  return absl::OkStatus();
}

void PathTimingTrajectory::EraseTrajectoryBefore(absl::Time time) {
  const double time_sec = TimeToSec(time);
  if (time_.empty() || time_sec < time_.front()) {
    return;
  }
  switch (options_.GetTimeSamplingMethod()) {
    case PathTimingTrajectoryOptions::TimeSamplingMethod::
        kSkipSamplesCloserThanTimeStep: {
      // Number of samples with a timestamp < time_sec.
      const int samples_smaller_than_time =
          GetSampleCountUntil(time_, time_sec);
      const auto values_at_time =
          InterpolateAtTime(time_sec, std::max(samples_smaller_than_time, 0));
      if (time_[samples_smaller_than_time] <
          time_sec + GetMinTimeDeltaToKeep()) {
        EraseSamplesUntil(samples_smaller_than_time);
      } else {
        EraseSamplesUntil(samples_smaller_than_time - 1);
      }
      // Fix first sample.
      time_.front() = time_sec;
      positions_.front() = values_at_time.position;
      velocities_.front() = values_at_time.velocity;
      accelerations_.front() = values_at_time.acceleration;
      path_parameter_.front() = values_at_time.path_parameter;
      path_parameter_derivative_.front() =
          values_at_time.path_parameter_derivative;
      second_path_parameter_derivative_.front() =
          values_at_time.second_path_parameter_derivative;
    } break;
    case PathTimingTrajectoryOptions::TimeSamplingMethod::kUniformlyInTime: {
      const int offset =
          std::min<int>(std::round((time_sec - time_.front()) / time_step_sec_),
                        time_.size() - 1);
      EraseSamplesUntil(offset);
    } break;
  }
}

absl::Status PathTimingTrajectory::Plan(absl::Time start,
                                        absl::Duration time_horizon) {
  const double start_sec = TimeToSec(start);
  if (path_ == nullptr) {
    return absl::Status(absl::StatusCode::kFailedPrecondition, "No path set.");
  }
  if (auto status = HandleTimeArguments(start); !status.ok()) {
    return status;
  }
  UpdatePathTrackingStatus();

  const bool planned_enough =
      (path_->GetState() != TimeablePath::State::kNewPath) &&
      (path_->GetState() != TimeablePath::State::kModifiedPath) &&
      (final_decel_start_ >= start + time_horizon);

  // Erase solution before (but excluding) start time.
  if (!time_.empty() && planned_enough) {
    LOG(INFO) << "Already planned enough, erasing obsolete solutions before "
              << start_sec;
    EraseTrajectoryBefore(start);
    return absl::OkStatus();
  }

  if (initial_plan_) {
    // Erase everything after starting time.
    auto get_time_offset_result = GetTimeOffsetAfter(start);
    if (!get_time_offset_result.ok()) {
      return get_time_offset_result.status();
    }
    int offset = *get_time_offset_result;
    time_.erase(time_.begin() + offset, time_.end());
    path_parameter_.erase(path_parameter_.begin() + offset,
                          path_parameter_.end());
    path_parameter_derivative_.erase(
        path_parameter_derivative_.begin() + offset,
        path_parameter_derivative_.end());
    second_path_parameter_derivative_.erase(
        second_path_parameter_derivative_.begin() + offset,
        second_path_parameter_derivative_.end());
    positions_.erase(positions_.begin() + offset, positions_.end());
    velocities_.erase(velocities_.begin() + offset, velocities_.end());
    accelerations_.erase(accelerations_.begin() + offset, accelerations_.end());
  }

  // Repeatedly plan over a fixed path horizon until the time horizon is
  // reached.
  absl::Time loop_start_time = start;
  bool time_horizon_reached = false;
  for (int loop = 0; !planned_to_end_ && !time_horizon_reached; loop++) {
    // Compute timing profile.
    if (auto status = ComputeTimingProfile(
            loop_start_time, start + time_horizon - loop_start_time);
        !status.ok()) {
      return status;
    }
    CHECK(profile_.GetLastExtremalIndex() < profile_.GetTimeSamples().size());
    CHECK(profile_.GetLastExtremalIndex() >= 0);
    // Clamp decleration start index within the last half of the planning
    // window to ensure sufficient planning progress.
    const int decel_start =
        std::max(profile_.GetLastExtremalIndex(),
                 static_cast<int>(options_.GetNumPathSamples() / 2));
    // Get the beginning of the final deceleration phase from the starting
    // index of the last backward extremal.
    final_decel_start_ =
        TimeFromSec(profile_.GetTimeSamples()[decel_start]);
    // Generate uniform sampling of the trajectory in time.
    planned_to_end_ = path_->CloseToEnd(path_horizon_);

    time_horizon_reached =
        (profile_.GetTimeSamples()[options_.GetNumPathSamples() - 1] -
         TimeToSec(start)) > time_horizon / absl::Seconds(1);

    if (loop >= options_.GetMaxPlanningIterations()) {
      return absl::Status(absl::StatusCode::kDeadlineExceeded,
                          "Reached maximum number of planning loops");
    }
    loop_start_time = final_decel_start_;
  }

  ResampleTrajectory(start_sec);

  initial_plan_ = true;
  if (!time_.empty()) {
    end_time_ = TimeFromSec(time_.back());
    ClampToTimeStepMultiple(&end_time_);
    CHECK(profile_.GetLastExtremalIndex() < profile_.GetTimeSamples().size());
    const int decel_start = profile_.GetLastExtremalIndex();
    final_decel_start_ =
        TimeFromSec(profile_.GetTimeSamples()[decel_start]);
    ClampToTimeStepMultiple(&final_decel_start_);
  } else {
    end_time_ = start_time_;
    final_decel_start_ = end_time_;
  }

  target_reached_ = planned_to_end_;

  // printf(
  //     "Planned from %.4f to %.4f. Deceleration starts at %.4f; time; %.4f, "
  //     "%.4f.",
  //     TimeToSec(start_time_), TimeToSec(end_time_),
  //     TimeToSec(final_decel_start_), time_.front(), time_.back());
  return absl::OkStatus();
}

int PathTimingTrajectory::TimeAtPathSamplesLowerIndex(int starting_index,
                                                      double time) const {
  for (int index = starting_index; index < time_at_path_samples_.size() - 1;
       ++index) {
    if (time_at_path_samples_[index + 1] > time) {
      return index;
    }
  }
  return time_at_path_samples_.size() - 1;
}

void PathTimingTrajectory::ResampleTrajectory(const double start_sec) {
  switch (options_.GetTimeSamplingMethod()) {
    case PathTimingTrajectoryOptions::TimeSamplingMethod::kUniformlyInTime:
      ResampleEquidistantlyInTime(start_sec);
      break;
    case PathTimingTrajectoryOptions::TimeSamplingMethod::
        kSkipSamplesCloserThanTimeStep:
      ResampleSkippingSamplesCloserThanTimeStep(start_sec);
      break;
  }
}

PathTimingTrajectory::InterpolationResult
PathTimingTrajectory::InterpolateAtTime(const double time_sec,
                                        int lower_index) const {
  InterpolationResult result;
  result.lower_index = TimeAtPathSamplesLowerIndex(lower_index, time_sec);
  const int upper_index =
      std::min<int>(time_at_path_samples_.size() - 1, result.lower_index + 1);
  const double interpolate_at =
      std::abs(time_at_path_samples_[upper_index] -
               time_at_path_samples_[result.lower_index]) <
              std::numeric_limits<double>::epsilon()
          ? 0.5
          : (time_sec - time_at_path_samples_[result.lower_index]) /
                (time_at_path_samples_[upper_index] -
                 time_at_path_samples_[result.lower_index]);

  result.position = eigenmath::InterpolateLinear(
      interpolate_at, position_at_path_samples_[result.lower_index],
      position_at_path_samples_[upper_index]);

  result.velocity = eigenmath::InterpolateLinear(
      interpolate_at, velocity_at_path_samples_[result.lower_index],
      velocity_at_path_samples_[upper_index]);

  // TODO: Remove clipping after fixing acceleration
  // violations.
  result.acceleration =
      eigenmath::InterpolateLinear(
          interpolate_at, acceleration_at_path_samples_[result.lower_index],
          acceleration_at_path_samples_[upper_index])
          .cwiseMax(-path_->GetMaxJointAcceleration())
          .cwiseMin(path_->GetMaxJointAcceleration());

  result.path_parameter = eigenmath::InterpolateLinear(
      interpolate_at, path_parameter_at_path_samples_[result.lower_index],
      path_parameter_at_path_samples_[upper_index]);
  result.path_parameter_derivative = eigenmath::InterpolateLinear(
      interpolate_at, path_velocity_at_path_samples_[result.lower_index],
      path_velocity_at_path_samples_[upper_index]);
  result.second_path_parameter_derivative = eigenmath::InterpolateLinear(
      interpolate_at, path_acceleration_at_path_samples_[result.lower_index],
      path_acceleration_at_path_samples_[upper_index]);

  return result;
}

void PathTimingTrajectory::ResampleEquidistantlyInTime(const double start_sec) {
  const double duration = time_at_path_samples_.back() - start_sec;
  const int uniform_samples = std::ceil(duration / time_step_sec_) + 1;
  time_.resize(uniform_samples);
  positions_.resize(uniform_samples);
  velocities_.resize(uniform_samples);
  accelerations_.resize(uniform_samples);
  path_parameter_.resize(uniform_samples);
  path_parameter_derivative_.resize(uniform_samples);
  second_path_parameter_derivative_.resize(uniform_samples);

  int lower_index = 0;
  for (int i = 0; i < uniform_samples; ++i) {
    const double time = start_sec + time_step_sec_ * i;
    const auto values_at_time = InterpolateAtTime(time, lower_index);
    lower_index = values_at_time.lower_index;
    time_[i] = time;
    positions_[i] = values_at_time.position;
    velocities_[i] = values_at_time.velocity;
    accelerations_[i] = values_at_time.acceleration;
    path_parameter_[i] = values_at_time.path_parameter;
    path_parameter_derivative_[i] = values_at_time.path_parameter_derivative;
    second_path_parameter_derivative_[i] =
        values_at_time.second_path_parameter_derivative;
  }
  positions_.back() = position_at_path_samples_.back();
  velocities_.back().setZero();
  accelerations_.back().setZero();
}

void PathTimingTrajectory::ResampleSkippingSamplesCloserThanTimeStep(
    const double start_sec) {
  const int lower_index = TimeAtPathSamplesLowerIndex(0, start_sec);
  const int reserve_size = time_at_path_samples_.size() - lower_index + 1;

  time_.clear();
  positions_.clear();
  velocities_.clear();
  accelerations_.clear();
  path_parameter_.clear();
  path_parameter_derivative_.clear();
  second_path_parameter_derivative_.clear();
  time_.reserve(reserve_size);
  positions_.reserve(reserve_size);
  velocities_.reserve(reserve_size);
  accelerations_.reserve(reserve_size);
  path_parameter_.reserve(reserve_size);
  path_parameter_derivative_.reserve(reserve_size);
  second_path_parameter_derivative_.reserve(reserve_size);

  const double kMinTimeDeltaToKeepSample = GetMinTimeDeltaToKeep();
  // Compute the first sample by interpolation, to ensure it is exactly
  // aligned with the requested starting time.
  auto values_at_start = InterpolateAtTime(start_sec, lower_index);
  time_.push_back(start_sec);
  positions_.push_back(values_at_start.position);
  velocities_.push_back(values_at_start.velocity);
  accelerations_.push_back(values_at_start.acceleration);
  path_parameter_.push_back(values_at_start.path_parameter);
  path_parameter_derivative_.push_back(
      values_at_start.path_parameter_derivative);
  second_path_parameter_derivative_.push_back(
      values_at_start.second_path_parameter_derivative);
  for (int i = lower_index + 1; i < time_at_path_samples_.size(); ++i) {
    const double& new_time_sample = time_at_path_samples_[i];
    if (std::abs(new_time_sample - time_.back()) < kMinTimeDeltaToKeepSample) {
      continue;
    }
    time_.push_back(time_at_path_samples_[i]);
    positions_.push_back(position_at_path_samples_[i]);
    velocities_.push_back(velocity_at_path_samples_[i]);
    accelerations_.push_back(acceleration_at_path_samples_[i]);
    path_parameter_.push_back(path_parameter_at_path_samples_[i]);
    path_parameter_derivative_.push_back(path_velocity_at_path_samples_[i]);
    second_path_parameter_derivative_.push_back(
        path_acceleration_at_path_samples_[i]);
  }
  // Ensure we exactly hit the final path sample (derivatives are always zero).
  positions_.back() = position_at_path_samples_.back();
  velocities_.back().setZero();
  accelerations_.back().setZero();
}

absl::Time PathTimingTrajectory::GetFinalDecelStart() const {
  return final_decel_start_;
}

size_t PathTimingTrajectory::NumTimeSamples() const { return time_.size(); }

absl::Status PathTimingTrajectory::SetPath(std::shared_ptr<TimeablePath> path) {
  if (path == nullptr) {
    return absl::Status(absl::StatusCode::kInvalidArgument, "Path is nullptr.");
  }
  // TODO Consolidate NumDofs and NumPathSamples into one
  // structure and remove these checks.
  if (path->NumDofs() != options_.GetNumDofs()) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::Substitute("Path has $0 dofs but planner has $1", path->NumDofs(),
                         options_.GetNumDofs()));
  }
  if (path->NumPathSamples() != options_.GetNumPathSamples()) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::Substitute("Path has $0 samples but planner has $1",
                         path->NumPathSamples(), options_.GetNumPathSamples()));
  }
  path_ = path;
  return absl::OkStatus();
}

void PathTimingTrajectory::SetProfileDebugVerbosity(int level) {
  profile_.SetDebugVerbosity(level);
}

absl::Time PathTimingTrajectory::GetNextPlanStartTime(absl::Time target_time) {
  return std::min(end_time_, std::max(target_time, start_time_));
}

void PathTimingTrajectory::EraseSamplesUntil(const int offset) {
  time_.erase(time_.begin(), time_.begin() + offset);
  path_parameter_.erase(path_parameter_.begin(),
                        path_parameter_.begin() + offset);
  path_parameter_derivative_.erase(path_parameter_derivative_.begin(),
                                   path_parameter_derivative_.begin() + offset);
  second_path_parameter_derivative_.erase(
      second_path_parameter_derivative_.begin(),
      second_path_parameter_derivative_.begin() + offset);
  positions_.erase(positions_.begin(), positions_.begin() + offset);
  velocities_.erase(velocities_.begin(), velocities_.begin() + offset);
  accelerations_.erase(accelerations_.begin(), accelerations_.begin() + offset);
}

void PathTimingTrajectory::TestOnlySetTimeSamples(
    absl::Span<const double> time) {
  time_ = {time.begin(), time.end()};
}

double PathTimingTrajectory::GetMinTimeDeltaToKeep() const {
  // Accept samples that are a little closer than the nominal time step, but not
  // too much. This is to avoid skipping samples that are only a little bit
  // denser than the configured time step, while avoiding very dense samples
  // that lead to very long trajectory buffers with samples that the controller
  // will skip.
  return 0.95 * time_step_sec_;
}

}  // namespace trajectory_planning

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

#ifndef TRAJECTORY_PLANNING_TRAJECTORY_PLANNING_TRAJECTORY_BUFFER_H_
#define TRAJECTORY_PLANNING_TRAJECTORY_PLANNING_TRAJECTORY_BUFFER_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/time/time.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "eigenmath/types.h"

namespace trajectory_planning {
// Options for the TrajectoryBuffer.
struct TrajectoryBufferOptions {
  // Timesteps that are closer than this value are treated as equal.
  double timestep_tolerance = 1e-6;
};

// A buffer for uniformly sampled trajectories. Supports appending segments to
// the end and discarding segments at the beginning.
// This class is general to any number of DOFs. In the context of the
// manipulation local planner, the trajectory contains the full base + arm
// state, i.e. each state contains wheel joint angles and arm joint angles.
class TrajectoryBuffer {
 public:
  static absl::StatusOr<std::unique_ptr<TrajectoryBuffer>> Create(
      TrajectoryBufferOptions options = TrajectoryBufferOptions{});
  // Clear the buffer.
  void Clear();
  // Reserve space for `size` samples.
  void Reserve(size_t size);
  // Returns the starting time of the trajectory.
  absl::Time GetStartTime() const;
  // Returns the end time of the trajectory (GetStartTime() +
  // GetNumSamples()*GetTimeStep()).
  // Note: This corresponds to a sample index that is one beyond the array
  // bounds.
  absl::Time GetEndTime() const;
  // Returns the number of trajectory samples.
  size_t GetNumSamples() const { return positions_.size(); }
  // Returns the number of times AppendSegmentAfter was called, -1.
  int GetSequenceNumber() const { return sequence_number_; }
  // Returns the vector of position samples.
  absl::Span<const eigenmath::VectorXd> GetPositions() const {
    return positions_;
  }
  // Returns the vector of velocity samples.
  absl::Span<const eigenmath::VectorXd> GetVelocities() const {
    return velocities_;
  }
  // Returns the vector of acceleration samples.
  absl::Span<const eigenmath::VectorXd> GetAccelerations() const {
    return accelerations_;
  }
  // Returns the timestamps in seconds.
  absl::Span<const double> GetTimes() const { return times_; }

  // Adds a trajectory segment to the buffer.
  // Any previously existing overlapping data is discarded.
  // 'times.front()' must be in [GetStartTime(), GetEndTime()].
  // In the special case that `time` == GetEndTime(), no data is discarded and
  // the new trajectory samples are appended at the end of the existing buffer.
  // The function maintains samples up until, but excluding the samples
  // at `times.front()`, and replaces samples starting at `times.front()`.
  // If the first timestep is within TrajectoryBufferOptions::timestep_tolerance
  // of an existing timestep, the newer sample will replace the older one.
  // Returns a non-ok status on error.
  absl::Status InsertSegment(
      absl::Span<const double> times,
      absl::Span<const eigenmath::VectorXd> positions,
      absl::Span<const eigenmath::VectorXd> velocities,
      absl::Span<const eigenmath::VectorXd> accelerations);

  // Append one sample.
  // Returns a non-ok status if an error occurred.
  absl::Status AppendSample(double time, const eigenmath::VectorXd& positions,
                            const eigenmath::VectorXd& velocities,
                            const eigenmath::VectorXd& accelerations);

  // Removes the trajectory segment before the given 'time.'
  // If the 'time' is before the start time, nothing is done.
  void DiscardSegmentBefore(absl::Time time);

  // Removes the trajectory segment before the given 'time_sec.'
  // If the 'time_sec' is before the start time, nothing is done.
  // If 'time_sec' doesn't align with one of the existing samples, a new first
  // sample is created by interpolation.
  void DiscardSegmentBefore(double time_sec);

  // Get all the positions in the buffer up to, but excluding, the given 'time.'
  // If the given time is not in the buffer range, an empty span is
  // returned.
  absl::Span<const eigenmath::VectorXd> GetPositionsUpToTime(
      absl::Time time) const;

  // Removes the trajectory segment after the given `index` and
  // changes the segment before it so the trajectory ends in a stop, using
  // `max_acceleration`. The stopping trajectory is resampled at `time_step`.
  // If the operation fails, the trajectory is unchanged an a non-ok status
  // is returned.
  absl::Status StopAtIndex(int index,
                           const eigenmath::VectorXd& max_acceleration,
                           double time_step);

  // Removes the trajectory segment following `time` and modifies the remaining
  // segment to end in a stop, using `max_acceleration`. The starting element of
  // the removed trajectory is the first sample whose value is larger than (or
  // exactly equal to) `time`. The stopping trajectory is resampled at
  // `time_step`. If the operation fails, the trajectory is unchanged and a
  // non-ok status is returned.
  absl::Status StopBeforeTime(absl::Time time,
                              const eigenmath::VectorXd& max_acceleration,
                              double time_step);
  absl::Status StopBeforeTime(double time_sec,
                              const eigenmath::VectorXd& max_acceleration,
                              double time_step);
  // Returns the position sample at `time`, or a status on error.
  // The value is computed by linear interpolation.
  absl::StatusOr<eigenmath::VectorXd> GetPositionAtTime(absl::Time time) const;

  // Same as above, but takes the time in seconds as input.
  // The value is computed by linear interpolation.
  absl::StatusOr<eigenmath::VectorXd> GetPositionAtTime(double time_sec) const;

  // Returns the velocity sample at `time`, or a status on error.
  // The value is computed by linear interpolation.
  absl::StatusOr<eigenmath::VectorXd> GetVelocityAtTime(absl::Time time) const;

  // Same as above, but takes the time in seconds as input.
  // The value is computed by linear interpolation.
  absl::StatusOr<eigenmath::VectorXd> GetVelocityAtTime(double time_sec) const;

  // Returns the acceleration sample at `time`, or a status on error.
  // The value is computed by linear interpolation.
  absl::StatusOr<eigenmath::VectorXd> GetAccelerationAtTime(
      absl::Time time) const;

  // Same as above, but takes the time in seconds as input.
  // The value is computed by linear interpolation.
  absl::StatusOr<eigenmath::VectorXd> GetAccelerationAtTime(
      double time_sec) const;

  // Add `offset` to all timestamps.
  void AddOffsetToTimestamps(absl::Duration offset);
  // Add `offset` to all timestamps.
  void AddOffsetToTimestamps(double offset);

 private:
  TrajectoryBuffer() = default;
  explicit TrajectoryBuffer(TrajectoryBufferOptions options)
      : options_(options) {}
  int GetIndexForTimeUnchecked(absl::Time time) const;

  // Returns an index pair (first, second) for the samples bracketing
  // `time_sec`, such that first <= time_sec <= second. Returns a status if an
  // error occurred (e.g., if time_sec is outside the range of trajectory
  // samples).
  absl::StatusOr<std::pair<int, int>> GetOffsetBracket(double time_sec) const;

  TrajectoryBufferOptions options_;
  int sequence_number_ = 0;
  std::vector<double> times_;
  std::vector<eigenmath::VectorXd> positions_;
  std::vector<eigenmath::VectorXd> velocities_;
  std::vector<eigenmath::VectorXd> accelerations_;
};

}  // namespace trajectory_planning

#endif  // TRAJECTORY_PLANNING_TRAJECTORY_PLANNING_TRAJECTORY_BUFFER_H_

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

#include "trajectory_planning/trajectory_buffer.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "eigenmath/interpolation.h"
#include "eigenmath/types.h"
#include "trajectory_planning/rescale_to_stop.h"
#include "trajectory_planning/sampled_trajectory.h"
#include "trajectory_planning/time.h"

namespace trajectory_planning {

using eigenmath::VectorXd;

absl::StatusOr<std::unique_ptr<TrajectoryBuffer>> TrajectoryBuffer::Create(
    TrajectoryBufferOptions options) {
  if (options.timestep_tolerance <= 0) {
    return absl::FailedPreconditionError(absl::StrCat(
        "timestep_tolerance (", options.timestep_tolerance, ") not positive"));
  }
  // Use WrapUnique as the ctor is private.
  return absl::WrapUnique(new TrajectoryBuffer(options));
}

absl::Time TrajectoryBuffer::GetStartTime() const {
  if (times_.empty()) {
    return TimeFromSec(0);
  }
  return TimeFromSec(times_.front());
}

absl::Time TrajectoryBuffer::GetEndTime() const {
  if (times_.empty()) {
    return absl::Time();
  }
  return TimeFromSec(times_.back());
}

void TrajectoryBuffer::Clear() {
  sequence_number_ = 0;
  times_.clear();
  positions_.clear();
  velocities_.clear();
  accelerations_.clear();
}

void TrajectoryBuffer::Reserve(size_t size) {
  times_.reserve(size);
  positions_.reserve(size);
  velocities_.reserve(size);
  accelerations_.reserve(size);
}

absl::Status TrajectoryBuffer::InsertSegment(
    absl::Span<const double> times, absl::Span<const VectorXd> positions,
    absl::Span<const VectorXd> velocities,
    absl::Span<const VectorXd> accelerations) {
  if (positions.size() != velocities.size()) {
    return absl::InvalidArgumentError(
        "positions and velocity arguments have different size.");
  }
  if (positions.size() != accelerations.size()) {
    return absl::InvalidArgumentError(
        "positions and accelerations arguments have different size.");
  }
  if (positions.size() != times.size()) {
    return absl::InvalidArgumentError(
        "positions and times arguments have different size.");
  }

  sequence_number_++;

  if (positions.empty()) {
    return absl::OkStatus();
  }

  // First sample with times_[k] greater or equal to times.front().
  auto it_upper =
      absl::c_upper_bound(times_, times.front(),
                          [](const auto& a, const auto& b) { return a <= b; });
  if (positions_.empty() || it_upper == times_.begin()) {
    times_.assign(times.begin(), times.end());
    positions_.assign(positions.begin(), positions.end());
    velocities_.assign(velocities.begin(), velocities.end());
    accelerations_.assign(accelerations.begin(), accelerations.end());
    sequence_number_ = 0;
    return absl::OkStatus();
  }

  // The first of the new samples is just beyond an existing sample: decrease
  // upper bound iterator s.t. we replace the almost identical sample.
  if (times.front() - *std::prev(it_upper) < options_.timestep_tolerance) {
    it_upper--;
  }
  const int samples_to_keep = std::distance(times_.begin(), it_upper);
  times_.resize(samples_to_keep);
  positions_.resize(samples_to_keep);
  velocities_.resize(samples_to_keep);
  accelerations_.resize(samples_to_keep);

  times_.insert(times_.end(), times.begin(), times.end());
  positions_.insert(positions_.end(), positions.begin(), positions.end());
  velocities_.insert(velocities_.end(), velocities.begin(), velocities.end());
  accelerations_.insert(accelerations_.end(), accelerations.begin(),
                        accelerations.end());

  return absl::OkStatus();
}

absl::Status TrajectoryBuffer::AppendSample(
    double time, const eigenmath::VectorXd& positions,
    const eigenmath::VectorXd& velocities,
    const eigenmath::VectorXd& accelerations) {
  if (!times_.empty() && times_.back() >= time) {
    return absl::InvalidArgumentError("time must be > times_.back().");
  }

  times_.push_back(time);
  positions_.push_back(positions);
  velocities_.push_back(velocities);
  accelerations_.push_back(accelerations);

  return absl::OkStatus();
}

void TrajectoryBuffer::DiscardSegmentBefore(absl::Time time) {
  return DiscardSegmentBefore(TimeToSec(time));
}

void TrajectoryBuffer::DiscardSegmentBefore(const double time_sec) {
  if (times_.empty()) {
    return;
  }
  if (time_sec <= times_.front()) {
    return;
  }
  if (time_sec > times_.back()) {
    Clear();
    return;
  }

  // First timestamp that does not satisfy 'time < time_sec'.
  auto it = absl::c_upper_bound(
      times_, time_sec, [](const double a, const double b) { return a <= b; });
  int offset = std::distance(times_.begin(), it);

  // Nothing to discard.
  if (offset <= 0) {
    return;
  }
  // If the time sample before 'offset' is very close to 'time_sec', keep it.
  // Also keep it if we will need to create a new initial sample by
  // interpolation.
  const bool close_to_existing_sample =
      time_sec - times_[offset - 1] <= options_.timestep_tolerance;
  const bool create_initial_sample_by_interpolation =
      std::abs(times_[offset] - time_sec) > options_.timestep_tolerance;

  if (close_to_existing_sample || create_initial_sample_by_interpolation) {
    --offset;
  }

  // Get interpolated positions at `time` if necessary.
  if (create_initial_sample_by_interpolation) {
    const auto start_position = GetPositionAtTime(time_sec);
    const auto start_velocity = GetVelocityAtTime(time_sec);
    const auto start_acceleration = GetAccelerationAtTime(time_sec);
    // Should never fail by construction.
    CHECK(start_position.status().ok());
    CHECK(start_velocity.status().ok());
    CHECK(start_acceleration.status().ok());
    times_[offset] = time_sec;
    positions_[offset] = *start_position;
    velocities_[offset] = *start_velocity;
    accelerations_[offset] = *start_acceleration;
  }

  // Erase elemnts from [begin, begin+offset).
  times_.erase(times_.begin(), times_.begin() + offset);
  positions_.erase(positions_.begin(), positions_.begin() + offset);
  velocities_.erase(velocities_.begin(), velocities_.begin() + offset);
  accelerations_.erase(accelerations_.begin(), accelerations_.begin() + offset);
}

absl::Span<const eigenmath::VectorXd> TrajectoryBuffer::GetPositionsUpToTime(
    absl::Time time) const {
  constexpr absl::Span<const eigenmath::VectorXd> kEmptySpan;

  if (times_.empty()) {
    return kEmptySpan;
  }
  const double time_sec = TimeToSec(time);
  if (time_sec < times_.front() || time_sec > times_.back()) {
    return kEmptySpan;
  }

  // First timestamp greater than `time`.
  auto it = absl::c_upper_bound(times_, time_sec);
  const int num_elements = std::distance(times_.begin(), it - 1);
  return absl::MakeConstSpan(&positions_[0], num_elements);
}

absl::StatusOr<VectorXd> TrajectoryBuffer::GetPositionAtTime(
    const absl::Time time) const {
  return GetPositionAtTime(TimeToSec(time));
}

absl::StatusOr<std::pair<int, int>> TrajectoryBuffer::GetOffsetBracket(
    double time_sec) const {
  if (times_.empty()) {
    return absl::FailedPreconditionError("No samples.");
  }
  if (time_sec < times_.front() || time_sec > times_.back()) {
    return absl::OutOfRangeError(
        absl::StrFormat("Time: %.4f, start_time: %.4f, end_time: %.4f",
                        time_sec, times_.front(), times_.back()));
  }

  const auto it_upper = absl::c_upper_bound(times_, time_sec);
  if (it_upper == times_.end()) {
    return std::make_pair(times_.size() - 1, times_.size() - 1);
  }
  const auto it_lower = std::prev(it_upper);
  return std::make_pair(std::distance(times_.begin(), it_lower),
                        std::distance(times_.begin(), it_upper));
}

absl::StatusOr<VectorXd> TrajectoryBuffer::GetPositionAtTime(
    const double time_sec) const {
  const auto offsets = GetOffsetBracket(time_sec);
  if (!offsets.ok()) {
    return offsets.status();
  }
  return eigenmath::InterpolateLinear(
      time_sec, times_[offsets->first], times_[offsets->second],
      positions_[offsets->first], positions_[offsets->second]);
}

absl::StatusOr<VectorXd> TrajectoryBuffer::GetVelocityAtTime(
    const absl::Time time) const {
  return GetVelocityAtTime(TimeToSec(time));
}

absl::StatusOr<VectorXd> TrajectoryBuffer::GetVelocityAtTime(
    const double time_sec) const {
  const auto offsets = GetOffsetBracket(time_sec);
  if (!offsets.ok()) {
    return offsets.status();
  }
  return eigenmath::InterpolateLinear(
      time_sec, times_[offsets->first], times_[offsets->second],
      velocities_[offsets->first], velocities_[offsets->second]);
}

absl::StatusOr<eigenmath::VectorXd> TrajectoryBuffer::GetAccelerationAtTime(
    absl::Time time) const {
  return GetAccelerationAtTime(TimeToSec(time));
}

absl::StatusOr<eigenmath::VectorXd> TrajectoryBuffer::GetAccelerationAtTime(
    double time_sec) const {
  const auto offsets = GetOffsetBracket(time_sec);
  if (!offsets.ok()) {
    return offsets.status();
  }
  return eigenmath::InterpolateLinear(
      time_sec, times_[offsets->first], times_[offsets->second],
      accelerations_[offsets->first], accelerations_[offsets->second]);
}

absl::Status TrajectoryBuffer::StopAtIndex(int index,
                                           const VectorXd& max_acceleration,
                                           const double time_step) {
  if (index <= 0 || index > GetNumSamples() - 1) {
    return absl::OutOfRangeError(absl::StrCat("index (", index,
                                              ")  out of range (", 0, ", ",
                                              GetNumSamples() - 1, "]."));
  }

  if (max_acceleration.minCoeff() <= 0.0) {
    return absl::InvalidArgumentError(
        absl::StrCat("max_acceleration has non-positive minimum coefficient ",
                     max_acceleration.minCoeff()));
  }

  if (time_step <= 0.0) {
    return absl::InvalidArgumentError(
        absl::StrCat("`time_step` should be positive but is ", time_step, "."));
  }

  constexpr double kVerySmall = 1e-4;
  if ((index == GetNumSamples() - 1) &&
      (velocities_.back().lpNorm<Eigen::Infinity>() < kVerySmall)) {
    velocities_.back().setZero();
    accelerations_.back().setZero();
    return absl::OkStatus();
  }

  const int samples_for_stop = index + 1;
  absl::Span<const double> times_for_stop =
      absl::MakeConstSpan(times_.data(), samples_for_stop);
  absl::Span<const VectorXd> positions_for_stop =
      absl::MakeConstSpan(positions_.data(), samples_for_stop);
  absl::Span<const VectorXd> velocities_for_stop =
      absl::MakeConstSpan(velocities_.data(), samples_for_stop);
  absl::Span<const VectorXd> accelerations_for_stop =
      absl::MakeConstSpan(accelerations_.data(), samples_for_stop);

  const auto rescaled_stop_status = RescaleTrajectoryBackwardToStop(
      max_acceleration, times_for_stop, positions_for_stop, velocities_for_stop,
      accelerations_for_stop);
  if (!rescaled_stop_status.ok()) {
    return rescaled_stop_status.status();
  }
  const auto rescaled_stop = *rescaled_stop_status;

  CHECK(!rescaled_stop.times.empty());

  // If the stopping trajectory uses all available path samples, verify that
  // the initial velocity is matched approximately.
  if (rescaled_stop.times.size() == index) {
    const auto get_velocity_result =
        GetVelocityAtTime(rescaled_stop.times.front());
    if (!get_velocity_result.ok()) {
      return get_velocity_result.status();
    }
    const VectorXd velocity_at_start = *get_velocity_result;
    constexpr double kAcceptableMatchError = 1e-2;
    if ((velocity_at_start - rescaled_stop.velocities.front())
            .lpNorm<Eigen::Infinity>() > kAcceptableMatchError) {
      return absl::NotFoundError(
          "No safe stopping trajectory found (likely not enough time).");
    }
  }
  return InsertSegment(rescaled_stop.times, rescaled_stop.positions,
                       rescaled_stop.velocities, rescaled_stop.accelerations);
}

absl::Status TrajectoryBuffer::StopBeforeTime(
    absl::Time time, const eigenmath::VectorXd& max_acceleration,
    double time_step) {
  return StopBeforeTime(TimeToSec(time), max_acceleration, time_step);
}

absl::Status TrajectoryBuffer::StopBeforeTime(
    double time_sec, const eigenmath::VectorXd& max_acceleration,
    double time_step) {
  if (times_.empty()) {
    return absl::OkStatus();
  }
  if (time_sec < times_.front()) {
    return absl::OutOfRangeError("time < times_.front().");
  }

  // First timestamp equal to or greater than `time`.
  auto it_upper = absl::c_lower_bound(times_, time_sec);
  const int index = std::min<int>(std::distance(times_.begin(), it_upper + 1),
                                  times_.size() - 1);
  return StopAtIndex(index, max_acceleration, time_step);
}

void TrajectoryBuffer::AddOffsetToTimestamps(const absl::Duration offset) {
  AddOffsetToTimestamps(offset / absl::Seconds(1));
}
void TrajectoryBuffer::AddOffsetToTimestamps(const double offset) {
  std::transform(times_.cbegin(), times_.cend(), times_.begin(),
                 [&offset](const double time) { return time + offset; });
}

}  // namespace trajectory_planning

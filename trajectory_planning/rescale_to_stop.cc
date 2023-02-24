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

#include "trajectory_planning/rescale_to_stop.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/log/check.h"
#include "absl/strings/substitute.h"
#include "eigenmath/types.h"
#include "trajectory_planning/sampled_trajectory.h"

using ::eigenmath::VectorXd;

namespace trajectory_planning {

absl::StatusOr<SampledTrajectory> RescaleTrajectoryBackwardToStop(
    const VectorXd& max_acceleration, absl::Span<const double> times,
    absl::Span<const VectorXd> positions, absl::Span<const VectorXd> velocities,
    absl::Span<const VectorXd> accelerations) {
  // The stopping trajectory is generated by:
  // 1) Re-parameterizing the trajectory, s.t. positions = positions(rate(t)),
  //    where rate(t) is the re-scaled time parameter.
  //    This assumes that the `positions`, `velocities` and `accelerations`
  //    arguments are samples from some underlying vector-valued function of
  //    t.
  // 2) Computing rate(t) backward, starting at the end of the trajectory.
  //    This is achieved by computing the maximum admissible (squared)
  //    derivative, from which the scaling rate is computed by explicit backward
  //    integration.
  //    Backward integration continues until the scaling rate is 1.0 (that is,
  //    the original trajectory's velocity is reached).
  //    See further comments inline below for this part of the computation.
  // 3) Computing new time samples from the scaling rate.
  // 4) Shifting time samples s.t. to line up the stopping segment with the
  //    input trajectory.

  if (auto status = AreInputsValidForSampledTrajectory(
          times, positions, velocities, accelerations);
      !status.ok()) {
    return status;
  }

  // If the trajectory already ends in a stop, return an empty stopping
  // trajectory.
  constexpr double kTiny = 1e-8;
  if (velocities.back().lpNorm<Eigen::Infinity>() < kTiny) {
    return SampledTrajectory{};
  }

  // Compute new time samples by backward integration along a maximum
  // deceleration extremal.
  const int joint_count = max_acceleration.size();
  const int sample_count = times.size();
  std::vector<double> rescaled_times;
  std::vector<VectorXd> rescaled_velocities;
  std::vector<VectorXd> rescaled_accelerations;

  // Begin at zero time, velocity & acceleration.
  rescaled_times.push_back(0.0);
  rescaled_accelerations.push_back(VectorXd::Zero(joint_count));
  rescaled_velocities.push_back(VectorXd::Zero(joint_count));

  double rate_squared = 0.0;
  double diff_rate_squared = 0.0;
  for (int i = sample_count - 1; i > 1; --i) {
    // The scaled acceleration is
    // 0.5*velocities[i]*diff_rate_squared+accelerations[i]*rate_squared.
    // This optained by differentiating position(rate(t)) w.r.t. t using the
    // chain rule.
    // The fastest stopping trajectory will have the fastest rate of change of
    // the scaling `rate`, s.t. the rescaled acceleration is still valid for
    // all joints.
    const VectorXd acceleration_bias = accelerations[i] * rate_squared;
    const VectorXd& velocity = velocities[i];
    diff_rate_squared = 0.0;
    constexpr double kTiny = 1e-8;
    // For each joint, solve the equation scaled_accleleration[joint] ==
    // -max_acceleration[joint] for diff_rate_squared,
    // and determine the smallest viable derivative of `rate_squared`.
    // That solution will have at least one (rescaled) joint acceleration at
    // the maximum.
    for (int joint = 0; joint < joint_count; ++joint) {
      if (std::abs(velocity[joint]) < kTiny) {
        continue;
      }
      for (const double sign : {-1.0, 1.0}) {
        const double diff_rate_squared_joint =
            -2.0 * (acceleration_bias[joint] + sign * max_acceleration[joint]) /
            velocity[joint];
        VectorXd scaled_acceleration(acceleration_bias +
                                     0.5 * velocity * diff_rate_squared_joint);
        const bool acceleration_valid =
            (max_acceleration - scaled_acceleration).minCoeff() >= -kTiny &&
            (-max_acceleration - scaled_acceleration).maxCoeff() <= kTiny;
        if (acceleration_valid && diff_rate_squared_joint < diff_rate_squared) {
          diff_rate_squared = diff_rate_squared_joint;
        }
      }
    }
    const double unscaled_dt = times[i] - times[i - 1];
    const double next_rate_squared =
        rate_squared - diff_rate_squared * unscaled_dt;
    // Use a clamped rate_squared for integration to ensure we don't overshoot
    // the original trajectory's velocity.
    const double clamped_rate_squared = std::min(next_rate_squared, 1.0);
    // Compute the new time delta via the trapezoidal rule.
    const double new_time_delta =
        2.0 * unscaled_dt /
        (std::sqrt(rate_squared) + std::sqrt(clamped_rate_squared));
    rescaled_times.push_back(rescaled_times.back() - new_time_delta);
    rescaled_velocities.push_back(std::sqrt(clamped_rate_squared) * velocity);
    rescaled_accelerations.push_back(acceleration_bias +
                                     0.5 * velocity * diff_rate_squared);
    // If we passed the original trajectory's velocity we're done.
    if (next_rate_squared >= 1.0) {
      break;
    }
    rate_squared = next_rate_squared;
  }

  // Reverse the trajectory so time runs forward with increasing indices.
  absl::c_reverse(rescaled_times);
  absl::c_reverse(rescaled_accelerations);
  absl::c_reverse(rescaled_velocities);

  // Fix the time offset, so the trajectory lines up with the input.
  const int switch_index = times.size() - rescaled_times.size();
  CHECK(switch_index >= 0) << absl::Substitute("switch_index= $0",
                                               switch_index);
  const auto stopping_positions =
      positions.subspan(switch_index, rescaled_times.size());
  const double switch_time = times[switch_index];
  const double time_offset = switch_time - rescaled_times.front();

  for (auto& time : rescaled_times) {
    time += time_offset;
  }

  return SampledTrajectory{
      .times = std::move(rescaled_times),
      .positions = {stopping_positions.begin(), stopping_positions.end()},
      .velocities = std::move(rescaled_velocities),
      .accelerations = std::move(rescaled_accelerations),
  };
}
}  // namespace trajectory_planning

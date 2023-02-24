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

#ifndef TRAJECTORY_PLANNING_TRAJECTORY_PLANNING_SAMPLED_TRAJECTORY_H_
#define TRAJECTORY_PLANNING_TRAJECTORY_PLANNING_SAMPLED_TRAJECTORY_H_

#include <vector>

#include "eigenmath/types.h"
#include "absl/status/status.h"
#include "absl/types/span.h"

namespace trajectory_planning {

// A (joint space) trajectory.
struct SampledTrajectory {
  // The time samples.
  std::vector<double> times;
  // Position samples of the trajectory.
  std::vector<eigenmath::VectorXd> positions;
  // Velocity samples of the trajectory.
  std::vector<eigenmath::VectorXd> velocities;
  // Acceleration samples of the trajectory.
  std::vector<eigenmath::VectorXd> accelerations;
};

// A (joint space) trajectory, uniformly sampled in time.
struct UniformlySampledTrajectory {
  // The sampling time.
  double time_step_sec = 0.0;
  // The starting time.
  double start_time_sec = 0.0;
  // Position samples of the trajectory.
  std::vector<eigenmath::VectorXd> positions;
  // Velocity samples of the trajectory.
  std::vector<eigenmath::VectorXd> velocities;
  // Acceleration samples of the trajectory.
  std::vector<eigenmath::VectorXd> accelerations;
};

// Returns an OkStatus if the given inputs would constitute a valid sampled
// trajectory.
absl::Status AreInputsValidForSampledTrajectory(
    absl::Span<const double> times,
    absl::Span<const eigenmath::VectorXd> positions,
    absl::Span<const eigenmath::VectorXd> velocities,
    absl::Span<const eigenmath::VectorXd> accelerations);

}  // namespace trajectory_planning

#endif  // TRAJECTORY_PLANNING_TRAJECTORY_PLANNING_SAMPLED_TRAJECTORY_H_

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

#ifndef TRAJECTORY_PLANNING_TRAJECTORY_PLANNING_RESCALE_TO_STOP_H_
#define TRAJECTORY_PLANNING_TRAJECTORY_PLANNING_RESCALE_TO_STOP_H_

#include "eigenmath/types.h"
#include "trajectory_planning/sampled_trajectory.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace trajectory_planning {

// Computes a stopping trajectory by time scaling the given trajectory, starting
// from the end.
// Returns a SampledTrajectory that ends in a stop and traces the tail of
// `positions`. Velocities and accelerations are computed s.t.
// `max_acceleration` is not exceeded.
// Returns a Status if an error occurred.
absl::StatusOr<SampledTrajectory> RescaleTrajectoryBackwardToStop(
    const eigenmath::VectorXd& max_acceleration, absl::Span<const double> times,
    absl::Span<const eigenmath::VectorXd> positions,
    absl::Span<const eigenmath::VectorXd> velocities,
    absl::Span<const eigenmath::VectorXd> accelerations);

}  // namespace trajectory_planning

#endif  // TRAJECTORY_PLANNING_TRAJECTORY_PLANNING_RESCALE_TO_STOP_H_

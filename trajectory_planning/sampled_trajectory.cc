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

#include "trajectory_planning/sampled_trajectory.h"

#include "eigenmath/types.h"
#include "absl/status/status.h"

using ::eigenmath::VectorXd;

namespace trajectory_planning {

absl::Status AreInputsValidForSampledTrajectory(
    absl::Span<const double> times, absl::Span<const VectorXd> positions,
    absl::Span<const VectorXd> velocities,
    absl::Span<const VectorXd> accelerations) {
  const int sample_count = times.size();
  if (sample_count < 2) {
    return absl::InvalidArgumentError("Need at least two samples.");
  }
  if (positions.size() != sample_count || velocities.size() != sample_count ||
      accelerations.size() != sample_count) {
    return absl::InvalidArgumentError("inconsistent sizes for samples.");
  }

  for (int i = 0; i < sample_count - 1; ++i) {
    if (times[i + 1] <= times[i]) {
      return absl::InvalidArgumentError(
          "Time samples not strictly increasing.");
    }
  }

  return absl::OkStatus();
}

}  // namespace trajectory_planning

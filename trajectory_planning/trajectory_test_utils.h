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

#ifndef TRAJECTORY_PLANNING_TRAJECTORY_PLANNING_TRAJECTORY_TEST_UTILS_H_
#define TRAJECTORY_PLANNING_TRAJECTORY_PLANNING_TRAJECTORY_TEST_UTILS_H_

#include "eigenmath/types.h"
#include "absl/types/span.h"

namespace trajectory_planning {

enum class FiniteDifferenceMethod { kBackward, kSymmetrical, kForward };

eigenmath::VectorXd FiniteDifferences(const eigenmath::VectorXd& prev,
                                      const eigenmath::VectorXd& current,
                                      const eigenmath::VectorXd& next,
                                      const double prev_time,
                                      const double current_time,
                                      const double next_time,
                                      FiniteDifferenceMethod method) {
  switch (method) {
    case FiniteDifferenceMethod::kSymmetrical:
      return (next - prev) / (next_time - prev_time);
    case FiniteDifferenceMethod::kBackward:
      return (current - prev) / (current_time - prev_time);
    case FiniteDifferenceMethod::kForward:
      return (next - current) / (next_time - current_time);
  }
  return eigenmath::VectorXd();
}

void ExpectConsistentFiniteDifferenceDerivatives(
    absl::Span<const double> times,
    absl::Span<const eigenmath::VectorXd> positions,
    absl::Span<const eigenmath::VectorXd> velocities,
    const double acceptable_difference,
    FiniteDifferenceMethod method = FiniteDifferenceMethod::kSymmetrical) {
  ASSERT_EQ(positions.size(), velocities.size());
  if (positions.size() <= 1) {
    return;
  }
  for (int i = 1; i < positions.size() - 1; ++i) {
    SCOPED_TRACE(absl::StrCat("At sample: ", i, " / ", positions.size()));
    const eigenmath::VectorXd velocity_numerical =
        FiniteDifferences(positions[i - 1], positions[i], positions[i + 1],
                          times[i - 1], times[i], times[i + 1], method);
    EXPECT_THAT(velocities[i], eigenmath::testing::IsApprox(
                                   velocity_numerical, acceptable_difference))
        << "diff= " << (velocities[i] - velocity_numerical).transpose();
  }
}

} // namespace trajectory_planning

#endif  // TRAJECTORY_PLANNING_TRAJECTORY_PLANNING_TRAJECTORY_TEST_UTILS_H_

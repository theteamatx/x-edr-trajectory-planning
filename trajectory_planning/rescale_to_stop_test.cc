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

#include <vector>

#include "absl/status/status.h"
#include "eigenmath/matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::eigenmath::VectorXd;
using ::eigenmath::testing::IsApprox;

namespace trajectory_planning {
namespace {
TEST(RescaleTrajectoryBackwardToStopTest, SucceedsForConstantVelocity) {
  constexpr int kSampleCount = 200;
  constexpr double kTimeStepSec = 8e-3;
  constexpr int kJointCount = 4;
  constexpr double kMaxAcceleration = 2.0;

  for (double velocity : {-1.0, 1.0}) {
    SCOPED_TRACE(absl::StrCat("velocity= ", velocity));
    std::vector<double> time_samples(kSampleCount);
    std::vector<VectorXd> position_samples(kSampleCount);
    std::vector<VectorXd> velocity_samples(kSampleCount);
    std::vector<VectorXd> acceleration_samples(kSampleCount);

    time_samples[0] = 0.0;
    position_samples[0].setZero(kJointCount);
    velocity_samples[0].setConstant(kJointCount, velocity);
    acceleration_samples[0].setZero(kJointCount);
    for (int i = 1; i < kSampleCount; ++i) {
      time_samples[i] = i * kTimeStepSec;
      acceleration_samples[i].setZero(kJointCount);
      velocity_samples[i].setConstant(kJointCount, velocity);
      position_samples[i].setConstant(kJointCount, velocity * time_samples[i]);
    }

    auto status_or_result = RescaleTrajectoryBackwardToStop(
        VectorXd::Constant(kJointCount, kMaxAcceleration), time_samples,
        position_samples, velocity_samples, acceleration_samples);
    ASSERT_TRUE(status_or_result.ok());
    SampledTrajectory result = *status_or_result;

    // Compute exact solution for comparison.
    const double ramp_up_time_sec = std::abs(velocity) / kMaxAcceleration;
    const double stopping_duration =
        velocity * velocity / (2.0 * kMaxAcceleration);

    ASSERT_GT(result.positions.size(), 0);
    ASSERT_GT(result.velocities.size(), 0);
    ASSERT_GT(result.accelerations.size(), 0);
    ASSERT_EQ(result.times.size(), result.positions.size());
    ASSERT_EQ(result.times.size(), result.velocities.size());
    ASSERT_EQ(result.times.size(), result.accelerations.size());

    EXPECT_NEAR(result.times.back() - result.times.front(), ramp_up_time_sec,
                kTimeStepSec);

    EXPECT_THAT(result.accelerations.back(),
                IsApprox(VectorXd::Zero(kJointCount)));
    EXPECT_THAT(result.velocities.back(),
                IsApprox(VectorXd::Zero(kJointCount)));
    double velocity_sign = (velocity == 0 ? 0 : (velocity < 0 ? -1 : 1));
    EXPECT_THAT((result.positions.back() - result.positions.front()),
                IsApprox(VectorXd::Constant(kJointCount,
                                            stopping_duration * velocity_sign),
                         std::abs(velocity * kTimeStepSec)));
  }
}

}  // namespace
}  // namespace trajectory_planning

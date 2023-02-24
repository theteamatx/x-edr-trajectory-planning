#include "trajectory_planning/sampled_trajectory.h"

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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"

using ::eigenmath::VectorXd;
using ::testing::HasSubstr;

namespace trajectory_planning {
namespace {

MATCHER_P2(StatusIs, s, str_matcher, "") {
  return arg.code() == s && testing::ExplainMatchResult(
                                str_matcher, arg.message(), result_listener);
}

TEST(AreInputsValidForSampledTrajectoryTest, FailsOnInvalidInput) {
  EXPECT_THAT(AreInputsValidForSampledTrajectory(
                  std::vector<double>(3), std::vector<VectorXd>(2),
                  std::vector<VectorXd>(2), std::vector<VectorXd>(2)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("inconsistent sizes")));

  EXPECT_THAT(AreInputsValidForSampledTrajectory(
                  {1.0, 1.0}, std::vector<VectorXd>(2),
                  std::vector<VectorXd>(2), std::vector<VectorXd>(2)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("not strictly increasing")));
  EXPECT_THAT(AreInputsValidForSampledTrajectory(
                  {1.0, 0.5}, std::vector<VectorXd>(2),
                  std::vector<VectorXd>(2), std::vector<VectorXd>(2)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("not strictly increasing")));

  EXPECT_THAT(AreInputsValidForSampledTrajectory(
                  {1.0}, std::vector<VectorXd>(1), std::vector<VectorXd>(1),
                  std::vector<VectorXd>(1)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("at least two samples")));

  // No checking of joint vector lengths, as those are much less likely to be
  // inconsistent.
}

}  // namespace
}  // namespace trajectory_planning

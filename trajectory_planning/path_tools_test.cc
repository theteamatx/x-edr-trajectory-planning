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

#include "trajectory_planning/path_tools.h"

#include <vector>

#include "absl/status/status.h"
#include "eigenmath/matchers.h"
#include "eigenmath/pose3.h"
#include "eigenmath/types.h"
#include "eigenmath/utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace trajectory_planning {
namespace {
using ::eigenmath::testing::IsApprox;
using ::testing::HasSubstr;
using ::testing::Not;

MATCHER_P(StatusIs, s, "") { return arg.code() == s; }
MATCHER_P2(StatusIs, s, str_matcher, "") {
  return arg.code() == s && testing::ExplainMatchResult(
                                str_matcher, arg.message(), result_listener);
}

constexpr double kEpsilon = 1e-10;

TEST(ProjectPointOnPath, FailsForInvalidArguments) {
  std::vector<eigenmath::VectorXd> waypoints;
  eigenmath::VectorXd point;

  EXPECT_THAT(
      ProjectPointOnPath(waypoints, point).status(),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("No waypoints")));

  point.resize(4);
  waypoints.resize(2);
  EXPECT_THAT(ProjectPointOnPath(waypoints, point).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr(" number of joints:")));
}

TEST(ProjectPointOnPath, HandlesSpecialCaseOfOnePoint) {
  const std::vector<eigenmath::VectorXd> waypoints(
      {eigenmath::MakeVector({1.0, 1.0})});
  const eigenmath::VectorXd point(eigenmath::MakeVector({1.0, 1.0}));

  const auto result = ProjectPointOnPath(waypoints, point);
  ASSERT_TRUE(result.ok());

  EXPECT_EQ(result->waypoint_index, 0);
  EXPECT_DOUBLE_EQ(result->distance_to_path, 0.0);
  EXPECT_DOUBLE_EQ(result->line_parameter, 0.0);
  EXPECT_THAT(result->projected_point, IsApprox(point, kEpsilon));
}

TEST(ProjectPointOnPath, WorksWhenFirstPointIsClosest) {
  const std::vector<eigenmath::VectorXd> waypoints(
      {eigenmath::MakeVector({1.0, 1.0}), eigenmath::MakeVector({2.0, 2.0})});
  const eigenmath::VectorXd point(eigenmath::MakeVector({1.0, 1.0}));

  const auto result = ProjectPointOnPath(waypoints, point);
  ASSERT_TRUE(result.ok());

  EXPECT_EQ(result->waypoint_index, 0);
  EXPECT_DOUBLE_EQ(result->distance_to_path, 0.0);
  EXPECT_DOUBLE_EQ(result->line_parameter, 0.0);
  EXPECT_THAT(result->projected_point, IsApprox(point, kEpsilon));
}

TEST(ProjectPointOnPath, WorksWhenLastPointIsClosest) {
  const std::vector<eigenmath::VectorXd> waypoints(
      {eigenmath::MakeVector({1.0, 1.0}), eigenmath::MakeVector({2.0, 2.0})});
  const eigenmath::VectorXd point(eigenmath::MakeVector({2.0, 2.0}));

  const auto result = ProjectPointOnPath(waypoints, point);
  ASSERT_TRUE(result.ok());

  EXPECT_EQ(result->waypoint_index, 0);
  EXPECT_DOUBLE_EQ(result->distance_to_path, 0.0);
  EXPECT_DOUBLE_EQ(result->line_parameter, 1.0);
  EXPECT_THAT(result->projected_point, IsApprox(point, kEpsilon));
}

TEST(ProjectPointOnPath, WorksWhenClosestPointIsBetweenWaypoints) {
  const std::vector<eigenmath::VectorXd> waypoints(
      {eigenmath::MakeVector({1.0, 1.0}), eigenmath::MakeVector({2.0, 2.0}),
       eigenmath::MakeVector({-3.0, -3.0})});
  constexpr double kLineParameter = 0.4;
  const eigenmath::VectorXd projected_point(
      waypoints[1] + kLineParameter * (waypoints[2] - waypoints[1]));
  const eigenmath::VectorXd point =
      projected_point + eigenmath::MakeVector({0.1, -0.1});

  const auto result = ProjectPointOnPath(waypoints, point);
  ASSERT_TRUE(result.ok());

  EXPECT_EQ(result->waypoint_index, 1);
  EXPECT_DOUBLE_EQ(result->distance_to_path, (projected_point - point).norm());
  EXPECT_DOUBLE_EQ(result->line_parameter, kLineParameter);
  EXPECT_THAT(result->projected_point, IsApprox(projected_point, kEpsilon));
}

TEST(ComputeStoppingPoint, ReturnsStatusOnInvalidInput) {
  EXPECT_THAT(
      ComputeStoppingPoint(std::vector<double>(2), std::vector<double>(1),
                           std::vector<double>(2), 0.0)
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("size")));
  EXPECT_THAT(
      ComputeStoppingPoint(std::vector<double>(2), std::vector<double>(2),
                           std::vector<double>(1), 0.0)
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("size")));
  EXPECT_THAT(
      ComputeStoppingPoint(std::vector<double>(2), std::vector<double>(2),
                           std::vector<double>(2, -1.0), 0.0)
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("positive")));
}

TEST(ComputeStoppingPoint, ReturnsPositionIfForZeroVelocity) {
  eigenmath::VectorXd position = eigenmath::VectorXd::Constant(3, 1.0);
  eigenmath::VectorXd velocity = eigenmath::VectorXd::Constant(3, 0.0);
  eigenmath::VectorXd acceleration = eigenmath::VectorXd::Constant(3, 1.0);
  constexpr double kCornerRounding = 0.1;
  auto stop =
      ComputeStoppingPoint(position, velocity, acceleration, kCornerRounding);
  ASSERT_TRUE(stop.ok());

  EXPECT_THAT(*stop, IsApprox(position, 0.0));
}

TEST(ComputeStoppingPoint, StoppingPointIsInDirectionOfVelocity) {
  eigenmath::VectorXd position = eigenmath::VectorXd::Constant(3, 1.0);
  eigenmath::VectorXd velocity = eigenmath::VectorXd::Constant(3, 1.0);
  eigenmath::VectorXd acceleration = eigenmath::VectorXd::Constant(3, 1.0);
  constexpr double kCornerRounding = 0.1;
  auto stop =
      ComputeStoppingPoint(position, velocity, acceleration, kCornerRounding);
  ASSERT_TRUE(stop.ok());
  EXPECT_THAT(*stop, Not(IsApprox(position, 0.0)));

  // If `stop` is on the ray from `position` in the direction of `velocity`,
  // the rank of the matrix of directions will be 1.
  eigenmath::MatrixXd directions(position.size(), 2);
  directions.col(0) = velocity;
  directions.col(1) = *stop - position;

  Eigen::FullPivLU<eigenmath::MatrixXd> lu(directions);
  EXPECT_EQ(lu.rank(), 1);
}

}  // namespace
}  // namespace trajectory_planning

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

#include "trajectory_planning/splines/spline_utils.h"

#include <vector>

#include "trajectory_planning/splines/bspline.h"
#include "eigenmath/matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace trajectory_planning {
namespace {

using eigenmath::testing::IsApprox;

#define ASSERT_OK(x) ASSERT_TRUE(x.ok());

TEST(PolyLineToBspline3WaypointsPosesTest, OneCorner) {
  std::vector<eigenmath::Pose3d> corners, control_points;
  corners.emplace_back(eigenmath::Vector3d(1, 2, 3));

  PolyLineToBspline3Waypoints(corners, 0, 0, &control_points);

  ASSERT_EQ(4, control_points.size());
  for (const auto& control_point : control_points) {
    EXPECT_THAT(control_point, IsApprox(corners[0]));
  }
}

TEST(PolyLineToBspline3WaypointsPosesTest, Translation) {
  std::vector<eigenmath::Pose3d> corners, control_points;
  corners.emplace_back(eigenmath::Vector3d(1, 0, 0));
  corners.emplace_back(eigenmath::Vector3d(2, 0, 0));
  corners.emplace_back(eigenmath::Vector3d(2, 1, 0));

  constexpr double kTranslationalRadius = 0.1;
  constexpr double kRotationalRadius = 0.1;
  PolyLineToBspline3Waypoints(corners, kTranslationalRadius, kRotationalRadius,
                              &control_points);

  ASSERT_EQ(7, control_points.size());
  EXPECT_THAT(control_points[0], IsApprox(corners[0]));
  EXPECT_THAT(control_points[3], IsApprox(corners[1]));
  EXPECT_THAT(control_points[6], IsApprox(corners[2]));

  eigenmath::Pose3d cp_1(eigenmath::Vector3d(1.1, 0, 0));
  EXPECT_THAT(control_points[1], IsApprox(cp_1));
  eigenmath::Pose3d cp_2(eigenmath::Vector3d(1.9, 0, 0));
  EXPECT_THAT(control_points[2], IsApprox(cp_2));
  eigenmath::Pose3d cp_4(eigenmath::Vector3d(2.0, 0.1, 0));
  EXPECT_THAT(control_points[4], IsApprox(cp_4));
  eigenmath::Pose3d cp_5(eigenmath::Vector3d(2.0, 0.9, 0));
  EXPECT_THAT(control_points[5], IsApprox(cp_5));
}

TEST(PolyLineToBspline3WaypointsPosesTest, Rotation) {
  std::vector<eigenmath::Pose3d> corners, control_points;
  corners.emplace_back(eigenmath::Quaterniond(Eigen::AngleAxis<double>(
      1.0, eigenmath::Vector3d(1, 2, 3).normalized())));
  corners.emplace_back(eigenmath::Quaterniond(Eigen::AngleAxis<double>(
      2.0, eigenmath::Vector3d(1, 2, 3).normalized())));
  corners.emplace_back(eigenmath::Quaterniond(Eigen::AngleAxis<double>(
      3.0, eigenmath::Vector3d(1, 2, 3).normalized())));

  constexpr double kTranslationalRadius = 0.1;
  constexpr double kRotationalRadius = 0.1;
  PolyLineToBspline3Waypoints(corners, kTranslationalRadius, kRotationalRadius,
                              &control_points);

  ASSERT_EQ(7, control_points.size());
  EXPECT_THAT(control_points[0], IsApprox(corners[0]));
  EXPECT_THAT(control_points[3], IsApprox(corners[1]));
  EXPECT_THAT(control_points[6], IsApprox(corners[2]));

  eigenmath::Pose3d cp_1(eigenmath::Quaterniond(Eigen::AngleAxis<double>(
      1.1, eigenmath::Vector3d(1, 2, 3).normalized())));
  EXPECT_THAT(control_points[1], IsApprox(cp_1));
  eigenmath::Pose3d cp_2(eigenmath::Quaterniond(Eigen::AngleAxis<double>(
      1.9, eigenmath::Vector3d(1, 2, 3).normalized())));
  EXPECT_THAT(control_points[2], IsApprox(cp_2));
  eigenmath::Pose3d cp_4(eigenmath::Quaterniond(Eigen::AngleAxis<double>(
      2.1, eigenmath::Vector3d(1, 2, 3).normalized())));
  EXPECT_THAT(control_points[4], IsApprox(cp_4));
  eigenmath::Pose3d cp_5(eigenmath::Quaterniond(Eigen::AngleAxis<double>(
      2.9, eigenmath::Vector3d(1, 2, 3).normalized())));
  EXPECT_THAT(control_points[5], IsApprox(cp_5));
}

TEST(PolyLineToBspline3WaypointsPosesTest, RadiusOutOfBounds) {
  std::vector<eigenmath::Pose3d> corners, control_points;
  corners.emplace_back(eigenmath::Vector3d(1, 0, 0));
  corners.emplace_back(eigenmath::Vector3d(2, 0, 0));

  constexpr double kTranslationalRadius = 0.6;
  constexpr double kRotationalRadius = 0.1;
  PolyLineToBspline3Waypoints(corners, kTranslationalRadius, kRotationalRadius,
                              &control_points);

  ASSERT_EQ(4, control_points.size());
  EXPECT_THAT(control_points[0], IsApprox(corners[0]));
  EXPECT_THAT(control_points[3], IsApprox(corners[1]));

  eigenmath::Pose3d cp_1(eigenmath::Vector3d(1.25, 0, 0));
  EXPECT_THAT(control_points[1], IsApprox(cp_1));
  eigenmath::Pose3d cp_2(eigenmath::Vector3d(1.75, 0, 0));
  EXPECT_THAT(control_points[2], IsApprox(cp_2));
}

TEST(PolyLineToBspline3WaypointsPosesTest, ZeroRadius) {
  std::vector<eigenmath::Pose3d> corners, control_points;
  corners.emplace_back(eigenmath::Vector3d(1, 0, 0));
  corners.emplace_back(eigenmath::Vector3d(2, 0, 0));

  constexpr double kTranslationalRadius = 0.0;
  constexpr double kRotationalRadius = 0.1;
  PolyLineToBspline3Waypoints(corners, kTranslationalRadius, kRotationalRadius,
                              &control_points);

  ASSERT_EQ(4, control_points.size());
  EXPECT_THAT(control_points[0], IsApprox(corners[0]));
  EXPECT_THAT(control_points[1], IsApprox(corners[0]));
  EXPECT_THAT(control_points[2], IsApprox(corners[1]));
  EXPECT_THAT(control_points[3], IsApprox(corners[1]));
}

TEST(PolyLineToBspline3Waypoints, PolyLineToBspline3WaypointsMaximumPathError) {
  constexpr int kDegree = 3;
  constexpr double kAcceptableDistanceError = 1.0e-10;
  for (double waypoint_distance = 0; waypoint_distance < 6;
       waypoint_distance += 0.5) {
    // Choose a path that is one dimensional, as this leads to the largest
    // path distance, since the `radius` parameter is directly applied only to
    // that dimension.
    std::vector<eigenmath::Vector3d> corners = {
        eigenmath::Vector3d(0, 0, 0),
        eigenmath::Vector3d(waypoint_distance, 0, 0),
        eigenmath::Vector3d(0, 0, 0),
    };
    for (double radius = 0; radius < waypoint_distance; radius += 0.1) {
      std::vector<eigenmath::Vector3d> control_points;
      PolyLineToBspline3Waypoints(corners, radius, &control_points);
      const int num_knots = BSpline3d::NumKnots(control_points.size(), kDegree);
      BSpline3d spline;
      ASSERT_OK(spline.Init(kDegree, num_knots));
      ASSERT_OK(spline.SetUniformKnotVector(num_knots));
      ASSERT_OK(spline.SetControlPoints(control_points));
      eigenmath::Vector3d point = eigenmath::Vector3d::Zero();
      ASSERT_OK(spline.EvalCurve(0.5, point));
      const double max_distance =
          (point - corners[1]).lpNorm<Eigen::Infinity>();
      EXPECT_NEAR(max_distance,
                  PolyLineToBspline3WaypointsMaximumPathError(
                      radius, waypoint_distance),
                  kAcceptableDistanceError)
          << "radius= " << radius
          << ", waypoint_distance= " << waypoint_distance;
    }
  }
}

}  // namespace
}  // namespace trajectory_planning

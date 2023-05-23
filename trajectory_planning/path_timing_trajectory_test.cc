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

#include "trajectory_planning/path_timing_trajectory.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <ios>
#include <iostream>
#include <iterator>
#include <memory>
#include <ostream>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "eigenmath/manifolds.h"
#include "eigenmath/matchers.h"
#include "eigenmath/types.h"
#include "eigenmath/utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "trajectory_planning/time.h"
#include "trajectory_planning/trajectory_test_utils.h"
#include "trajectory_planning/timeable_path_cartesian_spline.h"
#include "trajectory_planning/timeable_path_joint_spline.h"
#include "trajectory_planning/trajectory_buffer.h"

namespace trajectory_planning {
namespace {
using eigenmath::ExpSO3;
using eigenmath::LogSO3;
using eigenmath::LogSO3DerivativeManifold;
using eigenmath::MakeVector;
using eigenmath::Matrix3d;
using eigenmath::Matrix6Xd;
using eigenmath::Pose3d;
using eigenmath::Quaterniond;
using eigenmath::SO3d;
using eigenmath::Vector3d;
using eigenmath::VectorXd;
using eigenmath::testing::IsApprox;
using ::testing::ElementsAreArray;
using ::testing::Pointwise;

constexpr bool kVerboseDebugPrinting = true;
constexpr size_t kNDof = 3;
constexpr size_t kNumSamples = 1000;
constexpr absl::Duration kTimeStep = absl::Milliseconds(4);
constexpr absl::Duration kReplanInterval = absl::Milliseconds(200);
constexpr absl::Duration kHorizon = absl::Milliseconds(750);

#define CHECK_WITH_MSG(condition, error_message, args...) \
  char buffer[256];                                       \
  snprintf(buffer, 256, error_message, args);             \
  ABSL_CHECK_IMPL((condition), buffer);

#define ASSERT_OK_AND_ASSIGN(dest, call_result) \
  ASSERT_TRUE(call_result.ok());                \
  dest = *call_result;

#define ASSERT_OK(call_result) ASSERT_TRUE(call_result.ok());
#define EXPECT_OK(call_result) EXPECT_TRUE(call_result.ok());

MATCHER_P(StatusIs, s, "") { return arg.code() == s; }

MATCHER_P2(StatusIs, s, str_matcher, "") {
  return arg.code() == s && testing::ExplainMatchResult(
                                str_matcher, arg.message(), result_listener);
}

class PathTimingTrajectoryTest
    : public ::testing::TestWithParam<
          PathTimingTrajectoryOptions::TimeSamplingMethod> {
 protected:
};

TEST_P(PathTimingTrajectoryTest, PathErrors) {
  auto path_more_dofs = std::make_shared<TimeableJointSplinePath>(
      JointPathOptions().set_num_dofs(kNDof + 1).set_num_path_samples(
          kNumSamples));
  auto path_more_samples = std::make_shared<TimeableJointSplinePath>(
      JointPathOptions().set_num_dofs(kNDof).set_num_path_samples(kNumSamples +
                                                                  1));
  PathTimingTrajectory planner(PathTimingTrajectoryOptions()
                                   .SetTimeStep(kTimeStep)
                                   .SetNumDofs(kNDof)
                                   .SetNumPathSamples(kNumSamples));

  EXPECT_THAT(planner.SetPath(nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(planner.SetPath(path_more_dofs),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(planner.SetPath(path_more_samples),
              StatusIs(absl::StatusCode::kInvalidArgument));
  // No path set.
  EXPECT_THAT(planner.Plan(TimeFromSec(0), absl::Seconds(1)),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

// Verifies that adding waypoints -> calling Plan() works.
TEST_P(PathTimingTrajectoryTest, RestToRestPlanningWorks) {
  auto path = std::make_shared<TimeableJointSplinePath>(
      JointPathOptions().set_num_dofs(kNDof).set_num_path_samples(kNumSamples));
  PathTimingTrajectory planner(PathTimingTrajectoryOptions()
                                   .SetTimeStep(kTimeStep)
                                   .SetNumDofs(kNDof)
                                   .SetNumPathSamples(kNumSamples)
                                   .SetTimeSamplingMethod(GetParam()));
  VectorXd max_acceleration(kNDof);
  VectorXd max_velocity(kNDof);
  max_acceleration.setConstant(2.0);
  max_velocity.setConstant(1.0);

  ASSERT_OK(planner.SetPath(path));
  ASSERT_OK(path->SetWaypoints(
      {Vector3d(1, 2, 3), Vector3d(-1, -2, -3), Vector3d(1, 2, 3)}));
  ASSERT_OK(path->SetMaxJointVelocity(ToSpan(max_velocity)));
  ASSERT_OK(path->SetMaxJointAcceleration(ToSpan(max_acceleration)));

  // Call Plan() repeatedly while shifting the starting time forward.
  absl::Time start_time = TimeFromSec(0.0);
  for (int loop = 0; !planner.IsTrajectoryAtEnd(); loop++) {
    ASSERT_OK(planner.Plan(start_time, kHorizon));
    const int num_samples = planner.GetVelocities().size();
    ASSERT_EQ(num_samples, planner.GetPositions().size());
    ASSERT_EQ(num_samples, planner.GetAccelerations().size());
    const absl::Time next_start_time =
        std::min(planner.GetEndTime(), start_time + kReplanInterval);
    if (kVerboseDebugPrinting) {
      const double end_time_bound = planner.IsTrajectoryAtEnd()
                                        ? TimeToSec(planner.GetEndTime())
                                        : TimeToSec(next_start_time);
      printf(
          "TEST_sampling_method=%d start_time: %e end_time: %e target "
          "reached: %d decel time "
          "%e bound: %e\n",
          static_cast<int>(GetParam()), TimeToSec(planner.GetStartTime()),
          TimeToSec(planner.GetEndTime()), planner.IsTrajectoryAtEnd(),
          TimeToSec(planner.GetFinalDecelStart()), end_time_bound);

      for (int sample = 0; sample < planner.NumTimeSamples() &&
                           planner.GetTime()[sample] < end_time_bound;
           sample++) {
        printf("TRAJ-method-%d-%.3d  sample %d ", static_cast<int>(GetParam()),
               loop, sample);
        std::cout << " time: " << planner.GetTime()[sample];
        std::cout << " pos: " << planner.GetPositions()[sample].transpose();
        std::cout << " vel: " << planner.GetVelocities()[sample].transpose();
        std::cout << " acc: " << planner.GetAccelerations()[sample].transpose();
        std::cout << std::endl;
      }
    }
    start_time = next_start_time;
  }

  EXPECT_GT(planner.GetPositions().size(), 0);
  EXPECT_GT(planner.GetVelocities().size(), 0);
  EXPECT_THAT(planner.GetVelocities().back(),
              IsApprox(eigenmath::VectorXd::Zero(3)));
  EXPECT_THAT(planner.GetPositions().back(),
              IsApprox(path->GetWaypoints().back()));
}

TEST_P(PathTimingTrajectoryTest, NoDuplicateInitialSamples) {
  auto path = std::make_shared<TimeableJointSplinePath>(
      JointPathOptions().set_num_dofs(kNDof).set_num_path_samples(kNumSamples));
  PathTimingTrajectory planner(PathTimingTrajectoryOptions()
                                   .SetTimeStep(kTimeStep)
                                   .SetNumDofs(kNDof)
                                   .SetNumPathSamples(kNumSamples)
                                   .SetTimeSamplingMethod(GetParam()));
  const double kTimeEpsilon = 0.01 * absl::ToDoubleSeconds(kTimeStep);
  constexpr int kMinimumNumberOfSamples = 5;
  VectorXd max_acceleration(kNDof);
  VectorXd max_velocity(kNDof);
  max_acceleration.setConstant(2.0);
  max_velocity.setConstant(1.0);

  ASSERT_OK(planner.SetPath(path));
  ASSERT_OK(path->SetWaypoints(
      {Vector3d(1, 2, 3), Vector3d(-1, -2, -3), Vector3d(1, 2, 3)}));
  ASSERT_OK(path->SetMaxJointVelocity(ToSpan(max_velocity)));
  ASSERT_OK(path->SetMaxJointAcceleration(ToSpan(max_acceleration)));

  // Generate an initial plan.
  absl::Time start_time = TimeFromSec(0.0);
  ASSERT_OK(planner.Plan(start_time, kHorizon));
  EXPECT_EQ(start_time, planner.GetStartTime());
  ASSERT_GE(planner.GetTime().size(), kMinimumNumberOfSamples);
  EXPECT_DOUBLE_EQ(planner.GetTime().front(), TimeToSec(start_time));
  EXPECT_GE(planner.GetTime()[1], planner.GetTime()[0] + kTimeEpsilon);

  // Replan starting exactly at an existing sample, and fix that sample so it
  // is exactly representable by the int64/nanosecond absl::Time.
  for (const int sample : {0, 2}) {
    start_time = TimeFromSec(planner.GetTime()[sample]);
    std::vector<double> time_samples = planner.GetTime();
    time_samples[sample] = TimeToSec(start_time);
    planner.TestOnlySetTimeSamples(time_samples);
    ASSERT_EQ(TimeToSec(start_time), planner.GetTime()[sample]);
    ASSERT_OK(planner.Plan(start_time, kHorizon));
    EXPECT_EQ(start_time, planner.GetStartTime());
    ASSERT_GE(planner.GetTime().size(), kMinimumNumberOfSamples);
    EXPECT_DOUBLE_EQ(planner.GetTime().front(), TimeToSec(start_time));
    EXPECT_GE(planner.GetTime()[1], planner.GetTime()[0] + kTimeEpsilon)
        << "First two timesteps should differ significantly, but the "
           "difference= "
        << planner.GetTime()[1] - planner.GetTime()[0];
  }
  // Replan starting right after samples.
  for (const int sample : {0, 2}) {
    start_time = TimeFromSec(planner.GetTime()[sample]);
    std::vector<double> time_samples = planner.GetTime();
    time_samples[sample] = std::nexttoward(TimeToSec(start_time), 100.0);
    planner.TestOnlySetTimeSamples(time_samples);
    ASSERT_OK(planner.Plan(start_time, kHorizon));
    EXPECT_EQ(start_time, planner.GetStartTime());
    ASSERT_GE(planner.GetTime().size(), kMinimumNumberOfSamples);
    EXPECT_DOUBLE_EQ(planner.GetTime().front(), TimeToSec(start_time));
    EXPECT_GE(planner.GetTime()[1], planner.GetTime()[0] + kTimeEpsilon)
        << "First two timesteps should differ significantly, but the "
           "difference= "
        << planner.GetTime()[1] - planner.GetTime()[0];
  }

  // Replan starting right beofre samples.
  for (const int sample : {1, 3}) {
    start_time = TimeFromSec(planner.GetTime()[sample]);
    std::vector<double> time_samples = planner.GetTime();
    time_samples[sample] = std::nexttoward(TimeToSec(start_time), -100.0);
    planner.TestOnlySetTimeSamples(time_samples);
    ASSERT_OK(planner.Plan(start_time, kHorizon));
    EXPECT_EQ(start_time, planner.GetStartTime());
    ASSERT_GE(planner.GetTime().size(), kMinimumNumberOfSamples);
    EXPECT_DOUBLE_EQ(planner.GetTime().front(), TimeToSec(start_time));
    EXPECT_GE(planner.GetTime()[1], planner.GetTime()[0] + kTimeEpsilon)
        << "First two timesteps should differ significantly, but the "
           "difference= "
        << planner.GetTime()[1] - planner.GetTime()[0];
  }
}

TEST_P(PathTimingTrajectoryTest, IsInvariantToStartingTime) {
  auto path = std::make_shared<TimeableJointSplinePath>(
      JointPathOptions().set_num_dofs(kNDof).set_num_path_samples(kNumSamples));

  PathTimingTrajectory planner(PathTimingTrajectoryOptions()
                                   .SetTimeStep(kTimeStep)
                                   .SetNumDofs(kNDof)
                                   .SetNumPathSamples(kNumSamples)
                                   .SetTimeSamplingMethod(GetParam()));
  VectorXd max_acceleration(kNDof);
  VectorXd max_velocity(kNDof);
  max_acceleration.setConstant(2.0);
  max_velocity.setConstant(1.0);

  ASSERT_OK(planner.SetPath(path));
  ASSERT_OK(path->SetWaypoints(
      {Vector3d(1, 2, 3), Vector3d(-1, -2, -3), Vector3d(1, 2, 3)}));
  ASSERT_OK(path->SetMaxJointVelocity(ToSpan(max_velocity)));
  ASSERT_OK(path->SetMaxJointAcceleration(ToSpan(max_acceleration)));

  // Call Plan() starting at time 0.0.
  ASSERT_OK(planner.Plan(TimeFromSec(0.0), kHorizon));
  std::vector<eigenmath::VectorXd> positions_from_time_0 =
      planner.GetPositions();
  std::vector<eigenmath::VectorXd> velocities_from_time_0 =
      planner.GetVelocities();
  std::vector<eigenmath::VectorXd> accelerations_from_time_0 =
      planner.GetAccelerations();

  // Reset and plan from a different starting time.
  planner.Reset();
  ASSERT_OK(planner.SetPath(path));
  ASSERT_OK(path->SetWaypoints(
      {Vector3d(1, 2, 3), Vector3d(-1, -2, -3), Vector3d(1, 2, 3)}));
  ASSERT_OK(path->SetMaxJointVelocity(ToSpan(max_velocity)));
  ASSERT_OK(path->SetMaxJointAcceleration(ToSpan(max_acceleration)));

  ASSERT_OK(planner.Plan(TimeFromSec(42.0), kHorizon));

  EXPECT_THAT(planner.GetPositions(),
              Pointwise(eigenmath::testing::IsApproxTuple(1e-10),
                        positions_from_time_0));
}

TEST_P(PathTimingTrajectoryTest, SwitchToNewJointWaypointPathWorks) {
  auto path = std::make_shared<TimeableJointSplinePath>(
      JointPathOptions()
          .set_num_dofs(kNDof)
          .set_num_path_samples(kNumSamples)
          .set_delta_parameter(0.001));
  PathTimingTrajectory planner(PathTimingTrajectoryOptions()
                                   .SetTimeStep(kTimeStep)
                                   .SetNumDofs(kNDof)
                                   .SetNumPathSamples(kNumSamples)
                                   .SetTimeSamplingMethod(GetParam()));
  const std::array<VectorXd, 3> waypoints = {Vector3d(1.0, 2.0, 3.0),
                                             Vector3d(-1.0, -2.0, -3.0),
                                             Vector3d(0.5, 1.0, 1.5)};
  // The full trajectories, across switching boundaries.
  ASSERT_OK_AND_ASSIGN(auto trajectory, TrajectoryBuffer::Create());
  VectorXd max_acceleration(kNDof);
  VectorXd max_velocity(kNDof);
  constexpr double kMaxAcceleration = 2.0;
  constexpr double kMaxVelocity = 1.0;
  max_acceleration.setConstant(kMaxAcceleration);
  max_velocity.setConstant(kMaxVelocity);

  ASSERT_OK(planner.SetPath(path));
  ASSERT_OK(path->SetWaypoints(waypoints));
  ASSERT_OK(path->SetMaxJointVelocity(ToSpan(max_velocity)));
  ASSERT_OK(path->SetMaxJointAcceleration(ToSpan(max_acceleration)));

  absl::Time start_time = TimeFromSec(0.0);
  for (int loop = 0; !planner.IsTrajectoryAtEnd(); loop++) {
    ASSERT_OK(planner.Plan(start_time, kHorizon));
    const int num_samples = planner.GetVelocities().size();
    ASSERT_EQ(num_samples, planner.GetPositions().size());
    ASSERT_EQ(num_samples, planner.GetAccelerations().size());

    ASSERT_OK(trajectory->InsertSegment(
        planner.GetTime(), planner.GetPositions(), planner.GetVelocities(),
        planner.GetAccelerations()));

    const absl::Time next_start_time =
        planner.GetNextPlanStartTime(start_time + kReplanInterval);
    if (kVerboseDebugPrinting) {
      for (int sample = 0; sample < planner.NumTimeSamples(); sample++) {
        printf("FIRST-TRAJ-%.3d  sample %d ", loop, sample);
        std::cout << " time: " << planner.GetTime()[sample];
        std::cout << " pos: " << planner.GetPositions()[sample].transpose();
        std::cout << " vel: " << planner.GetVelocities()[sample].transpose();
        std::cout << " acc: " << planner.GetAccelerations()[sample].transpose();
        std::cout << std::endl;
      }
    }

    start_time = next_start_time;

    // Stop planning after reaching significant fraction of the maximum
    // velocity.
    if ((planner.GetVelocities().front().cwiseAbs() - max_velocity)
            .cwiseAbs()
            .minCoeff() < 0.3) {
      break;
    }
  }
  ASSERT_FALSE(planner.IsTrajectoryAtEnd());
  const std::array<VectorXd, 2> new_waypoints = {Vector3d(1.0, 2.0, 3.0),
                                                 Vector3d(0.5, 1.0, 5.5)};

  // Get stopping parameter.
  ASSERT_OK_AND_ASSIGN(const double stopping_path_parameter,
                       planner.GetPathStopParameter(start_time));
  // Modify existing path.
  ASSERT_OK(path->SwitchToWaypointPath(stopping_path_parameter, new_waypoints));
  // Set initial velocity such that the trajectory's velocity is continuous.
  ASSERT_OK_AND_ASSIGN(const VectorXd initial_velocity,
                       trajectory->GetVelocityAtTime(start_time));
  ASSERT_OK(path->SetInitialVelocity(initial_velocity));

  for (int loop = 0; !planner.IsTrajectoryAtEnd(); loop++) {
    ASSERT_OK(planner.Plan(start_time, kHorizon));
    const int num_samples = planner.GetVelocities().size();
    ASSERT_EQ(num_samples, planner.GetPositions().size());
    ASSERT_EQ(num_samples, planner.GetAccelerations().size());

    ASSERT_OK(trajectory->InsertSegment(
        planner.GetTime(), planner.GetPositions(), planner.GetVelocities(),
        planner.GetAccelerations()));

    const absl::Time next_start_time =
        planner.GetNextPlanStartTime(start_time + kReplanInterval);
    if (kVerboseDebugPrinting) {
      const double end_time_bound = planner.IsTrajectoryAtEnd()
                                        ? TimeToSec(planner.GetEndTime())
                                        : TimeToSec(next_start_time);
      for (int sample = 0; sample < planner.NumTimeSamples() &&
                           planner.GetTime()[sample] < end_time_bound;
           sample++) {
        printf("UPDATE-TRAJ-%.3d  sample %d ", loop, sample);
        std::cout << " time: " << planner.GetTime()[sample];
        std::cout << " pos: " << planner.GetPositions()[sample].transpose();
        std::cout << " vel: " << planner.GetVelocities()[sample].transpose();
        std::cout << " acc: " << planner.GetAccelerations()[sample].transpose();
        std::cout << std::endl;
      }
    }

    start_time = next_start_time;
  }

  EXPECT_THAT(planner.GetPositions().back(),
              IsApprox(new_waypoints.back(), 1e-10))
      << "Trajectory doesn't reach the path update's final waypoint.";
  EXPECT_THAT(planner.GetVelocities().back(),
              IsApprox(eigenmath::Vector3d::Zero(), 1e-10))
      << "Trajectory doesn't end with zero velocity.";

  // Expect finite differences of positions to match velocities. If this is
  // the case, the trajectory (including the stitching location), should be
  // smooth.
  // TODO Expect small errors everywhere after adjusting
  //   sampling strategy or (b/195265115) adding re-sampling error constraint.
  const int kFinalPartSamples = 20;
  const double kPathEndAcceptableVelocityFiniteDifferenceError = 1e-1;
  const double kAcceptableVelocityFiniteDifferenceError = 1e-2;
  absl::Span<const double> times_span = trajectory->GetTimes();
  absl::Span<const VectorXd> positions_span = trajectory->GetPositions();
  absl::Span<const VectorXd> velocities_span = trajectory->GetVelocities();
  // TODO Expect small errors everywhere after adjusting
  // sampling strategy.
  ExpectConsistentFiniteDifferenceDerivatives(
      times_span.subspan(0, times_span.size() - kFinalPartSamples),
      positions_span.subspan(0, positions_span.size() - kFinalPartSamples),
      velocities_span.subspan(0, velocities_span.size() - kFinalPartSamples),
      kAcceptableVelocityFiniteDifferenceError);
  ExpectConsistentFiniteDifferenceDerivatives(
      times_span.subspan(times_span.size() - kFinalPartSamples,
                         kFinalPartSamples),
      positions_span.subspan(positions_span.size() - kFinalPartSamples,
                             kFinalPartSamples),
      velocities_span.subspan(velocities_span.size() - kFinalPartSamples,
                              kFinalPartSamples),
      kPathEndAcceptableVelocityFiniteDifferenceError);
}

TEST_P(PathTimingTrajectoryTest, ResetAndReplan) {
  auto path = std::make_shared<TimeableJointSplinePath>(
      JointPathOptions().set_num_dofs(kNDof).set_num_path_samples(kNumSamples));
  PathTimingTrajectory planner(PathTimingTrajectoryOptions()
                                   .SetTimeStep(kTimeStep)
                                   .SetNumDofs(kNDof)
                                   .SetNumPathSamples(kNumSamples)
                                   .SetTimeSamplingMethod(GetParam()));
  const std::array<VectorXd, 3> waypoints = {Vector3d(1.0, 2.0, 3.0),
                                             Vector3d(-1.0, -2.0, -3.0),
                                             Vector3d(0.5, 1.0, 1.5)};
  ASSERT_OK_AND_ASSIGN(auto first_trajectory, TrajectoryBuffer::Create());
  VectorXd max_acceleration(kNDof);
  VectorXd max_velocity(kNDof);
  max_acceleration.setConstant(2.0);
  max_velocity.setConstant(1.0);

  ASSERT_OK(planner.SetPath(path));
  ASSERT_OK(path->SetWaypoints(waypoints));
  ASSERT_OK(path->SetMaxJointVelocity(ToSpan(max_velocity)));
  ASSERT_OK(path->SetMaxJointAcceleration(ToSpan(max_acceleration)));

  absl::Time start_time = TimeFromSec(0.0);
  for (int loop = 0; !planner.IsTrajectoryAtEnd(); loop++) {
    ASSERT_OK(planner.Plan(start_time, kHorizon));
    const int num_samples = planner.GetVelocities().size();
    ASSERT_EQ(num_samples, planner.GetPositions().size());
    ASSERT_EQ(num_samples, planner.GetAccelerations().size());

    // Cache the trajectory, so we can check that the second plan yields the
    // same result.
    ASSERT_OK(first_trajectory->InsertSegment(
        planner.GetTime(), planner.GetPositions(), planner.GetVelocities(),
        planner.GetAccelerations()));
    const absl::Time next_start_time =
        planner.GetNextPlanStartTime(start_time + kReplanInterval);
    if (kVerboseDebugPrinting) {
      const double end_time_bound = planner.IsTrajectoryAtEnd()
                                        ? TimeToSec(planner.GetEndTime())
                                        : TimeToSec(next_start_time);
      for (int sample = 0; sample < planner.NumTimeSamples() &&
                           planner.GetTime()[sample] < end_time_bound;
           sample++) {
        printf("FIRST-TRAJ-%.3d  sample %d ", loop, sample);
        std::cout << " time: " << planner.GetTime()[sample];
        std::cout << " pos: " << planner.GetPositions()[sample].transpose();
        std::cout << " vel: " << planner.GetVelocities()[sample].transpose();
        std::cout << " acc: " << planner.GetAccelerations()[sample].transpose();
        std::cout << std::endl;
      }
    }

    start_time = next_start_time;
  }

  // Add new waypoints and continue planning, calling  Reset() first.
  // Plan should start at the new initial waypoint.
  start_time = TimeFromSec(0.0);
  planner.Reset();
  ASSERT_OK(path->SetWaypoints(waypoints));
  // Call planner.
  ASSERT_OK_AND_ASSIGN(auto replanned_trajectory, TrajectoryBuffer::Create());

  for (int loop = 0; !planner.IsTrajectoryAtEnd(); loop++) {
    ASSERT_OK(planner.Plan(start_time, kHorizon));
    const absl::Time next_start_time =
        planner.GetNextPlanStartTime(start_time + kReplanInterval);

    int num_samples = planner.GetVelocities().size();
    ASSERT_EQ(num_samples, planner.GetPositions().size());
    ASSERT_EQ(num_samples, planner.GetAccelerations().size());
    ASSERT_OK(replanned_trajectory->InsertSegment(
        planner.GetTime(), planner.GetPositions(), planner.GetVelocities(),
        planner.GetAccelerations()));

    if (kVerboseDebugPrinting) {
      const double end_time_bound = planner.IsTrajectoryAtEnd()
                                        ? TimeToSec(planner.GetEndTime())
                                        : TimeToSec(next_start_time);
      for (int sample = 0; sample < planner.NumTimeSamples() &&
                           planner.GetTime()[sample] < end_time_bound;
           sample++) {
        printf("SECOND-TRAJ-%.3d  sample %d ", loop, sample);
        std::cout << " time: " << planner.GetTime()[sample];
        std::cout << " pos: " << planner.GetPositions()[sample].transpose();
        std::cout << " vel: " << planner.GetVelocities()[sample].transpose();
        std::cout << " acc: " << planner.GetAccelerations()[sample].transpose();
        std::cout << std::endl;
      }
    }

    start_time = next_start_time;
  }
  EXPECT_THAT(replanned_trajectory->GetPositions(),
              ElementsAreArray(first_trajectory->GetPositions()))
      << "Position trajectory produced for equal input after Reset is not "
         "equal.";
  EXPECT_THAT(replanned_trajectory->GetVelocities(),
              ElementsAreArray(first_trajectory->GetVelocities()))
      << "Velocity trajectory produced for equal input after Reset is not "
         "equal.";
  EXPECT_THAT(replanned_trajectory->GetAccelerations(),
              ElementsAreArray(first_trajectory->GetAccelerations()))
      << "Acceleration trajectory produced for equal input after Reset is "
         "not "
         "equal.";
}

// Fake kinematics for testing purposes:
// translation={joint[0],joint[1],joint[2]}
// quaternion = ExpSO3({joint[3],joint[4],joint[5]})
// For IK: joint[6] is copied from nullspace joint targets.
constexpr int kFakeDofs = 7;
Pose3d FakeFK(const VectorXd& joints) {
  return Pose3d(ExpSO3(Vector3d(joints(3), joints(4), joints(5))),
                Vector3d(joints(0), joints(1), joints(2)));
}

absl::Status FakePathIK(const eigenmath::VectorXd& initial_value,
                        const std::vector<Pose3d>& pose,
                        const std::vector<VectorXd>& joint,
                        std::vector<VectorXd>* solution) {
  CHECK(solution != nullptr);
  CHECK_WITH_MSG(pose.size() == joint.size(),
                 "pose.size(): %zu, joint.size() :%zu", pose.size(),
                 joint.size());
  solution->resize(pose.size());
  for (int idx = 0; idx < pose.size(); idx++) {
    (*solution)[idx].resize(kFakeDofs);
    CHECK(joint[idx].size() == kFakeDofs);
    (*solution)[idx].segment(0, 3) = pose[idx].translation();
    (*solution)[idx].segment(3, 3) = LogSO3(SO3d(pose[idx].quaternion()));
    (*solution)[idx][6] = joint[idx][6];
  }
  return absl::OkStatus();
}

absl::Status FakeJacobian(const VectorXd& joints, Matrix6Xd* jacobian) {
  CHECK(jacobian != nullptr);
  CHECK(jacobian->cols() == kFakeDofs);
  CHECK(joints.size() == kFakeDofs);

  Quaterniond quat = ExpSO3(Vector3d(joints.segment(3, 3))).quaternion();

  jacobian->setZero();
  jacobian->block(0, 0, 3, 3) = Matrix3d::Identity();
  jacobian->block(3, 3, 3, 4) = LogSO3DerivativeManifold(quat);

  return absl::OkStatus();
}

TEST(PathTimingTrajectoryTest, CartesianPathCornerRounding) {
  auto path = std::make_shared<TimeableCartesianSplinePath>(
      CartesianPathOptions()
          .set_path_ik_func(FakePathIK)
          .set_jacobian_func(FakeJacobian)
          .set_num_dofs(kFakeDofs)
          .set_num_path_samples(kNumSamples));

  double translation_rounding = 0.05;
  double rotation_rounding = 0.1;
  EXPECT_OK(path->SetRotationRounding(rotation_rounding));
  EXPECT_OK(path->SetTranslationRounding(translation_rounding));
  EXPECT_EQ(path->GetRotationRounding(), rotation_rounding);
  EXPECT_EQ(path->GetTranslationRounding(), translation_rounding);

  EXPECT_THAT(path->SetRotationRounding(0.0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(path->SetTranslationRounding(0.0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(path->SetRotationRounding(-1.0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(path->SetTranslationRounding(-1.0),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_P(PathTimingTrajectoryTest, SimpleCartesianPath) {
  auto path = std::make_shared<TimeableCartesianSplinePath>(
      CartesianPathOptions()
          .set_path_ik_func(FakePathIK)
          .set_jacobian_func(FakeJacobian)
          .set_num_dofs(kFakeDofs)
          .set_num_path_samples(kNumSamples));

  PathTimingTrajectory planner(PathTimingTrajectoryOptions()
                                   .SetTimeStep(kTimeStep)
                                   .SetNumDofs(kFakeDofs)
                                   .SetNumPathSamples(kNumSamples)
                                   .SetTimeSamplingMethod(GetParam()));
  VectorXd max_joint_acceleration = VectorXd::Constant(kFakeDofs, 2.0);
  VectorXd max_joint_velocity = VectorXd::Constant(kFakeDofs, 1.0);

  ASSERT_OK(planner.SetPath(path));

  std::vector<VectorXd> joint_waypoints = {
      MakeVector({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
      MakeVector({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0})};
  std::vector<Pose3d> pose_waypoints;
  pose_waypoints.reserve(joint_waypoints.size());
  for (const auto& wp : joint_waypoints) {
    pose_waypoints.emplace_back(FakeFK(wp));
  }

  ASSERT_OK(path->SetWaypoints(pose_waypoints, joint_waypoints));
  ASSERT_OK(path->SetMaxJointVelocity(ToSpan(max_joint_velocity)));
  ASSERT_OK(path->SetMaxJointAcceleration(ToSpan(max_joint_acceleration)));
  ASSERT_OK(path->SetMaxCartesianVelocity(0.5, 0.25));

  // Call Plan() repeatedly while shifting the starting time forward.
  absl::Time start_time = TimeFromSec(0.0);
  for (int loop = 0; !planner.IsTrajectoryAtEnd(); loop++) {
    ASSERT_OK(planner.Plan(start_time, kHorizon));
    const int num_samples = planner.GetVelocities().size();
    ASSERT_EQ(num_samples, planner.GetPositions().size());
    ASSERT_EQ(num_samples, planner.GetAccelerations().size());
    const absl::Time next_start_time =
        std::min(planner.GetEndTime(), start_time + kReplanInterval);
    if (kVerboseDebugPrinting) {
      const double end_time_bound = planner.IsTrajectoryAtEnd()
                                        ? TimeToSec(planner.GetEndTime())
                                        : TimeToSec(next_start_time);
      printf(
          "TEST: start_time: %e end_time: %e target reached: %d decel time "
          "%e "
          "bound: %e\n",
          TimeToSec(planner.GetStartTime()), TimeToSec(planner.GetEndTime()),
          planner.IsTrajectoryAtEnd(), TimeToSec(planner.GetFinalDecelStart()),
          end_time_bound);

      for (int sample = 0; sample < planner.NumTimeSamples() &&
                           planner.GetTime()[sample] < end_time_bound;
           sample++) {
        printf("TRAJ-%.3d  sample %d ", loop, sample);
        std::cout << " time: " << planner.GetTime()[sample];
        std::cout << " pos: " << planner.GetPositions()[sample].transpose();
        std::cout << " vel: " << planner.GetVelocities()[sample].transpose();
        std::cout << " acc: " << planner.GetAccelerations()[sample].transpose();
        std::cout << std::endl;
      }
    }
    start_time = next_start_time;
  }

  EXPECT_EQ(planner.GetPositions().size(), 1);
  EXPECT_EQ(planner.GetVelocities().size(), 1);
  EXPECT_THAT(planner.GetPositions().front(),
              IsApprox(path->GetJointWaypoints().front()));
  EXPECT_THAT(planner.GetVelocities().back(),
              IsApprox(eigenmath::VectorXd::Zero(kFakeDofs)));
  EXPECT_THAT(planner.GetVelocities().front(),
              IsApprox(eigenmath::VectorXd::Zero(kFakeDofs)));
  EXPECT_THAT(planner.GetPositions().back(),
              IsApprox(path->GetJointWaypoints().back()));
}

TEST_P(PathTimingTrajectoryTest, ZeroLengthCartesianPathWorks) {
  auto path = std::make_shared<TimeableCartesianSplinePath>(
      CartesianPathOptions()
          .set_path_ik_func(FakePathIK)
          .set_jacobian_func(FakeJacobian)
          .set_num_dofs(kFakeDofs)
          .set_num_path_samples(kNumSamples));

  PathTimingTrajectory planner(PathTimingTrajectoryOptions()
                                   .SetTimeStep(kTimeStep)
                                   .SetNumDofs(kFakeDofs)
                                   .SetNumPathSamples(kNumSamples)
                                   .SetTimeSamplingMethod(GetParam()));
  VectorXd max_joint_acceleration = VectorXd::Constant(kFakeDofs, 2.0);
  VectorXd max_joint_velocity = VectorXd::Constant(kFakeDofs, 1.0);

  ASSERT_OK(planner.SetPath(path));

  std::vector<VectorXd> joint_waypoints = {
      MakeVector({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
      MakeVector({0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
      MakeVector({0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}),
      MakeVector({0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0}),
      MakeVector({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0})};
  std::vector<Pose3d> pose_waypoints;
  pose_waypoints.reserve(joint_waypoints.size());
  for (const auto& wp : joint_waypoints) {
    pose_waypoints.emplace_back(FakeFK(wp));
  }

  ASSERT_OK(path->SetWaypoints(pose_waypoints, joint_waypoints));
  ASSERT_OK(path->SetMaxJointVelocity(ToSpan(max_joint_velocity)));
  ASSERT_OK(path->SetMaxJointAcceleration(ToSpan(max_joint_acceleration)));
  ASSERT_OK(path->SetMaxCartesianVelocity(0.5, 0.25));

  // Call Plan() repeatedly while shifting the starting time forward.
  absl::Time start_time = TimeFromSec(0.0);
  for (int loop = 0; !planner.IsTrajectoryAtEnd(); loop++) {
    ASSERT_OK(planner.Plan(start_time, kHorizon));
    const int num_samples = planner.GetVelocities().size();
    ASSERT_EQ(num_samples, planner.GetPositions().size());
    ASSERT_EQ(num_samples, planner.GetAccelerations().size());
    const absl::Time next_start_time =
        std::min(planner.GetEndTime(), start_time + kReplanInterval);
    if (kVerboseDebugPrinting) {
      const double end_time_bound = planner.IsTrajectoryAtEnd()
                                        ? TimeToSec(planner.GetEndTime())
                                        : TimeToSec(next_start_time);
      printf(
          "TEST: start_time: %e end_time: %e target reached: %d decel time "
          "%e "
          "bound: %e\n",
          TimeToSec(planner.GetStartTime()), TimeToSec(planner.GetEndTime()),
          planner.IsTrajectoryAtEnd(), TimeToSec(planner.GetFinalDecelStart()),
          end_time_bound);

      for (int sample = 0; sample < planner.NumTimeSamples() &&
                           planner.GetTime()[sample] < end_time_bound;
           sample++) {
        printf("TRAJ-%.3d  sample %d ", loop, sample);
        std::cout << " time: " << planner.GetTime()[sample];
        std::cout << " pos: " << planner.GetPositions()[sample].transpose();
        std::cout << " vel: " << planner.GetVelocities()[sample].transpose();
        std::cout << " acc: " << planner.GetAccelerations()[sample].transpose();
        std::cout << std::endl;
      }
    }
    start_time = next_start_time;
  }
}

TEST_P(PathTimingTrajectoryTest, SwitchToNewCartesianWaypointPathWorks) {
  auto path = std::make_shared<TimeableCartesianSplinePath>(
      CartesianPathOptions()
          .set_path_ik_func(FakePathIK)
          .set_jacobian_func(FakeJacobian)
          .set_num_dofs(kFakeDofs)
          .set_num_path_samples(kNumSamples)
          .set_delta_parameter(0.005));
  // The full position & velocity trajectories, across switching  boundaries.
  // Uses a btree_map so the values are sorted by the key (milliseconds).
  ASSERT_OK_AND_ASSIGN(auto trajectory, TrajectoryBuffer::Create());
  VectorXd max_acceleration(kFakeDofs);
  constexpr double kMaxJointAcceleration = 2.0;
  constexpr double kMaxJointVelocity = 1.0;
  constexpr double kMaxTranslationalCartesianVelocity = 0.5;
  constexpr double kMaxRotationalCartesianVelocity = 0.25;
  PathTimingTrajectory planner(PathTimingTrajectoryOptions()
                                   .SetTimeStep(kTimeStep)
                                   .SetNumDofs(kFakeDofs)
                                   .SetNumPathSamples(kNumSamples)
                                   .SetTimeSamplingMethod(GetParam()));
  const VectorXd max_joint_acceleration =
      VectorXd::Constant(kFakeDofs, kMaxJointAcceleration);
  const VectorXd max_joint_velocity =
      VectorXd::Constant(kFakeDofs, kMaxJointVelocity);

  ASSERT_OK(planner.SetPath(path));

  const std::vector<VectorXd> joint_waypoints = {
      MakeVector({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
      MakeVector({0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
      MakeVector({0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}),
      MakeVector({0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0}),
      MakeVector({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0})};
  std::vector<Pose3d> pose_waypoints;
  pose_waypoints.reserve(joint_waypoints.size());
  for (const auto& wp : joint_waypoints) {
    pose_waypoints.emplace_back(FakeFK(wp));
  }

  ASSERT_OK(path->SetWaypoints(pose_waypoints, joint_waypoints));
  ASSERT_OK(path->SetMaxJointVelocity(ToSpan(max_joint_velocity)));
  ASSERT_OK(path->SetMaxJointAcceleration(ToSpan(max_joint_acceleration)));
  ASSERT_OK(path->SetMaxCartesianVelocity(kMaxTranslationalCartesianVelocity,
                                          kMaxRotationalCartesianVelocity));

  // Call Plan() repeatedly while shifting the starting time forward.
  absl::Time start_time = TimeFromSec(0.0);
  for (int loop = 0; !planner.IsTrajectoryAtEnd(); loop++) {
    ASSERT_OK(planner.Plan(start_time, kHorizon));
    const int num_samples = planner.GetVelocities().size();
    ASSERT_EQ(num_samples, planner.GetPositions().size());
    ASSERT_EQ(num_samples, planner.GetAccelerations().size());

    ASSERT_OK(trajectory->InsertSegment(
        planner.GetTime(), planner.GetPositions(), planner.GetVelocities(),
        planner.GetAccelerations()));

    const absl::Time next_start_time =
        planner.GetNextPlanStartTime(start_time + kReplanInterval);
    if (kVerboseDebugPrinting) {
      for (int sample = 0; sample < planner.NumTimeSamples(); sample++) {
        printf("FIRST-TRAJ-%.3d  sample %d ", loop, sample);
        std::cout << " time: " << planner.GetTime()[sample];
        std::cout << " pos: " << planner.GetPositions()[sample].transpose();
        std::cout << " vel: " << planner.GetVelocities()[sample].transpose();
        std::cout << "acc : " << planner.GetAccelerations()[sample].transpose();
        std::cout << std::endl;
      }
    }

    start_time = next_start_time;

    // Stop planning after reaching significant fraction of the maximum
    // velocity.
    const double max_abs_initial_velocity =
        planner.GetVelocities().front().cwiseAbs().maxCoeff();
    constexpr double kFractionToStopAt = 0.3;
    if (max_abs_initial_velocity > kMaxJointVelocity * kFractionToStopAt ||
        max_abs_initial_velocity >
            kMaxTranslationalCartesianVelocity * kFractionToStopAt ||
        max_abs_initial_velocity >
            kMaxRotationalCartesianVelocity * kFractionToStopAt) {
      break;
    }
  }
  ASSERT_FALSE(planner.IsTrajectoryAtEnd());

  const std::vector<VectorXd> new_joint_waypoints = {
      MakeVector({-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0}),
      MakeVector({1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0})};
  std::vector<Pose3d> new_pose_waypoints;
  pose_waypoints.reserve(new_joint_waypoints.size());
  for (const auto& wp : new_joint_waypoints) {
    new_pose_waypoints.emplace_back(FakeFK(wp));
  }

  // Get stopping parameter.
  ASSERT_OK_AND_ASSIGN(const double stopping_path_parameter,
                       planner.GetPathStopParameter(start_time));
  // Modify existing path.
  ASSERT_OK(path->SwitchToWaypointPath(
      stopping_path_parameter, new_pose_waypoints, new_joint_waypoints));
  // Set initial velocity such that the trajectory's velocity is continuous.
  ASSERT_OK_AND_ASSIGN(const VectorXd initial_velocity,
                       trajectory->GetVelocityAtTime(start_time));
  ASSERT_OK(path->SetInitialVelocity(initial_velocity));

  for (int loop = 0; !planner.IsTrajectoryAtEnd(); loop++) {
    ASSERT_OK(planner.Plan(start_time, kHorizon));
    const int num_samples = planner.GetVelocities().size();
    ASSERT_EQ(num_samples, planner.GetPositions().size());
    ASSERT_EQ(num_samples, planner.GetAccelerations().size());

    ASSERT_OK(trajectory->InsertSegment(
        planner.GetTime(), planner.GetPositions(), planner.GetVelocities(),
        planner.GetAccelerations()));

    const absl::Time next_start_time =
        planner.GetNextPlanStartTime(start_time + kReplanInterval);
    if (kVerboseDebugPrinting) {
      const double end_time_bound = planner.IsTrajectoryAtEnd()
                                        ? TimeToSec(planner.GetEndTime())
                                        : TimeToSec(next_start_time);
      for (int sample = 0; sample < planner.NumTimeSamples() &&
                           planner.GetTime()[sample] < end_time_bound;
           sample++) {
        printf("UPDATE-TRAJ-%.3d  sample %d ", loop, sample);
        std::cout << " time: " << planner.GetTime()[sample];
        std::cout << " pos: " << planner.GetPositions()[sample].transpose();
        std::cout << " vel: " << planner.GetVelocities()[sample].transpose();
        std::cout << " acc: " << planner.GetAccelerations()[sample].transpose();
        std::cout << std::endl;
      }
    }

    start_time = next_start_time;
  }

  EXPECT_THAT(planner.GetPositions().back(),
              IsApprox(new_joint_waypoints.back(), 1e-10))
      << "Trajectory doesn't reach the path update's final waypoint.";
  EXPECT_THAT(planner.GetVelocities().back(),
              IsApprox(eigenmath::VectorXd::Zero(kFakeDofs), 1e-10))
      << "Trajectory doesn't end with zero velocity.";

  // Expect finite differences of positions to match velocities. If this is
  // the case, the trajectory (including the stitching location), should be
  // smooth.
  // TODO Expect small errors everywhere after adjusting
  //   sampling strategy or (b/195265115) adding re-sampling error constraint.
  const int kFinalPartSamples = 50;
  const double kAcceptableVelocityFiniteDifferenceError = 1e-2;
  const double kPathEndAcceptableVelocityFiniteDifferenceError = 1e-1;

  absl::Span<const double> times_span = trajectory->GetTimes();
  absl::Span<const VectorXd> positions_span = trajectory->GetPositions();
  absl::Span<const VectorXd> velocities_span = trajectory->GetVelocities();
  // TODO Expect small errors everywhere after adjusting
  // sampling strategy.
  ExpectConsistentFiniteDifferenceDerivatives(
      times_span.subspan(0, times_span.size() - kFinalPartSamples),
      positions_span.subspan(0, positions_span.size() - kFinalPartSamples),
      velocities_span.subspan(0, velocities_span.size() - kFinalPartSamples),
      kAcceptableVelocityFiniteDifferenceError);
  ExpectConsistentFiniteDifferenceDerivatives(
      times_span.subspan(times_span.size() - kFinalPartSamples,
                         kFinalPartSamples),
      positions_span.subspan(positions_span.size() - kFinalPartSamples,
                             kFinalPartSamples),
      velocities_span.subspan(velocities_span.size() - kFinalPartSamples,
                              kFinalPartSamples),
      kPathEndAcceptableVelocityFiniteDifferenceError);
}

// Minimal test of stopping parameter calculation.
// TODO: Add a test based on the analytical solution in a simple
// linear case.
TEST_P(PathTimingTrajectoryTest, GetPathStopParameterWorksInSimpleCase) {
  constexpr int kNumSamples = 500;
  constexpr absl::Duration kTimeStep = absl::Milliseconds(1);

  auto path = std::make_shared<TimeableJointSplinePath>(
      JointPathOptions()
          .set_num_dofs(kNDof)
          .set_constraint_safety(1.0)
          .set_num_path_samples(kNumSamples));

  PathTimingTrajectory planner(PathTimingTrajectoryOptions()
                                   .SetTimeStep(kTimeStep)
                                   .SetNumDofs(kNDof)
                                   .SetNumPathSamples(kNumSamples)
                                   .SetTimeSamplingMethod(GetParam()));
  VectorXd max_acceleration(kNDof);
  VectorXd max_velocity(kNDof);
  max_acceleration.setConstant(2.0);
  max_velocity.setConstant(1.0);
  // Use collinear points to simplify stop calculation.
  ASSERT_OK(planner.SetPath(path));
  ASSERT_OK(path->SetWaypoints(
      {Vector3d(0, 0, 0), Vector3d(1, 1, 1), Vector3d(2, 2, 2)}));
  ASSERT_OK(path->SetMaxJointVelocity(ToSpan(max_velocity)));
  ASSERT_OK(path->SetMaxJointAcceleration(ToSpan(max_acceleration)));

  // Generate an initial plan.
  absl::Time start_time = TimeFromSec(0.0);
  ASSERT_OK(planner.Plan(start_time, kHorizon));

  // Starting from the first sample, we should be able to stop immediately.
  ASSERT_OK_AND_ASSIGN(double stop_parameter,
                       planner.GetPathStopParameter(start_time));

  auto parameter_samples = planner.GetPathParameters();
  auto parameter_it = absl::c_upper_bound(parameter_samples, stop_parameter);
  int stop_index = std::distance(parameter_samples.begin(), parameter_it);
  // The first sample is not modified, the second sample will get zero
  // velocity, and upper_bound with resampling of the path parameter increases
  // this to 3.
  EXPECT_LE(stop_index, 3);

  // When starting from the last sample, the parameter should equal the final
  // value.
  double start_time_sec = planner.GetTime().back();
  start_time = TimeFromSec(start_time_sec);
  ASSERT_OK_AND_ASSIGN(stop_parameter,
                       planner.GetPathStopParameter(start_time));

  parameter_samples = planner.GetPathParameters();
  parameter_it = absl::c_upper_bound(parameter_samples, stop_parameter);
  EXPECT_EQ(parameter_it, parameter_samples.end());
  EXPECT_DOUBLE_EQ(stop_parameter, *std::prev(parameter_samples.end()));
}

INSTANTIATE_TEST_SUITE_P(
    PathTimingTrajectoryTestSuite, PathTimingTrajectoryTest,
    ::testing::Values(
        PathTimingTrajectoryOptions::TimeSamplingMethod::kUniformlyInTime,
        PathTimingTrajectoryOptions::TimeSamplingMethod::
            kSkipSamplesCloserThanTimeStep));

}  // namespace
}  // namespace trajectory_planning

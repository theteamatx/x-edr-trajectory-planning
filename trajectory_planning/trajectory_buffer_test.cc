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

#include "trajectory_planning/trajectory_buffer.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include "absl/time/time.h"
#include "eigenmath/matchers.h"
#include "eigenmath/types.h"
#include "trajectory_planning/sampled_trajectory.h"
#include "trajectory_planning/time.h"
#include "trajectory_planning/trajectory_test_utils.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace trajectory_planning {
namespace {
using ::eigenmath::VectorXd;
using ::eigenmath::testing::IsApprox;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::HasSubstr;

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

constexpr int kNumJoints = 9;
static constexpr double kSampleTimeSec = 8e-3;

// A simple quadratic test trajectory "x  = 1/2*(t-T)^2", with T the time of the
// final sample.
SampledTrajectory GetQuadraticTestTrajectory(
    const int sample_count, const double sample_time_sec = kSampleTimeSec,
    const int num_joints = kNumJoints) {
  const double final_time = (sample_count - 1) * sample_time_sec;

  SampledTrajectory trajectory;
  for (int i = 0; i < sample_count; ++i) {
    const double t = sample_time_sec * i;
    const double trel = t - final_time;
    trajectory.times.push_back(t);
    trajectory.positions.push_back(
        VectorXd::Constant(num_joints, 0.5 * trel * trel));
    trajectory.velocities.push_back(VectorXd::Constant(num_joints, trel));
    trajectory.accelerations.push_back(VectorXd::Constant(num_joints, 1.0));
  }

  return trajectory;
}

// A basic test trajectory.
SampledTrajectory GetTestTrajectory(
    const double time_offset, const int sample_count,
    const double sample_time_sec = kSampleTimeSec,
    const int num_joints = kNumJoints) {
  SampledTrajectory trajectory;
  trajectory.times.reserve(sample_count);
  trajectory.positions.reserve(sample_count);
  trajectory.velocities.reserve(sample_count);
  trajectory.accelerations.reserve(sample_count);
  for (int i = 0; i < sample_count; ++i) {
    trajectory.times.push_back(i * sample_time_sec + time_offset);
    trajectory.positions.push_back(VectorXd::Constant(num_joints, i));
    trajectory.velocities.push_back(VectorXd::Constant(num_joints, 10 * i));
    trajectory.accelerations.push_back(VectorXd::Constant(num_joints, 100 * i));
  }

  return trajectory;
}

TEST(TrajectoryBuffer, CreateChecksOptions) {
  EXPECT_THAT(TrajectoryBuffer::Create({.timestep_tolerance = -1}).status(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("not positive")));
}

TEST(TrajectoryBuffer, InsertSegmentWorks) {
  constexpr double kTimeOffsetSec = 1.0;

  ASSERT_OK_AND_ASSIGN(auto buffer, TrajectoryBuffer::Create());
  EXPECT_EQ(buffer->GetNumSamples(), 0);
  EXPECT_EQ(buffer->GetStartTime(), TimeFromSec(0));
  EXPECT_EQ(buffer->GetSequenceNumber(), 0);

  // Empty append updates sequence number, but doesn't do anything else.
  EXPECT_OK(buffer->InsertSegment({}, {}, {}, {}));
  EXPECT_EQ(buffer->GetStartTime(), TimeFromSec(0));
  EXPECT_EQ(buffer->GetEndTime(), TimeFromSec(0));
  EXPECT_EQ(buffer->GetSequenceNumber(), 1);
  EXPECT_EQ(buffer->GetNumSamples(), 0);

  buffer->Clear();

  // Generate some samples to append.
  SampledTrajectory trajectory = GetTestTrajectory(1.0, 10);
  // Append the first segment, which should become the whole trajectory.
  EXPECT_OK(buffer->InsertSegment(trajectory.times, trajectory.positions,
                                  trajectory.velocities,
                                  trajectory.accelerations));
  EXPECT_EQ(buffer->GetStartTime(),
            TimeFromSec(trajectory.times.front()));
  EXPECT_EQ(buffer->GetEndTime(),
            TimeFromSec(trajectory.times.back()));
  EXPECT_EQ(buffer->GetSequenceNumber(), 0);
  EXPECT_EQ(buffer->GetNumSamples(), trajectory.positions.size());

  EXPECT_THAT(buffer->GetTimes(), ElementsAreArray(trajectory.times));
  EXPECT_THAT(buffer->GetPositions(), ElementsAreArray(trajectory.positions));
  EXPECT_THAT(buffer->GetVelocities(), ElementsAreArray(trajectory.velocities));
  EXPECT_THAT(buffer->GetAccelerations(),
              ElementsAreArray(trajectory.accelerations));

  // Append another trajectory segment starting at 3*sample_time.
  trajectory = GetTestTrajectory(kTimeOffsetSec + 3 * kSampleTimeSec, 5);

  // Append after start_time + 2*sample_time, which should replace samples
  // starting with the fourth sample.
  EXPECT_OK(buffer->InsertSegment(trajectory.times, trajectory.positions,
                                  trajectory.velocities,
                                  trajectory.accelerations));
  EXPECT_EQ(buffer->GetStartTime(), TimeFromSec(1));
  EXPECT_EQ(buffer->GetEndTime(),
            TimeFromSec(trajectory.times.back()));
  EXPECT_EQ(buffer->GetSequenceNumber(), 1);
  EXPECT_EQ(buffer->GetNumSamples(), 3 + trajectory.positions.size());

  EXPECT_EQ(buffer->GetTimes().size(), buffer->GetPositions().size());
  EXPECT_EQ(buffer->GetTimes().size(), buffer->GetVelocities().size());
  EXPECT_EQ(buffer->GetTimes().size(), buffer->GetAccelerations().size());

  EXPECT_THAT(buffer->GetTimes(),
              ElementsAre(kTimeOffsetSec + 0 * kSampleTimeSec,
                          kTimeOffsetSec + 1 * kSampleTimeSec,
                          kTimeOffsetSec + 2 * kSampleTimeSec,
                          kTimeOffsetSec + 3 * kSampleTimeSec,
                          kTimeOffsetSec + 4 * kSampleTimeSec,
                          kTimeOffsetSec + 5 * kSampleTimeSec,
                          kTimeOffsetSec + 6 * kSampleTimeSec,
                          kTimeOffsetSec + 7 * kSampleTimeSec));

  EXPECT_THAT(
      buffer->GetPositions(),
      ElementsAre(
          VectorXd::Constant(kNumJoints, 0), VectorXd::Constant(kNumJoints, 1),
          VectorXd::Constant(kNumJoints, 2), VectorXd::Constant(kNumJoints, 0),
          VectorXd::Constant(kNumJoints, 1), VectorXd::Constant(kNumJoints, 2),
          VectorXd::Constant(kNumJoints, 3),
          VectorXd::Constant(kNumJoints, 4)));
  EXPECT_THAT(buffer->GetVelocities(),
              ElementsAre(VectorXd::Constant(kNumJoints, 10 * 0),
                          VectorXd::Constant(kNumJoints, 10 * 1),
                          VectorXd::Constant(kNumJoints, 10 * 2),
                          VectorXd::Constant(kNumJoints, 10 * 0),
                          VectorXd::Constant(kNumJoints, 10 * 1),
                          VectorXd::Constant(kNumJoints, 10 * 2),
                          VectorXd::Constant(kNumJoints, 10 * 3),
                          VectorXd::Constant(kNumJoints, 10 * 4)));
  EXPECT_THAT(buffer->GetAccelerations(),
              ElementsAre(VectorXd::Constant(kNumJoints, 100 * 0),
                          VectorXd::Constant(kNumJoints, 100 * 1),
                          VectorXd::Constant(kNumJoints, 100 * 2),
                          VectorXd::Constant(kNumJoints, 100 * 0),
                          VectorXd::Constant(kNumJoints, 100 * 1),
                          VectorXd::Constant(kNumJoints, 100 * 2),
                          VectorXd::Constant(kNumJoints, 100 * 3),
                          VectorXd::Constant(kNumJoints, 100 * 4)));
}

TEST(TrajectoryBuffer, AppendSampleWorks) {
  ASSERT_OK_AND_ASSIGN(auto buffer, TrajectoryBuffer::Create());

  EXPECT_OK(buffer->AppendSample(1.0, VectorXd::Constant(kNumJoints, 1),
                                 VectorXd::Constant(kNumJoints, 2),
                                 VectorXd::Constant(kNumJoints, 3)));

  // Time stamp must be strictly increasing.
  EXPECT_THAT(buffer->AppendSample(1.0, VectorXd::Constant(kNumJoints, 1),
                                   VectorXd::Constant(kNumJoints, 2),
                                   VectorXd::Constant(kNumJoints, 3)),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("time")));
  EXPECT_THAT(buffer->AppendSample(-1.0, VectorXd::Constant(kNumJoints, 1),
                                   VectorXd::Constant(kNumJoints, 2),
                                   VectorXd::Constant(kNumJoints, 3)),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("time")));

  EXPECT_OK(buffer->AppendSample(1.1, VectorXd::Constant(kNumJoints, 1.1),
                                 VectorXd::Constant(kNumJoints, 2.1),
                                 VectorXd::Constant(kNumJoints, 3.1)));

  ASSERT_EQ(buffer->GetNumSamples(), 2);
  EXPECT_THAT(buffer->GetPositions(),
              ElementsAre(VectorXd::Constant(kNumJoints, 1),
                          VectorXd::Constant(kNumJoints, 1.1)));
  EXPECT_THAT(buffer->GetVelocities(),
              ElementsAre(VectorXd::Constant(kNumJoints, 2),
                          VectorXd::Constant(kNumJoints, 2.1)));
  EXPECT_THAT(buffer->GetAccelerations(),
              ElementsAre(VectorXd::Constant(kNumJoints, 3),
                          VectorXd::Constant(kNumJoints, 3.1)));
}

TEST(TrajectoryBuffer, InsertSegmentUsesTimestepTolerance) {
  constexpr double kTimeOffsetSec = 1.0;
  constexpr double kTimestepTolerance = 1e-5;
  ASSERT_OK_AND_ASSIGN(
      auto buffer,
      TrajectoryBuffer::Create({.timestep_tolerance = kTimestepTolerance}));

  // Generate some samples to append.
  SampledTrajectory trajectory1 = GetTestTrajectory(1.0, 10);
  // Append the first segment, which should become the whole trajectory.
  EXPECT_OK(buffer->InsertSegment(trajectory1.times, trajectory1.positions,
                                  trajectory1.velocities,
                                  trajectory1.accelerations));

  // Append another trajectory segment starting at 3*sample_time+eps.
  SampledTrajectory trajectory2 =
      GetTestTrajectory(kTimeOffsetSec + 3 * kSampleTimeSec, 5);
  trajectory2.times.front() -= 0.5 * kTimestepTolerance;
  // Append after start_time + 2*sample_time, which should replace samples
  // starting with the fourth sample.
  EXPECT_OK(buffer->InsertSegment(trajectory2.times, trajectory2.positions,
                                  trajectory2.velocities,
                                  trajectory2.accelerations));

  EXPECT_THAT(buffer->GetTimes(),
              ElementsAre(kTimeOffsetSec + 0 * kSampleTimeSec,
                          kTimeOffsetSec + 1 * kSampleTimeSec,
                          kTimeOffsetSec + 2 * kSampleTimeSec,
                          kTimeOffsetSec + 3 * kSampleTimeSec -
                              0.5 * kTimestepTolerance,
                          kTimeOffsetSec + 4 * kSampleTimeSec,
                          kTimeOffsetSec + 5 * kSampleTimeSec,
                          kTimeOffsetSec + 6 * kSampleTimeSec,
                          kTimeOffsetSec + 7 * kSampleTimeSec));

  buffer->Clear();
  EXPECT_OK(buffer->InsertSegment(trajectory1.times, trajectory1.positions,
                                  trajectory1.velocities,
                                  trajectory1.accelerations));
  trajectory2 = GetTestTrajectory(kTimeOffsetSec + 3 * kSampleTimeSec, 5);
  trajectory2.times.front() += 0.5 * kTimestepTolerance;

  // Append after start_time + 2*sample_time, which should replace samples
  // starting with the fourth sample.
  EXPECT_OK(buffer->InsertSegment(trajectory2.times, trajectory2.positions,
                                  trajectory2.velocities,
                                  trajectory2.accelerations));

  EXPECT_THAT(buffer->GetTimes(),
              ElementsAre(kTimeOffsetSec + 0 * kSampleTimeSec,
                          kTimeOffsetSec + 1 * kSampleTimeSec,
                          kTimeOffsetSec + 2 * kSampleTimeSec,
                          kTimeOffsetSec + 3 * kSampleTimeSec +
                              0.5 * kTimestepTolerance,
                          kTimeOffsetSec + 4 * kSampleTimeSec,
                          kTimeOffsetSec + 5 * kSampleTimeSec,
                          kTimeOffsetSec + 6 * kSampleTimeSec,
                          kTimeOffsetSec + 7 * kSampleTimeSec));
}

TEST(TrajectoryBuffer, InsertSegmentFailsForInvalidArguments) {
  ASSERT_OK_AND_ASSIGN(auto buffer, TrajectoryBuffer::Create());

  SampledTrajectory trajectory = GetTestTrajectory(1.0, 10);

  // Appending fails if vector sizes are wrong.
  const std::vector<double> time_wrong_size(trajectory.times.size() / 2);
  const std::vector<eigenmath::VectorXd> position_wrong_size(
      trajectory.times.size() / 2);
  EXPECT_THAT(
      buffer->InsertSegment(time_wrong_size, trajectory.positions,
                            trajectory.velocities, trajectory.accelerations),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      buffer->InsertSegment(trajectory.times, position_wrong_size,
                            trajectory.velocities, trajectory.accelerations),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      buffer->InsertSegment(trajectory.times, trajectory.positions,
                            position_wrong_size, trajectory.accelerations),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(buffer->InsertSegment(trajectory.times, trajectory.positions,
                                    trajectory.velocities, position_wrong_size),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Appending at the final time step should work.
  trajectory.times.front() = TimeToSec(buffer->GetEndTime());
  for (int i = 1; i < trajectory.times.size(); ++i) {
    trajectory.times[i] = trajectory.times[i - 1] + kSampleTimeSec;
  }
  EXPECT_OK(buffer->InsertSegment(trajectory.times, trajectory.positions,
                                  trajectory.velocities,
                                  trajectory.accelerations));
}

TEST(TrajectoryBuffer, ClearWorks) {
  ASSERT_OK_AND_ASSIGN(auto buffer, TrajectoryBuffer::Create());

  SampledTrajectory trajectory = GetTestTrajectory(1.0, 10);

  // Append a first segment, so argument checking for following appends can be
  // tested.
  EXPECT_OK(buffer->InsertSegment(trajectory.times, trajectory.positions,
                                  trajectory.velocities,
                                  trajectory.accelerations));
  buffer->Clear();
  EXPECT_EQ(buffer->GetSequenceNumber(), 0);
  EXPECT_EQ(buffer->GetNumSamples(), 0);
  EXPECT_EQ(buffer->GetStartTime(), TimeFromSec(0));
}

TEST(TrajectoryBuffer, DiscardWorks) {
  ASSERT_OK_AND_ASSIGN(auto buffer, TrajectoryBuffer::Create());

  // Discard on an empty trajectory always works and doesn't do anything.
  buffer->DiscardSegmentBefore(TimeFromSec(10));
  EXPECT_EQ(buffer->GetSequenceNumber(), 0);
  EXPECT_EQ(buffer->GetNumSamples(), 0);
  buffer->DiscardSegmentBefore(TimeFromSec(-10));
  EXPECT_EQ(buffer->GetSequenceNumber(), 0);
  EXPECT_EQ(buffer->GetNumSamples(), 0);

  // Append a trajectory, to test discard.
  SampledTrajectory trajectory = GetTestTrajectory(1.0, 10);
  EXPECT_OK(buffer->InsertSegment(trajectory.times, trajectory.positions,
                                  trajectory.velocities,
                                  trajectory.accelerations));
  EXPECT_EQ(buffer->GetStartTime(),
            TimeFromSec(trajectory.times.front()));
  EXPECT_EQ(buffer->GetEndTime(),
            TimeFromSec(trajectory.times.back()));
  EXPECT_EQ(buffer->GetNumSamples(), trajectory.positions.size());

  // Discarding samples before start_time does nothing.
  buffer->DiscardSegmentBefore(buffer->GetStartTime());
  EXPECT_EQ(buffer->GetStartTime(),
            TimeFromSec(trajectory.times.front()));
  EXPECT_EQ(buffer->GetEndTime(),
            TimeFromSec(trajectory.times.back()));
  EXPECT_EQ(buffer->GetNumSamples(), trajectory.positions.size());
  EXPECT_EQ(buffer->GetPositions().size(), buffer->GetNumSamples());
  EXPECT_EQ(buffer->GetVelocities().size(), buffer->GetNumSamples());
  EXPECT_EQ(buffer->GetAccelerations().size(), buffer->GetNumSamples());
  EXPECT_EQ(buffer->GetTimes().size(), buffer->GetNumSamples());

  // Discarding samples before time, for time in [start_time, end_time], removes
  // samples before time, but does not discard the sample at time.
  buffer->DiscardSegmentBefore(TimeFromSec(trajectory.times[4]));
  EXPECT_EQ(buffer->GetStartTime(), TimeFromSec(trajectory.times[4]));
  EXPECT_EQ(buffer->GetTimes().front(), trajectory.times[4]);
  EXPECT_EQ(buffer->GetEndTime(),
            TimeFromSec(trajectory.times.back()));
  EXPECT_EQ(buffer->GetNumSamples(), trajectory.times.size() - 4);
  EXPECT_EQ(buffer->GetPositions().size(), buffer->GetNumSamples());
  EXPECT_EQ(buffer->GetVelocities().size(), buffer->GetNumSamples());
  EXPECT_EQ(buffer->GetAccelerations().size(), buffer->GetNumSamples());
  EXPECT_EQ(buffer->GetTimes().size(), buffer->GetNumSamples());

  // Discarding samples before end_time + epsilon is the same as Clear().
  buffer->DiscardSegmentBefore(
      TimeFromSec(trajectory.times.back() + 1.0 / NSECS_PER_SEC));
  EXPECT_EQ(buffer->GetStartTime(), TimeFromSec(0));
  EXPECT_EQ(buffer->GetEndTime(), TimeFromSec(0));
  EXPECT_EQ(buffer->GetNumSamples(), 0);
  EXPECT_EQ(buffer->GetPositions().size(), buffer->GetNumSamples());
  EXPECT_EQ(buffer->GetVelocities().size(), buffer->GetNumSamples());
  EXPECT_EQ(buffer->GetAccelerations().size(), buffer->GetNumSamples());
  EXPECT_EQ(buffer->GetTimes().size(), buffer->GetNumSamples());

  // DiscardSegmentBefore at exactly a sample.
  constexpr double kEpsilon = 1e-10;
  constexpr double kPositionEpsilon = kEpsilon * 100.0;
  buffer->Clear();
  EXPECT_OK(buffer->InsertSegment(trajectory.times, trajectory.positions,
                                  trajectory.velocities,
                                  trajectory.accelerations));
  buffer->DiscardSegmentBefore(trajectory.times[3]);
  EXPECT_NEAR(buffer->GetTimes().front(), trajectory.times[3], kEpsilon);
  EXPECT_THAT(buffer->GetPositions().front(),
              IsApprox(trajectory.positions[3], kPositionEpsilon))
      << "diff= " << buffer->GetPositions().front() - trajectory.positions[3];
  for (int i = 1; i < buffer->GetNumSamples(); ++i) {
    EXPECT_NEAR(buffer->GetTimes()[i] - buffer->GetTimes()[i - 1],
                kSampleTimeSec, kEpsilon);
  }

  // Discard before just before a sample.
  buffer->Clear();
  EXPECT_OK(buffer->InsertSegment(trajectory.times, trajectory.positions,
                                  trajectory.velocities,
                                  trajectory.accelerations));
  const double time_just_before_t6 = std::nextafter(trajectory.times[6], -1e99);
  buffer->DiscardSegmentBefore(time_just_before_t6);
  EXPECT_NEAR(buffer->GetTimes().front(), time_just_before_t6, kEpsilon);
  EXPECT_THAT(buffer->GetPositions().front(),
              IsApprox(trajectory.positions[6], kPositionEpsilon))
      << "diff= " << buffer->GetPositions().front() - trajectory.positions[6];
  for (int i = 1; i < buffer->GetNumSamples(); ++i) {
    EXPECT_NEAR(buffer->GetTimes()[i] - buffer->GetTimes()[i - 1],
                kSampleTimeSec, kEpsilon);
  }

  // Discard before just after a sample.
  buffer->Clear();
  EXPECT_OK(buffer->InsertSegment(trajectory.times, trajectory.positions,
                                  trajectory.velocities,
                                  trajectory.accelerations));
  const double time_just_after_t6 = std::nextafter(trajectory.times[6], 1e99);

  buffer->DiscardSegmentBefore(time_just_after_t6);
  EXPECT_NEAR(buffer->GetTimes().front(), time_just_after_t6, kEpsilon);
  EXPECT_THAT(buffer->GetPositions().front(),
              IsApprox(trajectory.positions[6], kPositionEpsilon))
      << "diff= " << buffer->GetPositions().front() - trajectory.positions[6];

  for (int i = 1; i < buffer->GetNumSamples(); ++i) {
    EXPECT_NEAR(buffer->GetTimes()[i] - buffer->GetTimes()[i - 1],
                kSampleTimeSec, kEpsilon);
  }

  // DiscardBefore somewhere between samples.
  buffer->Clear();
  EXPECT_OK(buffer->InsertSegment(trajectory.times, trajectory.positions,
                                  trajectory.velocities,
                                  trajectory.accelerations));

  const double time_between_samples =
      0.5 * (trajectory.times[5] + trajectory.times[6]);

  buffer->DiscardSegmentBefore(time_between_samples);
  EXPECT_NEAR(buffer->GetTimes().front(), time_between_samples, kEpsilon);
  // Position shouldn't match a sample, it should be interpolated.
  EXPECT_THAT(buffer->GetPositions().front(),
              ::testing::Not(IsApprox(trajectory.positions[6], kEpsilon)));
  EXPECT_THAT(buffer->GetPositions().front(),
              ::testing::Not(IsApprox(trajectory.positions[5], kEpsilon)));

  for (int i = 1; i < buffer->GetNumSamples(); ++i) {
    EXPECT_LE(buffer->GetTimes()[i] - buffer->GetTimes()[i - 1],
              kSampleTimeSec + 2 * kEpsilon);
  }
}

TEST(TrajectorBuffer, GetPositionsUpToTime) {
  ASSERT_OK_AND_ASSIGN(auto buffer, TrajectoryBuffer::Create());
  std::vector<VectorXd> positions;
  SampledTrajectory trajectory = GetTestTrajectory(1.0, 5);

  ASSERT_OK(buffer->InsertSegment(trajectory.times, trajectory.positions,
                                  trajectory.velocities,
                                  trajectory.accelerations));

  // A time outside the range should return an empty span.
  EXPECT_EQ(buffer->GetPositionsUpToTime(TimeFromSec(10)).size(), 0);
  EXPECT_EQ(buffer->GetPositionsUpToTime(TimeFromSec(-1)).size(), 0);

  // Exactly at a sample: should exclude the sample.
  auto span =
      buffer->GetPositionsUpToTime(TimeFromSec(trajectory.times[2]));
  EXPECT_EQ(span.size(), 2);
  EXPECT_THAT(span.back(), IsApprox(trajectory.positions[1], 1e-10));
  // Just before a sample: should be same as at sample.
  span = buffer->GetPositionsUpToTime(TimeFromSec(
      std::nexttoward(trajectory.times[2], trajectory.times[1])));
  EXPECT_EQ(span.size(), 1);
  EXPECT_THAT(span.back(), IsApprox(trajectory.positions[0], 1e-10));
  // Just after a sample: should include the sample.
  span = buffer->GetPositionsUpToTime(TimeFromSec(
      std::nexttoward(trajectory.times[2], trajectory.times[3])));
  EXPECT_EQ(span.size(), 2);
  EXPECT_THAT(span.back(), IsApprox(trajectory.positions[1], 1e-10));
}

TEST(TrajectorBuffer, GetPositionAtTime) {
  ASSERT_OK_AND_ASSIGN(auto buffer, TrajectoryBuffer::Create());
  std::vector<VectorXd> positions;
  SampledTrajectory trajectory = GetTestTrajectory(1.0, 5);

  ASSERT_OK(buffer->InsertSegment(trajectory.times, trajectory.positions,
                                  trajectory.velocities,
                                  trajectory.accelerations));

  // Accessing position & velocity at time samples should return what was
  // appended above.
  for (int i = 0; i < trajectory.times.size(); ++i) {
    ASSERT_OK_AND_ASSIGN(
        const VectorXd& position,
        buffer->GetPositionAtTime(TimeFromSec(trajectory.times[i])));

    EXPECT_THAT(position, IsApprox(trajectory.positions[i], 1e-10))
        << " i= " << i << " trajectory size: " << trajectory.times.size();

    ASSERT_OK_AND_ASSIGN(
        const VectorXd& velocity,
        buffer->GetVelocityAtTime(TimeFromSec(trajectory.times[i])));
    EXPECT_THAT(velocity, IsApprox(trajectory.velocities[i], 1e-10))
        << " i= " << i << " trajectory size: " << trajectory.times.size();
  }
}

TEST(TrajectorBufferTest, StopAtIndexSucceedsIfFeasible) {
  constexpr double kDurationSec = 1.0;
  constexpr double kSampleTimeSec = 1e-3;
  constexpr int kNumSamples = kDurationSec / (kSampleTimeSec) + 1;
  const SampledTrajectory trajectory =
      GetQuadraticTestTrajectory(kNumSamples, kSampleTimeSec);
  EXPECT_THAT(trajectory.velocities.back(),
              IsApprox(eigenmath::VectorXd::Zero(kNumJoints)));

  ASSERT_OK_AND_ASSIGN(auto buffer, TrajectoryBuffer::Create());
  ASSERT_OK(buffer->InsertSegment(trajectory.times, trajectory.positions,
                                  trajectory.velocities,
                                  trajectory.accelerations));

  constexpr int kStopIndex = kNumSamples / 2;
  const eigenmath::VectorXd max_acceleration =
      eigenmath::VectorXd::Constant(kNumJoints, 5.0);
  EXPECT_OK(buffer->StopAtIndex(kStopIndex, max_acceleration, kSampleTimeSec));

  EXPECT_GE(buffer->GetNumSamples(), kStopIndex + 1);
  EXPECT_THAT(buffer->GetVelocities().back(),
              IsApprox(eigenmath::VectorXd::Zero(kNumJoints)));
  constexpr double kEpsilon = 1e-8;
  for (const auto& acceleration : buffer->GetAccelerations()) {
    EXPECT_TRUE(
        (acceleration.array().abs() <= max_acceleration.array() + kEpsilon)
            .all())
        << "acceleration= " << acceleration.transpose()
        << " max_acceleration= " << max_acceleration.transpose();
  }
  // The difference betweeen the resampled velocity and the finite difference
  // velocity of the resampled positions largely depends on the maximum rescaled
  // timestep in the stopping trajectory, which is rather large for the final
  // samples.
  constexpr double kAcceptableDifference = 5e-2;
  ExpectConsistentFiniteDifferenceDerivatives(
      buffer->GetTimes(), buffer->GetPositions(), buffer->GetVelocities(),
      kAcceptableDifference);
}

TEST(TrajectorBufferTest, DetectsErrors) {
  ASSERT_OK_AND_ASSIGN(auto buffer, TrajectoryBuffer::Create());
  constexpr int kSampleCount = 5;
  const SampledTrajectory trajectory = GetTestTrajectory(1.0, kSampleCount);
  const eigenmath::VectorXd max_acceleration =
      eigenmath::VectorXd::Constant(kNumJoints, 4.0);
  EXPECT_THAT(buffer->StopAtIndex(-1, max_acceleration, kSampleTimeSec),
              StatusIs(absl::StatusCode::kOutOfRange));
  EXPECT_THAT(buffer->StopAtIndex(0, max_acceleration, kSampleTimeSec),
              StatusIs(absl::StatusCode::kOutOfRange));

  ASSERT_OK(buffer->InsertSegment(trajectory.times, trajectory.positions,
                                  trajectory.velocities,
                                  trajectory.accelerations));

  EXPECT_THAT(
      buffer->StopAtIndex(kSampleCount, max_acceleration, kSampleTimeSec),
      StatusIs(absl::StatusCode::kOutOfRange));

  EXPECT_THAT(buffer->StopAtIndex(kSampleCount / 2, max_acceleration, -0.1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(TrajectorBufferTest, StopAtIndexFailsIfInfeasible) {
  ASSERT_OK_AND_ASSIGN(auto buffer, TrajectoryBuffer::Create());
  constexpr double kDurationSec = 1.0;
  constexpr double kSampleTimeSec = 1e-3;
  constexpr int kNumSamples = kDurationSec / (kSampleTimeSec) + 1;
  const SampledTrajectory trajectory =
      GetQuadraticTestTrajectory(kNumSamples, kSampleTimeSec);

  ASSERT_OK(buffer->InsertSegment(trajectory.times, trajectory.positions,
                                  trajectory.velocities,
                                  trajectory.accelerations));

  constexpr int kStopIndex = 2;
  const eigenmath::VectorXd max_acceleration =
      eigenmath::VectorXd::Constant(kNumJoints, 1.0);
  EXPECT_THAT(buffer->StopAtIndex(kStopIndex, max_acceleration, kSampleTimeSec),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(TrajectorBufferTest, StopBeforeTimeFailsIfInfeasible) {
  ASSERT_OK_AND_ASSIGN(auto buffer, TrajectoryBuffer::Create());
  constexpr double kDurationSec = 1.0;
  constexpr double kSampleTimeSec = 1e-3;
  constexpr int kNumSamples = kDurationSec / (kSampleTimeSec) + 1;
  const SampledTrajectory trajectory =
      GetQuadraticTestTrajectory(kNumSamples, kSampleTimeSec);

  ASSERT_OK(buffer->InsertSegment(trajectory.times, trajectory.positions,
                                  trajectory.velocities,
                                  trajectory.accelerations));

  const eigenmath::VectorXd max_acceleration =
      eigenmath::VectorXd::Constant(kNumJoints, 1.0);
  EXPECT_THAT(buffer->StopBeforeTime(kSampleTimeSec * 2, max_acceleration,
                                     kSampleTimeSec),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(buffer->StopBeforeTime(kSampleTimeSec * 3.1415, max_acceleration,
                                     kSampleTimeSec),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(TrajectorBufferTest, StopBeforeTimeSucceedsIfFeasible) {
  constexpr double kDurationSec = 1.0;
  constexpr double kSampleTimeSec = 1e-3;
  constexpr int kNumSamples = kDurationSec / (kSampleTimeSec) + 1;
  const SampledTrajectory trajectory =
      GetQuadraticTestTrajectory(kNumSamples, kSampleTimeSec);
  EXPECT_THAT(trajectory.velocities.back(),
              IsApprox(eigenmath::VectorXd::Zero(kNumJoints)));

  ASSERT_OK_AND_ASSIGN(auto buffer, TrajectoryBuffer::Create());
  ASSERT_OK(buffer->InsertSegment(trajectory.times, trajectory.positions,
                                  trajectory.velocities,
                                  trajectory.accelerations));

  // Stop exactly at a sample.
  constexpr int kStopIndex = kNumSamples / 2;
  const eigenmath::VectorXd max_acceleration =
      eigenmath::VectorXd::Constant(kNumJoints, 5.0);
  EXPECT_OK(buffer->StopBeforeTime(buffer->GetTimes()[kStopIndex],
                                   max_acceleration, kSampleTimeSec));

  EXPECT_GE(buffer->GetNumSamples(), kStopIndex + 1);
  EXPECT_THAT(buffer->GetVelocities().back(),
              IsApprox(eigenmath::VectorXd::Zero(kNumJoints)));
  constexpr double kEpsilon = 1e-8;
  for (const auto& acceleration : buffer->GetAccelerations()) {
    EXPECT_TRUE(
        (acceleration.array().abs() <= max_acceleration.array() + kEpsilon)
            .all())
        << "acceleration= " << acceleration.transpose()
        << " max_acceleration= " << max_acceleration.transpose();
  }
  // The difference betweeen the resampled velocity and the finite difference
  // velocity of the resampled positions largely depends on the maximum rescaled
  // timestep in the stopping trajectory, which is rather large for the final
  // samples.
  constexpr double kAcceptableDifference = 5e-2;
  ExpectConsistentFiniteDifferenceDerivatives(
      buffer->GetTimes(), buffer->GetPositions(), buffer->GetVelocities(),
      kAcceptableDifference);

  // StopBefore with a time halfway between two samples.
  buffer->Clear();
  ASSERT_OK(buffer->InsertSegment(trajectory.times, trajectory.positions,
                                  trajectory.velocities,
                                  trajectory.accelerations));
  const double stop_before_time = 0.5 * (buffer->GetTimes()[kStopIndex] +
                                         buffer->GetTimes()[kStopIndex + 1]);
  EXPECT_OK(buffer->StopBeforeTime(stop_before_time, max_acceleration,
                                   kSampleTimeSec));
  EXPECT_GE(buffer->GetNumSamples(), kStopIndex + 1);
}

TEST(TrajectorBufferTest,
     StopBeforeTimeSucceedsIfCutoffTimeBeyondFinalTimestep) {
  ASSERT_OK_AND_ASSIGN(auto buffer, TrajectoryBuffer::Create());
  constexpr int kSampleCount = 50;
  constexpr double kTimeStep = 0.01;
  constexpr int kDofs = 3;
  constexpr double kVelocity = 1.0;
  std::vector<double> times(kSampleCount);
  std::vector<eigenmath::VectorXd> positions(kSampleCount);
  std::vector<eigenmath::VectorXd> velocities(kSampleCount);
  std::vector<eigenmath::VectorXd> accelerations(kSampleCount);

  for (int i = 0; i < kSampleCount; ++i) {
    times[i] = i * kTimeStep;
    positions[i].setConstant(kDofs, kVelocity * kTimeStep * i);
    velocities[i].setConstant(kDofs, kVelocity);
    accelerations[i].setZero(kDofs);
  }

  ASSERT_OK(buffer->InsertSegment(times, positions, velocities, accelerations));

  EXPECT_OK(buffer->StopBeforeTime(times.back() + 12.0,
                                   eigenmath::VectorXd::Constant(kDofs, 2.0),
                                   kTimeStep));
  EXPECT_THAT(buffer->GetVelocities().back(),
              IsApprox(eigenmath::VectorXd::Zero(kDofs)));
}

}  // namespace
}  // namespace trajectory_planning

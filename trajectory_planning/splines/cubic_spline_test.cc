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

#include "trajectory_planning/splines/cubic_spline.h"

#include <random>
#include <valarray>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "gtest/gtest.h"
#include "trajectory_planning/splines/finite_difference_test_utils.h"

ABSL_FLAG(int32_t, verbosity, 0, "verbosity level");

namespace trajectory_planning {
namespace {

constexpr double kTiny = 1e-10;
// Check if:
// - functions catch errors
// - if code intended for real-time use allocates
TEST(CubicSpline, InputAndAllocation) {
  {  // valid case
    CubicSpline spline;
    ASSERT_TRUE(spline.Init(10));
  }
  {  // too few knots
    CubicSpline spline;
    ASSERT_FALSE(spline.Init(1));
  }
  {  // too many knots
    CubicSpline spline;
    ASSERT_FALSE(spline.Init(1000000000));
  }
  {  // set points function
    CubicSpline spline;
    std::valarray<double> knots = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::valarray<double> knots_small = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::valarray<double> knots_large = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    ASSERT_TRUE(spline.Init(knots.size()));

    std::valarray<double> points(spline.NumPoints());
    std::valarray<double> points_large(spline.NumPoints() + 1);
    std::valarray<double> points_small(spline.NumPoints() - 1);

    ASSERT_FALSE(spline.SetKnotVector(knots_small));
    ASSERT_FALSE(spline.SetKnotVector(knots_large));
    ASSERT_TRUE(spline.SetKnotVector(knots));
    ASSERT_FALSE(spline.SetControlPoints(points_small));
    ASSERT_FALSE(spline.SetControlPoints(points_large));
    ASSERT_TRUE(spline.SetControlPoints(points));
    ASSERT_TRUE(spline.CalculateParameters());
    double val, dval, ddval;
    for (double u = knots[0]; u <= knots[knots.size() - 1]; u += 0.1) {
      ASSERT_TRUE(spline.EvalCurve(u, &val));
      ASSERT_TRUE(spline.EvalCurveAndDerivatives(u, &val, &dval, &ddval));
    }
  }
}

TEST(CubicSpline, ParameterRange) {
  // umin,umax matching knot vector
  {
    // set points function
    CubicSpline spline;
    std::valarray<double> knots = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    ASSERT_TRUE(spline.Init(knots.size()));
    std::valarray<double> points(spline.NumPoints());
    double val, dval, ddval;

    ASSERT_TRUE(spline.SetKnotVector(knots));
    ASSERT_TRUE(spline.SetControlPoints(points));
    ASSERT_TRUE(spline.CalculateParameters());

    ASSERT_TRUE(spline.EvalCurve(0.5, &val));
    ASSERT_TRUE(spline.EvalCurveAndDerivatives(0.5, &val, &dval, &ddval));

    ASSERT_TRUE(spline.EvalCurve(0.0, &val));
    ASSERT_TRUE(spline.EvalCurveAndDerivatives(0.0, &val, &dval, &ddval));

    ASSERT_TRUE(spline.EvalCurve(9.0, &val));
    ASSERT_TRUE(spline.EvalCurveAndDerivatives(9.0, &val, &dval, &ddval));

    ASSERT_FALSE(spline.EvalCurve(-0.1, &val));
    ASSERT_FALSE(spline.EvalCurveAndDerivatives(-0.1, &val, &dval, &ddval));

    ASSERT_FALSE(spline.EvalCurve(9.1, &val));
    ASSERT_FALSE(spline.EvalCurveAndDerivatives(9.1, &val, &dval, &ddval));
  }

  // umin,umax restricting parameter range
  {
    // set points function
    CubicSpline spline;
    std::valarray<double> knots = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    ASSERT_TRUE(spline.Init(knots.size()));
    std::valarray<double> points(spline.NumPoints());
    double val, dval, ddval;

    ASSERT_TRUE(spline.SetKnotVector(knots, 0.1, 8.9));
    ASSERT_TRUE(spline.SetControlPoints(points));
    ASSERT_TRUE(spline.CalculateParameters());

    ASSERT_TRUE(spline.EvalCurve(0.5, &val));
    ASSERT_TRUE(spline.EvalCurveAndDerivatives(0.5, &val, &dval, &ddval));

    ASSERT_TRUE(spline.EvalCurve(0.1, &val));
    ASSERT_TRUE(spline.EvalCurveAndDerivatives(0.1, &val, &dval, &ddval));

    ASSERT_TRUE(spline.EvalCurve(8.9, &val));
    ASSERT_TRUE(spline.EvalCurveAndDerivatives(8.9, &val, &dval, &ddval));

    ASSERT_FALSE(spline.EvalCurve(0, &val));
    ASSERT_FALSE(spline.EvalCurveAndDerivatives(-0.1, &val, &dval, &ddval));

    ASSERT_FALSE(spline.EvalCurve(9, &val));
    ASSERT_FALSE(spline.EvalCurveAndDerivatives(9.1, &val, &dval, &ddval));
  }
}

// test if spline interpolation works
TEST(CubicSpline, Interpolation) {
  constexpr size_t kNumPoints = 10;
  CubicSpline spline;
  ASSERT_TRUE(spline.Init(kNumPoints));
  std::valarray<double> knots(kNumPoints);
  std::valarray<double> points(kNumPoints);

  std::uniform_real_distribution<double> dtdist(0.01, 1);
  std::uniform_real_distribution<double> pdist(-10, 10);
  std::default_random_engine engine;

  for (size_t idx = 0; idx < kNumPoints; idx++) {
    points[idx] = pdist(engine);
  }

  knots[0] = pdist(engine);
  for (size_t idx = 1; idx < kNumPoints; idx++) {
    knots[idx] = knots[idx - 1] + dtdist(engine);
  }

  ASSERT_TRUE(spline.SetKnotVector(knots));
  ASSERT_TRUE(spline.SetControlPoints(points));
  ASSERT_TRUE(spline.CalculateParameters());
  double val, dval, ddval;

  for (size_t idx = 0; idx < knots.size(); idx++) {
    ASSERT_TRUE(spline.EvalCurve(knots[idx], &val));
    ASSERT_GE(kTiny, val - points[idx]);
    ASSERT_TRUE(
        spline.EvalCurveAndDerivatives(knots[idx], &val, &dval, &ddval));
    ASSERT_GE(kTiny, std::fabs(val - points[idx]));
  }
}

// test if derivatives and continuity are as expected
TEST(CubicSpline, Derivatives) {
  constexpr size_t kNumPoints = 10;
  constexpr double kDt = 1e-6;
  constexpr double kMaxVelError = 1e-7;  // tied to kDt and distributions
  constexpr double kMaxAccError = 1e-4;  // tied to kDt and distributions
  CubicSpline spline;
  ASSERT_TRUE(spline.Init(kNumPoints));
  std::valarray<double> knots(kNumPoints);
  std::valarray<double> points(kNumPoints);

  std::uniform_real_distribution<double> dtdist(0.1, 1);
  std::uniform_real_distribution<double> pdist(-1, 1);
  std::default_random_engine engine;

  for (size_t idx = 0; idx < kNumPoints; idx++) {
    points[idx] = pdist(engine);
  }

  knots[0] = pdist(engine);
  for (size_t idx = 1; idx < kNumPoints; idx++) {
    knots[idx] = knots[idx - 1] + dtdist(engine);
  }

  ASSERT_TRUE(spline.SetKnotVector(knots));
  ASSERT_TRUE(spline.SetControlPoints(points));
  ASSERT_TRUE(spline.CalculateParameters());
  double val, val0, dval, ddval;

  test::FiniteDifferenceTest<double, double> fd_vel;
  test::FiniteDifferenceTest<double, double> fd_acc;

  fd_vel.Init("velocity", kDt);
  fd_acc.Init("acceleration", kDt);

  for (double t = knots[0]; t <= knots[kNumPoints - 1]; t += kDt) {
    ASSERT_TRUE(spline.EvalCurve(t, &val0));
    ASSERT_TRUE(spline.EvalCurveAndDerivatives(t, &val, &dval, &ddval));
    ASSERT_EQ(val0, val);
    fd_vel.Update(val, dval);
    fd_acc.Update(dval, ddval);
    if (absl::GetFlag(FLAGS_verbosity) > 2) {
      fd_vel.PrintCurrent();
      fd_acc.PrintCurrent();
    }
  }

  if (absl::GetFlag(FLAGS_verbosity) > 1) {
    fd_vel.PrintMaxError();
    fd_acc.PrintMaxError();
  }
  ASSERT_GE(kMaxVelError, fd_vel.GetMaxError());
  ASSERT_GE(kMaxAccError, fd_acc.GetMaxError());
}

// final velocities
TEST(CubicSpline, BoundaryConditions) {
  constexpr size_t kNumPoints = 7;
  constexpr size_t kNumLoops = 5;

  std::valarray<double> knots(kNumPoints);
  std::valarray<double> points(kNumPoints);

  std::uniform_real_distribution<double> dtdist(0.1, 1);
  std::uniform_real_distribution<double> pdist(-1, 1);
  std::default_random_engine engine;

  for (size_t loop = 0; loop < kNumLoops; loop++) {
    for (size_t idx = 0; idx < kNumPoints; idx++) {
      points[idx] = pdist(engine);
    }

    knots[0] = pdist(engine);
    for (size_t idx = 1; idx < kNumPoints; idx++) {
      knots[idx] = knots[idx - 1] + dtdist(engine);
    }

    CubicSpline spline;
    double val, dval, ddval;
    ASSERT_TRUE(spline.Init(kNumPoints));

    // 1: set nothing: default is natural spline
    ASSERT_TRUE(spline.SetKnotVector(knots));
    ASSERT_TRUE(spline.SetControlPoints(points));
    ASSERT_TRUE(spline.CalculateParameters());

    ASSERT_TRUE(spline.EvalCurveAndDerivatives(knots[0], &val, &dval, &ddval));
    ASSERT_GE(kTiny, fabs(ddval));
    ASSERT_TRUE(spline.EvalCurveAndDerivatives(knots[kNumPoints - 1], &val,
                                               &dval, &ddval));
    ASSERT_GE(kTiny, fabs(ddval));

    // 2: different accelerations
    double a0 = pdist(engine);
    double a1 = pdist(engine);
    spline.SetBoundaryConditions(
        {CubicSpline::BoundaryCond::kAcceleration, a0},
        {CubicSpline::BoundaryCond::kAcceleration, a1});
    ASSERT_TRUE(spline.CalculateParameters());
    ASSERT_TRUE(spline.EvalCurveAndDerivatives(knots[0], &val, &dval, &ddval));
    ASSERT_GE(kTiny, fabs(ddval - a0));
    ASSERT_TRUE(spline.EvalCurveAndDerivatives(knots[kNumPoints - 1], &val,
                                               &dval, &ddval));
    ASSERT_GE(kTiny, fabs(ddval - a1));

    // 3: different velocities
    double v0 = pdist(engine);
    double v1 = pdist(engine);
    spline.SetBoundaryConditions({CubicSpline::BoundaryCond::kVelocity, v0},
                                 {CubicSpline::BoundaryCond::kVelocity, v1});
    ASSERT_TRUE(spline.CalculateParameters());
    ASSERT_TRUE(spline.EvalCurveAndDerivatives(knots[0], &val, &dval, &ddval));
    ASSERT_GE(kTiny, fabs(dval - v0));
    ASSERT_TRUE(spline.EvalCurveAndDerivatives(knots[kNumPoints - 1], &val,
                                               &dval, &ddval));
    ASSERT_GE(kTiny, fabs(dval - v1));

    // 4: initial velocity & final acceleration
    v0 = pdist(engine);
    a1 = pdist(engine);
    spline.SetBoundaryConditions(
        {CubicSpline::BoundaryCond::kVelocity, v0},
        {CubicSpline::BoundaryCond::kAcceleration, a1});
    ASSERT_TRUE(spline.CalculateParameters());
    ASSERT_TRUE(spline.EvalCurveAndDerivatives(knots[0], &val, &dval, &ddval));
    ASSERT_GE(kTiny, fabs(dval - v0));
    ASSERT_TRUE(spline.EvalCurveAndDerivatives(knots[kNumPoints - 1], &val,
                                               &dval, &ddval));
    ASSERT_GE(kTiny, fabs(ddval - a1));

    // 5: initial acceleration & final velocity
    a0 = pdist(engine);
    v1 = pdist(engine);
    spline.SetBoundaryConditions({CubicSpline::BoundaryCond::kAcceleration, a0},
                                 {CubicSpline::BoundaryCond::kVelocity, v1});
    ASSERT_TRUE(spline.CalculateParameters());
    ASSERT_TRUE(spline.EvalCurveAndDerivatives(knots[0], &val, &dval, &ddval));
    ASSERT_GE(kTiny, fabs(ddval - a0));
    ASSERT_TRUE(spline.EvalCurveAndDerivatives(knots[kNumPoints - 1], &val,
                                               &dval, &ddval));
    ASSERT_GE(kTiny, fabs(dval - v1));
  }
}

}  // namespace
}  // namespace trajectory_planning

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

#include "trajectory_planning/time_optimal_path_timing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <valarray>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/flag.h"
#include "absl/strings/str_cat.h"

ABSL_FLAG(int, verbosity, 0, "verbosity level");
ABSL_FLAG(bool, plot_extremals, false, "write plotfile for extremals");

namespace trajectory_planning {
namespace {
using Scalar = TimeOptimalPathProfile::Scalar;
using Array = Eigen::ArrayX<Scalar>;

constexpr Scalar kTiny = std::numeric_limits<Scalar>::epsilon() * 1e5;
// Extra log files at high verbosity levels go here.
constexpr char kprintfir[] = "/tmp/";

// Make constraints for 1D sinusoidal path with num_samples samples along the
// path with path parameter from s0 to s1, cirle radius R and box constraints
// for velocity
// [-vmax, vmax] and acceleration [-amax, amax]
// This checks solutions with singular points on the maximum velocity boundary
// curve.
std::vector<TimeOptimalPathProfile::Constraint> MakeSineExampleConstraints(
    size_t num_samples, Scalar s0, Scalar s1, Scalar R, Scalar vmax,
    Scalar amax) {
  std::vector<TimeOptimalPathProfile::Constraint> constraints(num_samples);
  for (auto &c : constraints) {
    c.resize(2);
  }

  const Scalar ds = (s1 - s0) / (num_samples - 1);
  for (size_t i = 0; i < num_samples; i++) {
    const Scalar s = i * ds + s0;
    constraints[i].a_coefficient(0) = -R * sin(s);
    constraints[i].a_coefficient(1) = 0.0;

    constraints[i].b_coefficient(0) = -R * cos(s);
    constraints[i].b_coefficient(1) = pow(R * sin(s), 2);

    constraints[i].upper(0) = amax;
    constraints[i].upper(1) = pow(vmax, 2);

    constraints[i].lower(0) = -amax;
    constraints[i].lower(1) = 0;
  }

  return constraints;
}

// Verify the sine following solution (t=time, s = path parameter, sd = ds/dt,
// sdd = d^2s/dt^2,
// R= radius, vmax, amax box constraints for Cartesian velocities and
// accelerations)
bool VerifySineExampleSolution(const char *filename, const Array &t,
                               const Array &s, const Array &sd,
                               const Array &sdd, const Scalar R,
                               const Scalar vmax, const Scalar amax) {
  if (t.size() != s.size() || t.size() != sd.size() || t.size() != sdd.size()) {
    fprintf(stderr, "array sizes don't match\n");
    return false;
  }

  FILE *fp = nullptr;
  if (absl::GetFlag(FLAGS_verbosity) >= 3) {
    fp = fopen(filename, "w");
    if (nullptr == fp) {
      fprintf(stderr, "openining %s (%s)\n", filename, strerror(errno));
      return false;
    }
  }

  const Array sin_s = sin(s);
  const Array cos_s = cos(s);
  const Array sd2 = sd * sd;

  const Array x = R * cos_s;
  const Array xd = -R * sin_s * sd;
  const Array xdd = -R * sin_s * sdd - R * cos_s * sd2;
  // check velocity & acceleration bounds
  constexpr Scalar kMaxViolation = kTiny;
  EXPECT_GT(kMaxViolation, (xd.abs() - vmax).maxCoeff());
  EXPECT_GT(kMaxViolation, (xdd.abs() - amax).maxCoeff());

  if (nullptr != fp) {
    for (int idx = 0; idx < t.size(); idx++) {
      fprintf(fp, "t= %.18e s= %.18e %.18e %.18e x= %.18e %.18e %.18e\n",
              t[idx], s[idx], sd[idx], sdd[idx], x[idx], xd[idx], xdd[idx]);
    }
    fclose(fp);
  }
  return true;
}

// Make constraints for circular path with num_samples samples along the path
// with
// path parameter from s0 to s1, cirle radius R and box constraints for velocity
// [-vmax, vmax] and acceleration [-amax, amax] that are equal in both Cartesian
// directions.
std::vector<TimeOptimalPathProfile::Constraint> MakeCircleExampleConstraints(
    size_t num_samples, Scalar s0, Scalar s1, Scalar R, Scalar vmax,
    Scalar amax) {
  std::vector<TimeOptimalPathProfile::Constraint> constraints(num_samples);
  for (auto &c : constraints) {
    c.resize(4);
  }

  const Scalar ds = (s1 - s0) / (num_samples - 1);
  for (size_t i = 0; i < num_samples; i++) {
    const Scalar s = i * ds + s0;
    constraints[i].a_coefficient(0) = -R * sin(s);
    constraints[i].a_coefficient(1) = R * cos(s);
    constraints[i].a_coefficient(2) = 0.0;
    constraints[i].a_coefficient(3) = 0.0;

    constraints[i].b_coefficient(0) = -R * cos(s);
    constraints[i].b_coefficient(1) = -R * sin(s);
    constraints[i].b_coefficient(2) = pow(R * sin(s), 2);
    constraints[i].b_coefficient(3) = pow(R * cos(s), 2);

    constraints[i].upper(0) = amax;
    constraints[i].upper(1) = amax;
    constraints[i].upper(2) = pow(vmax, 2);
    constraints[i].upper(3) = pow(vmax, 2);

    constraints[i].lower(0) = -amax;
    constraints[i].lower(1) = -amax;
    constraints[i].lower(2) = 0;
    constraints[i].lower(3) = 0;
  }

  return constraints;
}

// Verify the circle following solution (t=time, s = path parameter, sd = ds/dt,
// sdd = d^2s/dt^2,
// R= radius, vmax, amax box constraints for Cartesian velocities and
// accelerations)
bool VerifyCircleExampleSolution(const char *filename, const Array &t,
                                 const Array &s, const Array &sd,
                                 const Array &sdd, const Scalar R,
                                 const Scalar vmax, const Scalar amax) {
  if (t.size() != s.size() || t.size() != sd.size() || t.size() != sdd.size()) {
    fprintf(stderr, "array sizes don't match\n");
    return false;
  }

  FILE *fp = nullptr;
  if (absl::GetFlag(FLAGS_verbosity) >= 3) {
    fp = fopen(filename, "w");
    if (nullptr == fp) {
      fprintf(stderr, "openining %s (%s)\n", filename, strerror(errno));
      return false;
    }
  }

  const Array sin_s = sin(s);
  const Array cos_s = cos(s);
  const Array sd2 = sd * sd;

  const Array x = R * cos_s;
  const Array y = R * sin_s;
  const Array xd = -R * sin_s * sd;
  const Array yd = R * cos_s * sd;
  const Array xdd = -R * sin_s * sdd - R * cos_s * sd2;
  const Array ydd = R * cos_s * sdd - R * sin_s * sd2;
  // check velocity & acceleration bounds
  constexpr Scalar kMaxViolation = kTiny;
  EXPECT_GT(kMaxViolation, (xd.abs() - vmax).maxCoeff());
  EXPECT_GT(kMaxViolation, (xdd.abs() - amax).maxCoeff());
  EXPECT_GT(kMaxViolation, (yd.abs() - vmax).maxCoeff());
  EXPECT_GT(kMaxViolation, (ydd.abs() - amax).maxCoeff());

  if (nullptr != fp) {
    for (int idx = 0; idx < t.size(); idx++) {
      fprintf(fp,
              "t= %.18e s= %.18e %.18e %.18e x= %.18e %.18e %.18e y= %.18e "
              "%.18e %.18e \n",
              t[idx], s[idx], sd[idx], sdd[idx], x[idx], xd[idx], xdd[idx],
              y[idx], yd[idx], ydd[idx]);
    }
    fclose(fp);
  }
  return true;
}

// Make constraints for path following of a line in a a plane with given slope,
// initial and final path parameter values s0, s1 and box constraints for
// Cartesian
// velocity and acceleration vmax, amax
std::vector<TimeOptimalPathProfile::Constraint> MakeLineExampleConstraints(
    size_t num_samples, Scalar s0, Scalar s1, Scalar slope, Scalar vmax,
    Scalar amax) {
  std::vector<TimeOptimalPathProfile::Constraint> constraints(num_samples);
  for (auto &c : constraints) {
    c.resize(4);
  }

  for (size_t i = 0; i < num_samples; i++) {
    constraints[i].a_coefficient(0) = 1;
    constraints[i].a_coefficient(1) = slope;
    constraints[i].a_coefficient(2) = 0;
    constraints[i].a_coefficient(3) = 0;

    constraints[i].b_coefficient(0) = 0;
    constraints[i].b_coefficient(1) = 0;
    constraints[i].b_coefficient(2) = pow(slope, 2);
    constraints[i].b_coefficient(3) = 1;

    constraints[i].upper(0) = amax;
    constraints[i].upper(1) = amax;
    constraints[i].upper(2) = pow(vmax, 2);
    constraints[i].upper(3) = pow(vmax, 2);

    constraints[i].lower(0) = -amax;
    constraints[i].lower(1) = -amax;
    constraints[i].lower(2) = 0.0;
    constraints[i].lower(3) = 0.0;
  }

  return constraints;
}

// Verify timing solution for following line in the plane (t= time, s=path
// parameter,
// s= ds/dt, sdd= d^2s/dt^2, vmax, amax box constraints for velocities and
// accelerations).
bool VerifyLineExampleSolution(const char *filename, const Array &t,
                               const Array &s, const Array &sd,
                               const Array &sdd, const Scalar slope,
                               Scalar vmax, Scalar amax) {
  if (t.size() != s.size() || t.size() != sd.size() || t.size() != sdd.size()) {
    fprintf(stderr, "array sizes don't match\n");
    return false;
  }
  FILE *fp = nullptr;
  if (absl::GetFlag(FLAGS_verbosity) >= 3) {
    fp = fopen(filename, "w");
    if (nullptr == fp) {
      fprintf(stderr, "openining %s (%s)\n", filename, strerror(errno));
    }
  }

  const Array &x = s;
  const Array y = slope * s;
  const Array &xd = sd;
  const Array yd = slope * sd;
  const Array &xdd = sdd;
  const Array ydd = slope * sdd;

  // check velocity & acceleration bounds
  constexpr Scalar kMaxViolation = kTiny;
  EXPECT_GT(kMaxViolation, (xd.abs() - vmax).maxCoeff());
  EXPECT_GT(kMaxViolation, (xdd.abs() - amax).maxCoeff());
  EXPECT_GT(kMaxViolation, (yd.abs() - vmax).maxCoeff());
  EXPECT_GT(kMaxViolation, (ydd.abs() - amax).maxCoeff());

  if (nullptr != fp) {
    for (int idx = 0; idx < t.size(); idx++) {
      fprintf(fp,
              "t= %.18e s= %.18e %.18e %.18e x= %.18e %.18e %.18e y= %.18e "
              "%.18e %.18e \n",
              t[idx], s[idx], sd[idx], sdd[idx], x[idx], xd[idx], xdd[idx],
              y[idx], yd[idx], ydd[idx]);
    }
    fclose(fp);
  }
  return true;
}

// Make constraints for scalar problem, with box constraints on acceleration and
// velocity
// directly applied to the path parameter.
std::vector<TimeOptimalPathProfile::Constraint>
MakeScalarStraightExampleConstraints(size_t num_samples, Scalar s0, Scalar s1,
                                     Scalar vmax, Scalar amax) {
  std::vector<TimeOptimalPathProfile::Constraint> constraints(num_samples);
  for (auto &c : constraints) {
    c.resize(2);
  }

  for (size_t i = 0; i < num_samples; i++) {
    constraints[i].a_coefficient(0) = 1;
    constraints[i].a_coefficient(1) = 0;

    constraints[i].b_coefficient(0) = 0;
    constraints[i].b_coefficient(1) = 1;

    constraints[i].upper(0) = amax;
    constraints[i].upper(1) = pow(vmax, 2);

    constraints[i].lower(0) = -amax;
    constraints[i].lower(1) = 0.0;
  }

  return constraints;
}

// Verify solution for scalar timing problem (t= time, s=path parameter,
// s= ds/dt, sdd= d^2s/dt^2, vmax, amax box constraints for velocities and
// accelerations).
bool VerifyScalarStraightExampleSolution(const char *filename, const Array &t,
                                         const Array &s, const Array &sd,
                                         const Array &sdd, Scalar vmax,
                                         Scalar amax) {
  if (t.size() != s.size() || t.size() != sd.size() || t.size() != sdd.size()) {
    fprintf(stderr, "array sizes don't match\n");
    return false;
  }
  FILE *fp = nullptr;
  if (absl::GetFlag(FLAGS_verbosity) >= 3) {
    fp = fopen(filename, "w");
    if (nullptr == fp) {
      fprintf(stderr, "openining %s (%s)\n", filename, strerror(errno));
    }
  }

  // check velocity & acceleration bounds
  constexpr Scalar kMaxViolation = kTiny;
  EXPECT_GT(kMaxViolation, (sd.abs() - vmax).maxCoeff());
  EXPECT_GT(kMaxViolation, (sdd.abs() - amax).maxCoeff());

  if (nullptr != fp) {
    for (int idx = 0; idx < t.size(); idx++) {
      fprintf(fp, "t= %.18e s= %.18e %.18e %.18e\n", t[idx], s[idx], sd[idx],
              sdd[idx]);
    }
    fclose(fp);
  }
  return true;
}

// Make scalar example where the scalar value is a cubic function of the
// path parameter, parametrized by m, and box constraints for the velocity and
// acceleration given by vmax and amax, respectively.
// The path is given by x = m[0]*s + m[1]*s^2 + m[2]*s^3.
std::vector<TimeOptimalPathProfile::Constraint>
MakeScalarCurvedExampleConstraints(size_t num_samples, Scalar s0, Scalar s1,
                                   std::array<Scalar, 3> m, Scalar vmax,
                                   Scalar amax) {
  std::vector<TimeOptimalPathProfile::Constraint> constraints(num_samples);
  for (auto &c : constraints) {
    c.resize(2);
  }

  const Scalar ds = (s1 - s0) / (num_samples - 1);
  for (size_t i = 0; i < num_samples; i++) {
    const Scalar s = i * ds + s0;
    constraints[i].a_coefficient(0) =
        3.0 * m[0] * s * s + 2.0 * m[1] * s + m[2];
    constraints[i].b_coefficient(0) = 6.0 * m[0] * s + 2.0 * m[1];

    constraints[i].a_coefficient(1) = 0;
    constraints[i].b_coefficient(1) =
        std::pow(3.0 * m[0] * s * s + 2.0 * m[1] * s + m[2], 2.0);

    constraints[i].upper(0) = amax;
    constraints[i].upper(1) = vmax;
    constraints[i].lower(0) = -amax;
    constraints[i].lower(1) = 0.0;
  }

  return constraints;
}

// Verify scalar nonlinear example (t= time, s=path parameter,
// s= ds/dt, sdd= d^2s/dt^2,  amax box constraints for accelerations,
// vmax upper limit for velocity).
bool VerifyScalarCurvedExampleSolution(const char *filename, const Array &t,
                                       const Array &s, const Array &sd,
                                       const Array &sdd,
                                       std::array<Scalar, 3> m, Scalar vmax,
                                       Scalar amax) {
  if (t.size() != s.size() || t.size() != sd.size() || t.size() != sdd.size()) {
    fprintf(stderr, "array sizes don't match\n");
    return false;
  }
  FILE *fp = nullptr;
  if (absl::GetFlag(FLAGS_verbosity) >= 3) {
    fp = fopen(filename, "w");
    if (nullptr == fp) {
      fprintf(stderr, "openining %s (%s)\n", filename, strerror(errno));
    }
  }

  const Array x = m[0] * s.pow(3.0) + m[1] * s.pow(2.0) + m[2] * s;
  const Array xd =
      3.0 * m[0] * s.pow(2.0) * sd + 2.0 * m[1] * s * sd + m[2] * sd;
  const Array xdd = 6.0 * m[0] * s * sd.pow(2.0) +
                    3.0 * m[0] * s.pow(2.0) * sdd + 2.0 * m[1] * sd.pow(2.0) +
                    2.0 * m[1] * s * sdd + m[2] * sdd;

  // Check velocity & acceleration bounds.
  constexpr Scalar kMaxViolation = kTiny;
  EXPECT_GT(kMaxViolation, (xdd.abs() - amax).maxCoeff());
  EXPECT_GT(kMaxViolation, (xd.abs() - vmax).maxCoeff());

  // Expect that the middle section is close to the maximum velocity.
  const Scalar max_velocity_error_in_middle_segment =
      (xd.segment(0.3 * xd.size(), 0.3 * xd.size()) - vmax).maxCoeff();
  printf("max_velocity_error_in_middle_segment= %e\n",
         max_velocity_error_in_middle_segment);
  EXPECT_LT(max_velocity_error_in_middle_segment, kMaxViolation);

  if (nullptr != fp) {
    for (int idx = 0; idx < t.size(); idx++) {
      fprintf(fp, "t= %.18e s= %.18e %.18e %.18e x= %.18e %.18e %.18e\n",
              t[idx], s[idx], sd[idx], sdd[idx], x[idx], xd[idx], xdd[idx]);
    }
    fclose(fp);
  }
  return true;
}

TEST(TimeOptimalPathTimingTest, Sine) {
  // check for even and odd number of samples: misses/hit critical point
  std::vector<size_t> num_samples_array = {30, 31, 100, 111};
  std::vector<Scalar> offsets = {0, M_PI_2, M_PI_4};
  constexpr Scalar kPathDStart = 0.0;
  constexpr Scalar kPathDDStart = 0.0;
  constexpr Scalar kTimeStart = 0.0;

  for (auto offset : offsets) {
    SCOPED_TRACE(testing::Message() << " offset= " << offset);
    for (auto num_samples : num_samples_array) {
      SCOPED_TRACE(testing::Message() << " num_samples= " << num_samples);
      TimeOptimalPathProfile opt;

      Scalar path_start = offset;
      Scalar path_end = M_PI + offset;
      constexpr Scalar kMaxAcc = 1;
      constexpr Scalar kMaxVel = 1.2;
      constexpr Scalar R = 2.0;
      std::vector<TimeOptimalPathProfile::Constraint> constraints;
      constraints = MakeSineExampleConstraints(num_samples, path_start,
                                               path_end, R, kMaxVel, kMaxAcc);

      ASSERT_TRUE(opt.InitSolver(num_samples, constraints[0].size()));

      // set constraints and other problem parameters.
      // This also tells the solver that problem setup is done
      ASSERT_TRUE(opt.SetupProblem(constraints, path_start, path_end,
                                   kPathDStart, kPathDDStart, kTimeStart));

      if (absl::GetFlag(FLAGS_verbosity) >= 2) {
        printf("Setup done .. calling optimize");
      }
      opt.SetDebugVerbosity(absl::GetFlag(FLAGS_verbosity));
      ASSERT_TRUE(opt.OptimizePathParameter());

      if (absl::GetFlag(FLAGS_verbosity) >= 2) {
        printf("Optimization done");
      }
      std::string solution = absl::StrCat(kprintfir, "sine-solution-", offset,
                                          "-", num_samples, ".txt");
      EXPECT_TRUE(VerifySineExampleSolution(
          solution.c_str(), opt.GetTimeSamples(), opt.GetPathParameter(),
          opt.GetPathVelocity(), opt.GetPathAcceleration(), R, kMaxVel,
          kMaxAcc))
          << "num_samples= " << num_samples << " offset= " << offset;

      if (absl::GetFlag(FLAGS_plot_extremals)) {
        EXPECT_TRUE(opt.PlotAllExtremals(
            absl::StrCat(kprintfir, "sine_extremals.txt").c_str()));
      }
      ASSERT_TRUE(opt.SolutionSatisfiesConstraints().ok());
    }
  }
}

TEST(TimeOptimalPathTimingTest, Circle) {
  constexpr Scalar kPathStart = 0;
  constexpr Scalar kPathDDStart = 0.0;
  constexpr Scalar kPathEnd = M_PI;
  constexpr Scalar kTimeStart = 0.0;
  constexpr Scalar kMaxAcc = 1;
  constexpr Scalar kMaxVel = 1.2;
  constexpr Scalar R = 2.0;

  // check for even/odd number of samples: misses/hits critical point;
  struct TestData {
    size_t num_samples;
    Scalar sd_start;
  };
  std::vector<TestData> test_data = {
      {50, 0.0},
      {51, 0.0},
      {50, 0.1},
      {51, 0.1},
  };

  for (auto data : test_data) {
    SCOPED_TRACE(testing::Message() << " num_samples= " << data.num_samples
                                    << " sd_start= " << data.sd_start);
    TimeOptimalPathProfile opt;
    std::vector<TimeOptimalPathProfile::Constraint> constraints;
    constraints = MakeCircleExampleConstraints(data.num_samples, kPathStart,
                                               kPathEnd, R, kMaxVel, kMaxAcc);

    ASSERT_TRUE(opt.InitSolver(data.num_samples, constraints[0].size()));

    // set constraints and other problem parameters.
    // This also tells the solver that problem setup is done
    ASSERT_TRUE(opt.SetupProblem(constraints, kPathStart, kPathEnd,
                                 data.sd_start, kPathDDStart, kTimeStart));

    if (absl::GetFlag(FLAGS_verbosity) >= 2) {
      printf("Setup done .. calling optimize");
    }
    opt.SetDebugVerbosity(absl::GetFlag(FLAGS_verbosity));
    ASSERT_TRUE(opt.OptimizePathParameter());

    Scalar s0, sd0, sdd0;
    EXPECT_TRUE(
        opt.GetPathParameterAndDerivatives(kTimeStart, &s0, &sd0, &sdd0));
    EXPECT_DOUBLE_EQ(data.sd_start, sd0);

    if (absl::GetFlag(FLAGS_verbosity) >= 2) {
      printf("Optimization done");
    }

    EXPECT_TRUE(VerifyCircleExampleSolution(
        absl::StrCat(kprintfir, "circle_solution.txt").c_str(),
        opt.GetTimeSamples(), opt.GetPathParameter(), opt.GetPathVelocity(),
        opt.GetPathAcceleration(), R, kMaxVel, kMaxAcc));

    ASSERT_TRUE(opt.SolutionSatisfiesConstraints().ok());

    if (absl::GetFlag(FLAGS_plot_extremals)) {
      EXPECT_TRUE(opt.PlotAllExtremals(
          absl::StrCat(kprintfir, "circle_extremals.txt").c_str()));
    }
  }
}

// optimal timing along line: simplest 2D case
TEST(TimeOptimalPathTimingTest, Line) {
  TimeOptimalPathProfile opt;

  constexpr size_t kNumSamples = 30;
  constexpr Scalar kPathStart = 0;
  constexpr Scalar kPathEnd = M_PI;
  constexpr Scalar kPathDStart = 0.0;
  constexpr Scalar kPathDDStart = 0.0;
  constexpr Scalar kTimeStart = 0.0;
  constexpr Scalar kMaxAcc = 1;
  constexpr Scalar kMaxVel = 1.0;
  constexpr Scalar slope = 2.0;
  std::vector<TimeOptimalPathProfile::Constraint> constraints;

  constraints = MakeLineExampleConstraints(kNumSamples, kPathStart, kPathEnd,
                                           slope, kMaxVel, kMaxAcc);
  ASSERT_TRUE(opt.InitSolver(kNumSamples, constraints[0].size()));

  // set constraints and other problem parameters.
  // This also tells the solver that problem setup is done
  ASSERT_TRUE(opt.SetupProblem(constraints, kPathStart, kPathEnd, kPathDStart,
                               kPathDDStart, kTimeStart));

  if (absl::GetFlag(FLAGS_verbosity) >= 2) {
    printf("Setup done .. calling optimize");
  }

  opt.SetDebugVerbosity(absl::GetFlag(FLAGS_verbosity));

  ASSERT_TRUE(opt.OptimizePathParameter());

  if (absl::GetFlag(FLAGS_verbosity) >= 2) {
    printf("Optimization done");
  }

  EXPECT_TRUE(VerifyLineExampleSolution(
      absl::StrCat(kprintfir, "line_solution.txt").c_str(),
      opt.GetTimeSamples(), opt.GetPathParameter(), opt.GetPathVelocity(),
      opt.GetPathAcceleration(), slope, kMaxVel, kMaxAcc));

  if (absl::GetFlag(FLAGS_plot_extremals)) {
    EXPECT_TRUE(opt.PlotAllExtremals(
        absl::StrCat(kprintfir, "line_extremals.txt").c_str()));
  }
}

// optimal timing in scalar case (1D): simplest possible case
TEST(TimeOptimalPathTimingTest, ScalarStraight) {
  TimeOptimalPathProfile opt;

  constexpr size_t kNumSamples = 30;
  constexpr Scalar kPathStart = 0;
  constexpr Scalar kPathEnd = 1.0;
  constexpr Scalar kPathDStart = 0.0;
  constexpr Scalar kPathDDStart = 0.0;
  constexpr Scalar kTimeStart = 0.0;
  constexpr Scalar kMaxAcc = 1.0;
  constexpr Scalar kMaxVel = 0.5;
  std::vector<TimeOptimalPathProfile::Constraint> constraints;

  constraints = MakeScalarStraightExampleConstraints(
      kNumSamples, kPathStart, kPathEnd, kMaxVel, kMaxAcc);
  ASSERT_TRUE(opt.InitSolver(kNumSamples, constraints[0].size()));

  // set constraints and other problem parameters.
  // This also tells the solver that problem setup is done
  ASSERT_TRUE(opt.SetupProblem(constraints, kPathStart, kPathEnd, kPathDStart,
                               kPathDDStart, kTimeStart));

  if (absl::GetFlag(FLAGS_verbosity) >= 2) {
    printf("Setup done .. calling optimize");
  }
  opt.SetDebugVerbosity(absl::GetFlag(FLAGS_verbosity));
  ASSERT_TRUE(opt.OptimizePathParameter());

  if (absl::GetFlag(FLAGS_verbosity) >= 2) {
    printf("Optimization done");
  }

  EXPECT_TRUE(VerifyScalarStraightExampleSolution(
      absl::StrCat(kprintfir, "scalar_straight_solution.txt").c_str(),
      opt.GetTimeSamples(), opt.GetPathParameter(), opt.GetPathVelocity(),
      opt.GetPathAcceleration(), kMaxVel, kMaxAcc));

  if (absl::GetFlag(FLAGS_plot_extremals)) {
    EXPECT_TRUE(opt.PlotAllExtremals(
        absl::StrCat(kprintfir, "scalar_straight_extremals.txt").c_str()));
  }
}

// This test is parameterized such that the boundary curve for (sd)^2 has a
// smooth maximum within the planned path section, but the maximum sd is
// always bounded. The optimal solution should be a simple velocity ramp, with
// the central section tracking the boundary curve.
// This is a counter example to the 'skipped critical point'
// case covered by the Sine test.
TEST(TimeOptimalPathTimingTest, SmoothBoundaryCurveWithMaximumIsTracked) {
  TimeOptimalPathProfile opt;

  constexpr size_t kNumSamples = 100;
  constexpr Scalar kPathStart = -3;
  constexpr Scalar kPathDStart = 0.0;
  constexpr Scalar kPathDDStart = 0.0;
  constexpr Scalar kTimeStart = 0.0;
  constexpr Scalar kPathEnd = 1.0;
  constexpr Scalar kMaxVel = 1.0;
  constexpr Scalar kMaxAcc = 0.2;
  constexpr std::array<Scalar, 3> kM = {{1, 1, 2}};

  std::vector<TimeOptimalPathProfile::Constraint> constraints;

  constraints = MakeScalarCurvedExampleConstraints(
      kNumSamples, kPathStart, kPathEnd, kM, kMaxVel, kMaxAcc);
  ASSERT_TRUE(opt.InitSolver(kNumSamples, constraints[0].size()));

  // set constraints and other problem parameters.
  // This also tells the solver that problem setup is done
  ASSERT_TRUE(opt.SetupProblem(constraints, kPathStart, kPathEnd, kPathDStart,
                               kPathDDStart, kTimeStart));

  if (absl::GetFlag(FLAGS_verbosity) >= 2) {
    printf("Setup done .. calling optimize");
  }
  opt.SetDebugVerbosity(absl::GetFlag(FLAGS_verbosity));
  ASSERT_TRUE(opt.OptimizePathParameter());

  if (absl::GetFlag(FLAGS_verbosity) >= 2) {
    printf("Optimization done");
  }

  EXPECT_TRUE(VerifyScalarCurvedExampleSolution(
      absl::StrCat(kprintfir, "scalar_curved_solution.txt").c_str(),
      opt.GetTimeSamples(), opt.GetPathParameter(), opt.GetPathVelocity(),
      opt.GetPathAcceleration(), kM, kMaxVel, kMaxAcc));

  if (absl::GetFlag(FLAGS_plot_extremals)) {
    EXPECT_TRUE(opt.PlotAllExtremals(
        absl::StrCat(kprintfir, "scalar_curved_extremals.txt").c_str()));
  }
}

TEST(TimeOptimalPathTimingTest, FindMaxSd2Random) {
  TimeOptimalPathProfile profile;
  TimeOptimalPathProfile::Constraint constraint;

  std::mt19937 gen;
  std::uniform_real_distribution<> ldist(-10, 0);
  std::uniform_real_distribution<> udist(0, 10);
  std::uniform_real_distribution<> abdist(-100, 100);
  std::uniform_int_distribution<> szdist(2, 50);

  constexpr int kNumTestCases = 1e5;
  for (int counter = 0; counter < kNumTestCases; counter++) {
    constraint.resize(szdist(gen));
    profile.InitSolver(10, constraint.size());
    if (counter % 10000 == 0) {
      printf("== case %d\n", counter);
      fflush(stdout);
    }
    for (int idx = 0; idx < constraint.size(); idx++) {
      constraint.a_coefficient(idx) = abdist(gen);
      constraint.b_coefficient(idx) = abdist(gen);
      constraint.lower(idx) = ldist(gen);
      constraint.upper(idx) = udist(gen);
    }
    Scalar sd2max_ref, sddmax_ref, sd2zero_ref;
    Scalar sd2max, sddmax, sd2zero;
    profile.FindMaxSd2BruteForce(constraint, &sd2max_ref, &sddmax_ref,
                                 &sd2zero_ref);
    profile.FindMaxSd2Simplex(constraint, &sd2max, &sddmax, &sd2zero);
    ASSERT_NEAR(sd2max_ref, sd2max, 1e-8);
    ASSERT_NEAR(sddmax_ref, sddmax, 1e-8);
    ASSERT_NEAR(sd2zero_ref, sd2zero, 1e-8);
  }
}

// Test LP solver on regression cases found during development.
// They are from cubic b-spline timing with augmented acceleration constraints
// using maximum and minimum path curvature between discrete path sample points.
// This leads to multiple inequalities of different slope intersecting in one
// (or almost one) point, as well as exacty horizontal and redundant
// constraints.
TEST(TimeOptimalPathTimingTest, FindMaxSd2Regression) {
  using Map = Eigen::Map<const Eigen::ArrayX<TimeOptimalPathProfile::Scalar>>;
  struct LPInfo {
    int sz;
    std::vector<TimeOptimalPathProfile::Scalar> a;
    std::vector<TimeOptimalPathProfile::Scalar> b;
    std::vector<TimeOptimalPathProfile::Scalar> lower;
    std::vector<TimeOptimalPathProfile::Scalar> upper;
  };

  TimeOptimalPathProfile profile;
  TimeOptimalPathProfile::Constraint constraint;
  Scalar sd2max_ref, sddmax_ref, sd2zero_ref;
  Scalar sd2max, sddmax, sd2zero;
  std::vector<LPInfo> lpinfo = {
      {30,
       {3.825007867086686719e+00, 3.892509184519236776e+00,
        3.960010501951786388e+00, 3.892509184519236776e+00,
        3.960010501951786388e+00, 3.892509184519236776e+00,
        3.825007867086686719e+00, 3.892509184519236776e+00,
        3.960010501951786388e+00, 3.892509184519236776e+00,
        3.960010501951786388e+00, 3.892509184519236776e+00,
        1.976214770076943683e+00, 2.115296431921398046e+00,
        2.254378093765852853e+00, 2.115296431921398046e+00,
        2.254378093765852853e+00, 2.115296431921398046e+00,
        1.976214770076943683e+00, 2.115296431921398046e+00,
        2.254378093765852853e+00, 2.115296431921398046e+00,
        2.254378093765852853e+00, 2.115296431921398046e+00,
        0.000000000000000000e+00, 0.000000000000000000e+00,
        0.000000000000000000e+00, 0.000000000000000000e+00,
        0.000000000000000000e+00, 0.000000000000000000e+00},
       {-5.777713e+01, -5.533262e+01, -5.288810e+01, -5.533262e+01,
        -5.288810e+01, -5.533262e+01, -5.937871e+01, -5.710278e+01,
        -5.482685e+01, -5.710278e+01, -5.482685e+01, -5.710278e+01,
        -5.777713e+01, -5.533262e+01, -5.288810e+01, -5.533262e+01,
        -5.288810e+01, -5.533262e+01, -5.937871e+01, -5.710278e+01,
        -5.482685e+01, -5.710278e+01, -5.482685e+01, -5.710278e+01,
        9.507914e+00,  1.010867e+01,  1.072783e+01,  1.010867e+01,
        1.072783e+01,  1.010867e+01},
       {-1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        0.000000000000000000e+00,  0.000000000000000000e+00,
        0.000000000000000000e+00,  0.000000000000000000e+00,
        0.000000000000000000e+00,  0.000000000000000000e+00},
       {1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        9.801000000000001933e-01, 8.100000000000000533e-01,
        8.100000000000000533e-01, 8.100000000000000533e-01,
        1.822500000000000231e+00, 1.822500000000000231e+00}},
      {30,
       {1.496328299908580295e+00,  1.664752639701297277e+00,
        1.833176979494014480e+00,  1.664752639701297277e+00,
        1.833176979494014480e+00,  1.664752639701297277e+00,
        1.496328299908580295e+00,  1.664752639701297277e+00,
        1.833176979494014480e+00,  1.664752639701297277e+00,
        1.833176979494014480e+00,  1.664752639701297277e+00,
        -2.745845390740697667e-02, 2.467574243176107185e-01,
        5.209733025426284136e-01,  2.467574243176107185e-01,
        5.209733025426284136e-01,  2.467574243176107185e-01,
        -2.745845390740697667e-02, 2.467574243176107185e-01,
        5.209733025426284136e-01,  2.467574243176107185e-01,
        5.209733025426284136e-01,  2.467574243176107185e-01,
        0.000000000000000000e+00,  0.000000000000000000e+00,
        0.000000000000000000e+00,  0.000000000000000000e+00,
        0.000000000000000000e+00,  0.000000000000000000e+00},
       {-4.736687529332619562e+01, -4.382654637683421583e+01,
        -4.028621746034223605e+01, -4.382654637683421583e+01,
        -4.028621746034223605e+01, -4.382654637683421583e+01,
        -4.896845266031066046e+01, -4.559671083508020217e+01,
        -4.222496900984974388e+01, -4.559671083508020217e+01,
        -4.222496900984974388e+01, -4.559671083508020217e+01,
        -4.736687529332619562e+01, -4.382654637683421583e+01,
        -4.028621746034223605e+01, -4.382654637683421583e+01,
        -4.028621746034223605e+01, -4.382654637683421583e+01,
        -4.896845266031066046e+01, -4.559671083508020217e+01,
        -4.222496900984974388e+01, -4.559671083508020217e+01,
        -4.222496900984974388e+01, -4.559671083508020217e+01,
        7.828950348045660146e-01,  1.199778443477866441e+00,
        1.705307971634331787e+00,  1.199778443477866441e+00,
        1.705307971634331787e+00,  1.199778443477866441e+00},
       {-1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        0.000000000000000000e+00,  0.000000000000000000e+00,
        0.000000000000000000e+00,  0.000000000000000000e+00,
        0.000000000000000000e+00,  0.000000000000000000e+00},
       {1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        9.801000000000001933e-01, 8.100000000000000533e-01,
        8.100000000000000533e-01, 8.100000000000000533e-01,
        1.822500000000000231e+00, 1.822500000000000231e+00}},
      {30,
       {6.805325853767882016e+00, 6.813878652272981284e+00,
        6.822431450778079665e+00, 6.813878652272981284e+00,
        6.822431450778079665e+00, 6.813878652272981284e+00,
        6.805325853767882016e+00, 6.813878652272981284e+00,
        6.822431450778079665e+00, 6.813878652272981284e+00,
        6.822431450778079665e+00, 6.813878652272981284e+00,
        4.606525925626393736e+00, 4.649816244213740291e+00,
        4.693106562801087733e+00, 4.649816244213740291e+00,
        4.693106562801087733e+00, 4.649816244213740291e+00,
        4.606525925626393736e+00, 4.649816244213740291e+00,
        4.693106562801087733e+00, 4.649816244213740291e+00,
        4.693106562801087733e+00, 4.649816244213740291e+00,
        0.000000000000000000e+00, 0.000000000000000000e+00,
        0.000000000000000000e+00, 0.000000000000000000e+00,
        0.000000000000000000e+00, 0.000000000000000000e+00},
       {-6.898816974761648169e+01, -6.772376656315506693e+01,
        -6.645936337869365218e+01, -6.772376656315506693e+01,
        -6.645936337869365218e+01, -6.772376656315506693e+01,
        -7.058974711460095364e+01, -6.949393102140105327e+01,
        -6.839811492820115291e+01, -6.949393102140105327e+01,
        -6.839811492820115291e+01, -6.949393102140105327e+01,
        -6.898816974761648169e+01, -6.772376656315506693e+01,
        -6.645936337869365218e+01, -6.772376656315506693e+01,
        -6.645936337869365218e+01, -6.772376656315506693e+01,
        -7.058974711460095364e+01, -6.949393102140105327e+01,
        -6.839811492820115291e+01, -6.949393102140105327e+01,
        -6.839811492820115291e+01, -6.949393102140105327e+01,
        3.509147567120334799e+01,  3.535542878001770362e+01,
        3.562037088172993293e+01,  3.535542878001770362e+01,
        3.562037088172993293e+01,  3.535542878001770362e+01},
       {-1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        -1.520000000000000018e+00, -1.520000000000000018e+00,
        0.000000000000000000e+00,  0.000000000000000000e+00,
        0.000000000000000000e+00,  0.000000000000000000e+00,
        0.000000000000000000e+00,  0.000000000000000000e+00},
       {1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        1.520000000000000018e+00, 1.520000000000000018e+00,
        9.801000000000001933e-01, 8.100000000000000533e-01,
        8.100000000000000533e-01, 8.100000000000000533e-01,
        1.822500000000000231e+00, 1.822500000000000231e+00}},
      {30,
       {
           6.805325853767882016e+00, 6.813878652272981284e+00,
           6.822431450778079665e+00, 6.813878652272981284e+00,
           6.822431450778079665e+00, 6.813878652272981284e+00,
           6.805325853767882016e+00, 6.813878652272981284e+00,
           6.822431450778079665e+00, 6.813878652272981284e+00,
           6.822431450778079665e+00, 6.813878652272981284e+00,
           4.606525925626393736e+00, 4.649816244213740291e+00,
           4.693106562801087733e+00, 4.649816244213740291e+00,
           4.693106562801087733e+00, 4.649816244213740291e+00,
           4.606525925626393736e+00, 4.649816244213740291e+00,
           4.693106562801087733e+00, 4.649816244213740291e+00,
           4.693106562801087733e+00, 4.649816244213740291e+00,
           0.000000000000000000e+00, 0.000000000000000000e+00,
           0.000000000000000000e+00, 0.000000000000000000e+00,
           0.000000000000000000e+00, 0.000000000000000000e+00,
       },
       {
           -6.898816974761648169e+01, -6.772376656315506693e+01,
           -6.645936337869365218e+01, -6.772376656315506693e+01,
           -6.645936337869365218e+01, -6.772376656315506693e+01,
           -7.058974711460095364e+01, -6.949393102140105327e+01,
           -6.839811492820115291e+01, -6.949393102140105327e+01,
           -6.839811492820115291e+01, -6.949393102140105327e+01,
           -6.898816974761648169e+01, -6.772376656315506693e+01,
           -6.645936337869365218e+01, -6.772376656315506693e+01,
           -6.645936337869365218e+01, -6.772376656315506693e+01,
           -7.058974711460095364e+01, -6.949393102140105327e+01,
           -6.839811492820115291e+01, -6.949393102140105327e+01,
           -6.839811492820115291e+01, -6.949393102140105327e+01,
           3.509147567120334799e+01,  3.535542878001770362e+01,
           3.562037088172993293e+01,  3.535542878001770362e+01,
           3.562037088172993293e+01,  3.535542878001770362e+01,
       },
       {
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           0.000000000000000000e+00,  0.000000000000000000e+00,
           0.000000000000000000e+00,  0.000000000000000000e+00,
           0.000000000000000000e+00,  0.000000000000000000e+00,
       },
       {
           1.520000000000000018e+00, 1.520000000000000018e+00,
           1.520000000000000018e+00, 1.520000000000000018e+00,
           1.520000000000000018e+00, 1.520000000000000018e+00,
           1.520000000000000018e+00, 1.520000000000000018e+00,
           1.520000000000000018e+00, 1.520000000000000018e+00,
           1.520000000000000018e+00, 1.520000000000000018e+00,
           1.520000000000000018e+00, 1.520000000000000018e+00,
           1.520000000000000018e+00, 1.520000000000000018e+00,
           1.520000000000000018e+00, 1.520000000000000018e+00,
           1.520000000000000018e+00, 1.520000000000000018e+00,
           1.520000000000000018e+00, 1.520000000000000018e+00,
           1.520000000000000018e+00, 1.520000000000000018e+00,
           9.801000000000001933e-01, 8.100000000000000533e-01,
           8.100000000000000533e-01, 8.100000000000000533e-01,
           1.822500000000000231e+00, 1.822500000000000231e+00,
       }},
      {30,
       {
           -1.147111618806319289e+00, -7.511828387779710958e-01,
           -3.552540587496231805e-01, -7.511828387779710958e-01,
           -3.552540587496231805e-01, -7.511828387779710958e-01,
           -1.147111618806319289e+00, -7.511828387779710958e-01,
           -3.552540587496231805e-01, -7.511828387779710958e-01,
           -3.552540587496231805e-01, -7.511828387779710958e-01,
           -2.195889101800662768e+00, -1.644167807464051112e+00,
           -1.092446513127439234e+00, -1.644167807464051112e+00,
           -1.092446513127439234e+00, -1.644167807464051112e+00,
           -2.195889101800662768e+00, -1.644167807464051112e+00,
           -1.092446513127439234e+00, -1.644167807464051112e+00,
           -1.092446513127439234e+00, -1.644167807464051112e+00,
           0.000000000000000000e+00,  0.000000000000000000e+00,
           0.000000000000000000e+00,  0.000000000000000000e+00,
           0.000000000000000000e+00,  0.000000000000000000e+00,
       },
       {
           -3.215189030697376893e+01, -2.700998402349732430e+01,
           -2.186807774002087967e+01, -2.700998402349732430e+01,
           -2.186807774002087967e+01, -2.700998402349732430e+01,
           -3.375346767395824088e+01, -2.878014848174331775e+01,
           -2.380682928952839461e+01, -2.878014848174331775e+01,
           -2.380682928952839461e+01, -2.878014848174331775e+01,
           -3.215189030697376893e+01, -2.700998402349732430e+01,
           -2.186807774002087967e+01, -2.700998402349732430e+01,
           -2.186807774002087967e+01, -2.700998402349732430e+01,
           -3.375346767395824088e+01, -2.878014848174331775e+01,
           -2.380682928952839461e+01, -2.878014848174331775e+01,
           -2.380682928952839461e+01, -2.878014848174331775e+01,
           2.460577021352918337e+00,  1.233404363514767565e+00,
           4.258242959213968115e-01,  1.233404363514767565e+00,
           4.258242959213968115e-01,  1.233404363514767565e+00,
       },
       {
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           -1.520000000000000018e+00, -1.520000000000000018e+00,
           0.000000000000000000e+00,  0.000000000000000000e+00,
           0.000000000000000000e+00,  0.000000000000000000e+00,
           0.000000000000000000e+00,  0.000000000000000000e+00,
       },
       {
           1.520000000000000018e+00, 1.520000000000000018e+00,
           1.520000000000000018e+00, 1.520000000000000018e+00,
           1.520000000000000018e+00, 1.520000000000000018e+00,
           1.520000000000000018e+00, 1.520000000000000018e+00,
           1.520000000000000018e+00, 1.520000000000000018e+00,
           1.520000000000000018e+00, 1.520000000000000018e+00,
           1.520000000000000018e+00, 1.520000000000000018e+00,
           1.520000000000000018e+00, 1.520000000000000018e+00,
           1.520000000000000018e+00, 1.520000000000000018e+00,
           1.520000000000000018e+00, 1.520000000000000018e+00,
           1.520000000000000018e+00, 1.520000000000000018e+00,
           1.520000000000000018e+00, 1.520000000000000018e+00,
           9.801000000000001933e-01, 8.100000000000000533e-01,
           8.100000000000000533e-01, 8.100000000000000533e-01,
           1.822500000000000231e+00, 1.822500000000000231e+00,
       }}};

  for (const auto &lp : lpinfo) {
    constraint.resize(lp.sz);
    profile.InitSolver(10, constraint.size());
    constraint.a_coefficient() = Map(lp.a.data(), lp.a.size());
    constraint.b_coefficient() = Map(lp.b.data(), lp.b.size());
    constraint.upper() = Map(lp.upper.data(), lp.upper.size());
    constraint.lower() = Map(lp.lower.data(), lp.lower.size());

    profile.FindMaxSd2BruteForce(constraint, &sd2max_ref, &sddmax_ref,
                                 &sd2zero_ref);
    profile.FindMaxSd2Simplex(constraint, &sd2max, &sddmax, &sd2zero);
    ASSERT_NEAR(sd2max_ref, sd2max, 1e-8);
    ASSERT_NEAR(sd2zero_ref, sd2zero, 1e-8);
  }
}

}  // namespace
}  // namespace trajectory_planning

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

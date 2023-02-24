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
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"

namespace trajectory_planning {
namespace {
// Log text to stderr with a given priority tag.
// *Note* This redirects to fprintf and isn't real-time safe!
//        If real-time safety is a requirement, replace this with a
//        call to a real-time safe logging facility!
void LogText(const char* filename, int line,
             const char tag,
             const char* format,
             ...){
  fprintf (stderr, "[%c %s:%d]", tag, filename, line);
  va_list args;
  va_start (args, format);
  vfprintf (stderr, format, args);
  va_end (args);
  fprintf(stderr,"\n");
}
}

#define LOGI(fmt, ...) LogText(__FILE__, __LINE__, 'I', fmt, ##__VA_ARGS__)
#define LOGE(fmt, ...) LogText(__FILE__, __LINE__, 'E', fmt, ##__VA_ARGS__)

const char *TimeOptimalPathProfile::Boundary::ToString(const uint8_t type) {
#define MAKE_CASE(x) \
  case Boundary::x:  \
    return #x
  switch (type) {
    MAKE_CASE(kNone);
    MAKE_CASE(kSource);
    MAKE_CASE(kSink);
    MAKE_CASE(kTrajectory);
    default:
      return "invalid enum";
  }
#undef MAKE_CASE
}

const char *TimeOptimalPathProfile::ToString(const SolverState &state) {
#define MAKE_CASE(x)   \
  case SolverState::x: \
    return #x
  switch (state) {
    MAKE_CASE(kInvalidState);
    MAKE_CASE(kAllocated);
    MAKE_CASE(kProblemDefined);
    MAKE_CASE(kProblemSolved);
    default:
      return "invalid enum";
  }
#undef MAKE_CASE
}

TimeOptimalPathProfile::DebugVerbosity TimeOptimalPathProfile::debug_ =
    DebugVerbosity::kNoOutput;

void TimeOptimalPathProfile::SetDebugVerbosity(int level) {
  if ((level >= DebugVerbosity::kNoOutput) && (level <= DebugVerbosity::kAll)) {
    debug_ = static_cast<DebugVerbosity>(level);
  } else {
    LOGE("Invalid  argument: level= %d, but should be in [%d, %d].", level,
         DebugVerbosity::kNoOutput, DebugVerbosity::kAll);
  }
}

void TimeOptimalPathProfile::DebugLog(DebugVerbosity log_level, const char *fmt,
                                      ...) {
  if (debug_ >= log_level) {
    va_list ap;
    va_start(ap, fmt);
    LOGI(fmt, ap);
    va_end(ap);
  }
}

void TimeOptimalPathProfile::PrintProblemDebugInfo() {
  LOGI(
      "--- Problem description --- \n"
      "num_samples= %d; nuconstraints_= %d"
      "solver_state= %s \n",
      num_samples_, num_constraints_, ToString(solver_state_));

  for (int sample = 0; sample < num_samples_; sample++) {
    LOGI(" --- Sample %d ---\n ", sample);
    LOGI("A= {");
    for (int c = 0; c < num_constraints_; c++) {
      LOGI("%e, ", constraints_[sample].a_coefficient(c));
    }
    LOGI("};\n");
    LOGI("B= {");
    for (int c = 0; c < num_constraints_; c++) {
      LOGI("%e, ", constraints_[sample].b_coefficient(c));
    }
    LOGI("};\n");
    LOGI("upper= {");
    for (int c = 0; c < num_constraints_; c++) {
      LOGI("%e, ", constraints_[sample].upper(c));
    }
    LOGI("};\n");
    LOGI("lower= {");
    for (int c = 0; c < num_constraints_; c++) {
      LOGI("%e, ", constraints_[sample].lower(c));
    }
    LOGI("};\n");
  }
}

bool TimeOptimalPathProfile::InitSolver(int num_samples,
                                        int num_constraints) {
  num_constraints_ = num_constraints;
  num_samples_ = num_samples;
  constraints_.resize(num_samples_);
  for (auto &c : constraints_) {
    c.resize(num_constraints_);
  }

  boundary_.resize(num_samples_);
  time_.resize(num_samples_);
  s_.resize(num_samples_);
  sd2_.resize(num_samples_);
  sd_.resize(num_samples_);
  sdd_.resize(num_samples_);
  solution_type_.resize(num_samples_);
  constr_work_.resize(num_constraints_);
  index_value_work_.reserve(num_samples_);
  constraint_set_.reserve(2 * num_constraints_);
  active_set_.reserve(2 * num_constraints_);
  last_extremal_index_ = 0;
  solver_state_ = SolverState::kAllocated;

  return true;
}

bool TimeOptimalPathProfile::SetupProblem(
    const std::vector<Constraint> &constraints, const Scalar s_start,
    const Scalar s_end, const Scalar sd_start, const Scalar sdd_start,
    const Scalar time_start) {
  if (static_cast<int>(constraints.size()) != num_samples_) {
    LOGE("Wrong sampling dimension for constraints.");
    return false;
  }
  for (const auto &c : constraints) {
    if (c.size() != num_constraints_) {
      LOGE("Constraint size error.");
      return false;
    }
    constr_work_ = c.upper() - c.lower();
    if (constr_work_.maxCoeff() <= 0) {
      LOGE("Infeasible bounds, at least one upper limit not > lower limit:");
      for (int idx = 0; idx < num_constraints_; idx++) {
        LOGE("Constraint %d: upper= %e; lower= %e.", idx, c.upper(idx),
             c.lower(idx));
      }
      return false;
    }
  }

  if (s_start >= s_end) {
    LOGE("s_start must be < s_end (got %e < %e; diff= %.e)", s_start, s_end,
         s_end - s_start);
    return false;
  }
  if (sd_start < 0) {
    LOGE("sd_start must but >=0 (got %e)", sd_start);
    return false;
  }

  constraints_ = constraints;
  s_end_ = s_end;
  s_start_ = s_start;
  sd_start_ = sd_start;
  sdd_start_ = sdd_start;
  time_start_ = time_start;

  return SetSetupDone();
}

void TimeOptimalPathProfile::SetMaxNumSolverLoops(const int num_loops) {
  max_num_loops_ = num_loops;
}

bool TimeOptimalPathProfile::PlotAllExtremals(const char *filename) {
  Eigen::ArrayX<Scalar> old_sd2(sd2_);
  Eigen::ArrayX<Scalar> old_sdd(sdd_);
  char name[64];
  int end_idx;
  FILE *fp = fopen(filename, "w");
  if (nullptr == fp) {
    LOGE("Error opening %s (%s).", filename, strerror(errno));
    return false;
  }

  // plot boundary curve
  for (int idx = 0; idx < num_samples_; idx++) {
    fprintf(fp,
            "boundary-curve  idx= %d s= %e sd2max= %e sdd_min= %e sdd_max= %e "
            "type= %d sd2max_zero= %e\n",
            idx, s_[idx], boundary_.sd2_max[idx],
            boundary_.sdd_min_for_sd2_max[idx],
            boundary_.sdd_max_for_sd2_max[idx], boundary_.type[idx],
            boundary_.sd2_max_for_sdd0[idx]);
  }

  for (int start_idx = 0; start_idx < num_samples_; start_idx++) {
    // 1. kForward extremals starting at start_idx
    sd2_ = std::numeric_limits<Scalar>::quiet_NaN();
    sd2_[start_idx] = 0.0;
    sd2_[num_samples_ - 1] = 0.0;

    fprintf(fp, "new forward=extremal %.3d starting at sd2_max= %e\n",
            start_idx, sd2_[start_idx]);

    end_idx = AddForwardExtremal(start_idx);

    snprintf(name, sizeof(name), "plot-extremal-forward-%.3d", start_idx);
    Plot(fp, name, start_idx, end_idx);

    fprintf(fp, "extremal=forward %.3d : got end_idx= %d\n", start_idx,
            end_idx);

    // 2. backward extremals starting at start_idx
    sd2_ = std::numeric_limits<Scalar>::quiet_NaN();
    sd2_[start_idx] = 0.0;
    sd2_[0] = 0.0;

    // extremal[start_idx] = boundary.sd2_max[start_idx];
    fprintf(fp, "new backward=extremal %.3d starting at sd2_max= %e\n",
            start_idx, sd2_[start_idx]);

    end_idx = AddBackwardExtremal(start_idx);
    snprintf(name, sizeof(name), "plot-extremal-backward-%.3d", start_idx);
    Plot(fp, name, end_idx, start_idx);

    fprintf(fp, "extremal=backward %.3d : got end_idx= %d\n", start_idx,
            end_idx);
  }

  fclose(fp);
  sd2_ = old_sd2;
  sdd_ = old_sdd;
  return true;
}

bool TimeOptimalPathProfile::PlotSolution(const char *filename) {
  FILE *fp = fopen(filename, "w");
  if (nullptr == fp) {
    LOGE("Error opening %s (%s).", filename, strerror(errno));
    return false;
  }

  for (int idx = 0; idx < num_samples_; idx++) {
    fprintf(fp, "t= %e s= %e %e %e\n", time_[idx], s_[idx], sd_[idx],
            sdd_[idx]);
  }

  fclose(fp);
  return true;
}

bool TimeOptimalPathProfile::OptimizePathParameter() {
  if (solver_state_ != SolverState::kProblemDefined) {
    LOGE("Error, problem not defined/set up.");
    return false;
  }

  // Initialize the solution for the squared path parameter velocities to NaN.
  // This indicates that they have not been computed yet and facilitates using
  // one array to assemble the solution from different extremal curves.
  // As new extremal (segments) are added, non-NaN values indicate previously
  // computed values from a different extremal, while NaN values indicate a
  // previously not visited section of the path.
  sd2_ = std::numeric_limits<Scalar>::quiet_NaN();
  sdd_ = std::numeric_limits<Scalar>::quiet_NaN();
  std::fill(solution_type_.begin(), solution_type_.end(), kNoSolution);
  solution_type_.front() = kForward;
  solution_type_.back() = kBackward;
  sd2_[0] = sd_start_ * sd_start_;
  sd2_[num_samples_ - 1] = 0;

  dt_max_ = 0.0;
  ////////////////////////////////
  DebugLog(kMainAlgorithm, "Calculating velocity boundary curve.\n");
  // 1. construct boundary.
  if (!CalculateBoundary()) {
    LOGE("Error in CalculateBoundary().");
    return false;
  }
  DebugLog(kMainAlgorithm, "Complete boundary calculation.\n");
  ////////////////////////////////
  // 2. iforw_lo, iforw_hi, iback_lo, iback_hi, icrit_lo, icrit_hi
  int iforw_lo = 0;
  int iback_hi = num_samples_ - 1;
  int iback_lo, iforw_hi;
  int icrit;
  int icrit_lo, icrit_hi;
  DebugLog(kMainAlgorithm, "Calculating first extremals ..\n");

  iback_lo = AddBackwardExtremal(iback_hi);
  iforw_hi = AddForwardExtremal(iforw_lo);

  icrit_hi = iback_lo;
  // Expand critical point search region upper boundary to make sure
  // critical point isn't missed if first backward extremal stopped
  // at a critical point.
  if ((iforw_hi < icrit_hi) &&
      ((icrit_hi < num_samples_ - 2) && (icrit_hi >= 2))) {
    sd2_[icrit_hi] = std::numeric_limits<Scalar>::quiet_NaN();
    icrit_hi++;
    iback_lo++;
  }
  icrit_lo = iforw_hi;
  for (int loop = 0; loop < max_num_loops_; loop++) {
    DebugLog(
        kExtremalLoop,
        "Loop= %d; iback_lo: %d, iforw_hi: %d | icrit_lo= %d icrit_hi= %d\n",
        loop, iback_lo, iforw_hi, icrit_lo, icrit_hi);
    // print current solution curve for debugging
    if (debug_ > kExtremalLoop) {
      char name[32];
      snprintf(name, sizeof(name), "sd2-%.3d", loop);
      Plot(stdout, name, 0, num_samples_ - 1);
    }

    if (iforw_hi >= icrit_hi) {
      DebugLog(kExtremalLoop,
               "Completed construction of solution from extremals.\n");
      break;
    }
    // Find critical point between icrit_lo and icrit_hi (and use value).
    icrit = NextCriticalPoint(icrit_lo, icrit_hi);
    if (icrit < 0 || icrit >= sd2_.size()) {
      // Re-start extremals between min/max search interval, at boundary
      // curve.
      icrit = 0.5 * (icrit_lo + icrit_hi);
    }
    if (icrit > 0 && icrit < sd2_.size() - 1) {
      sd2_[icrit] = boundary_.sd2_max[icrit];
    }

    DebugLog(kExtremalLoop,
             "Found critical point at %d, set sd2sol[%d]= %e ; is_traj= %d\n",
             icrit, icrit, sd2_[icrit],
             boundary_.type[icrit] & Boundary::kTrajectory);

    if (boundary_.sd2_max[icrit - 1] <= boundary_.sd2_max[icrit]) {
      iback_hi = icrit - 1;
      sd2_[icrit - 1] = boundary_.sd2_max[icrit - 1];
    } else {
      iback_hi = icrit;
    }

    iback_lo = AddBackwardExtremal(iback_hi);
    iforw_lo = icrit;
    iforw_hi = AddForwardExtremal(iforw_lo);
    // Check if we connected critical point to front part of solution.
    if (iback_lo > icrit_lo) {
      LOGE(
          "Could not connect from critical point to initial trajectory "
          "portion,\n"
          " iback_lo > icrit_lo (%d > %d); iback_hi= %d, \n"
          " increase number of samples\n",
          iback_lo, icrit_lo, iback_hi);
      return false;
    }

    DebugLog(kExtremalLoop,
             "New extremals at loop= %d: iback_lo= %d, iforw_hi= %d\n", loop,
             iback_lo, iforw_hi);
    icrit_lo = iforw_hi;
  }
  for (int idx = 0; idx < num_samples_; idx++) {
    // All sd2 values should have been computed.
    if (std::isnan(sd2_[idx])) {
      LOGE("No solution found,  sd2_[%d]= NaN", idx);
      return false;
    }
    // Some sdd values might not have been set if they are at an extremal
    // intersection. Try to compute a valid derivative and fall back to setting
    // a zero derivative if this fails. Consider modifying sd2_[idx] if
    // remaining constraint violations are an issue.
    if (std::isnan(sdd_[idx])) {
      ComputeSddAtIntersection(idx, "nan-sdd");
    }
  }

  // Enforce boundary constraint, if possible.
  if (AreDerivativesValid(0, sdd_start_, sd2_[0])) {
    sdd_[0] = sdd_start_;
  }

  ////////////////////////////////
  // find timing profile t(s) / s(t).
  sd_ = sd2_.sqrt();

  if (sd2_[num_samples_ - 1] != 0) {
    LOGE(
        "Non-zero terminal velocity not implemented, but sd[num_samples_-1]= "
        "%e\n",
        sd2_[num_samples_ - 1]);
    return false;
  }

  // Find the beginning of the final desceleration phase. This is different
  // from the index returned by AddBackwardExtremal, as it excludes any parts
  // on the velocity boundary curve.
  // Start searching at num_samples_ - 2, as the last sample always has
  // sdd == 0.0.
  last_extremal_index_ = std::max<int>(1, num_samples_ - 2);
  while (last_extremal_index_ >= 1) {
    // Last extremal extremal stops when sdd > 0 (speed increases) or sd2 is
    // on the boundary curve (at maximum speed).
    if (sdd_[last_extremal_index_] > 0.0 ||
        std::abs(sd2_[last_extremal_index_] -
                 boundary_.sd2_max[last_extremal_index_]) < kTiny) {
      break;
    }
    last_extremal_index_--;
  }

  // Integration based on assumption of piecewise linear sd^2(s) function
  // (that is, piecewise constant d(sd^2)/ds).
  // ==> rearrange and integrate(dt/ds, ds)
  time_[0] = time_start_;
  for (int idx = 1; idx < num_samples_; idx++) {
    if ((sd2_[idx - 1] > 0) || (sd2_[idx] > 0)) {
      const Scalar dt = 2.0 * ds_ / (sd_[idx - 1] + sd_[idx]);
      time_[idx] = time_[idx - 1] + dt;
      dt_max_ = std::max(dt, dt_max_);
    } else {  // both == 0
      DebugLog(kMainAlgorithm,
               "Two consecutive zero velocity points (at idx= %d).\n", idx);
      // This means that speed is zero for multiple neighboring positions, or
      // path is a point, and dt is undetermined.
      // Set 0.0 here.
      time_[idx] = time_[idx - 1];
      // enforce zero acceleration at boundary into stationary point
      sdd_[idx - 1] = 0;
      sdd_[idx] = 0;
    }
  }
  low_idx_ = 0;
  while ((time_[low_idx_] == time_[low_idx_ + 1]) &&
         (low_idx_ < num_samples_ - 2)) {
    low_idx_++;
  }
  high_idx_ = num_samples_ - 1;
  while ((time_[high_idx_] == time_[high_idx_ - 1]) &&
         (high_idx_ >= low_idx_)) {
    high_idx_--;
  }
  if (debug_ >= DebugVerbosity::kAll) {
    static int solveno = 0;
    for (int idx = 0; idx < num_samples_; idx++) {
      DebugLog(kAll, "DEBUG-SOLVE=%.3d: %d t= %e s= %e %e %e bound= %e\n",
               solveno, idx, time_[idx], s_[idx], sd_[idx], sdd_[idx],
               boundary_.sd2_max[idx]);
    }
    solveno++;
  }

  solver_state_ = SolverState::kProblemSolved;
  return true;
}

absl::Status TimeOptimalPathProfile::SolutionSatisfiesConstraints() {
  if (solver_state_ != SolverState::kProblemSolved) {
    return absl::FailedPreconditionError("No valid solution.");
  }
  int violation_count = 0;
  for (int sample = 0; sample < num_samples_; ++sample) {
    const auto &constraint = constraints_[sample];
    for (int i = 0; i < constraint.size(); i++) {
      const Scalar value = constraint.a_coefficient(i) * sdd_[sample] +
                           constraint.b_coefficient(i) * sd2_[sample];
      const bool violated = (value + kTiny < constraint.lower(i)) ||
                            (value - kTiny > constraint.upper(i));
      if (violated) {
        ++violation_count;
      }
      DebugLog(kAll,
               "DEBUG_CONSTRAINT_%.2d lower: %e upper: %e constr: %e "
               "violated: %d\n",
               i, constraint.lower(i), constraint.upper(i), value, violated);
    }
  }
  if (violation_count > 0) {
    return absl::NotFoundError(
        absl::StrCat("Number of constraint violations: ", violation_count));
  }
  return absl::OkStatus();
}

bool TimeOptimalPathProfile::SetIthConstraint(const Constraint &constr,
                                              const int i) {
  if (constr.size() != num_constraints_) {
    LOGE("Size error.");
    return false;
  }
  if (i >= num_samples_) {
    LOGE("Error, index too large.");
    return false;
  }
  constraints_[i] = constr;

  return true;
}

bool TimeOptimalPathProfile::SetSetupDone() {
  if (!IsSetupValid()) {
    return false;
  }

  ds_ = (s_end_ - s_start_) / (num_samples_ - 1);
  assert(ds_ > 0);
  inv_ds_ = Scalar(1) / ds_;

  for (int idx = 0; idx < num_samples_; idx++) {
    s_[idx] = ds_ * idx + s_start_;
  }
  s_[num_samples_ - 1] = s_end_;

  solver_state_ = SolverState::kProblemDefined;

  return true;
}

bool TimeOptimalPathProfile::IsSetupValid() const {
  // Check upper>lower
  for (const auto &c : constraints_) {
    if ((c.lower() >= c.upper()).any()) {
      LOGE("Constraints must satisfy: lower < upper.");
      return false;
    }
  }
  // Check timing parameter range
  if (s_start_ >= s_end_) {
    LOGE("Error, path parameter final value not larger starting value.");
    return false;
  }
  // Check minimum problem size
  constexpr int kMinNumSamples = 2;
  if (num_samples_ < kMinNumSamples) {
    LOGE("Error, need at least %d samples.", kMinNumSamples);
    return false;
  }

  // Add more viability checks here as needed.
  return true;
}

bool TimeOptimalPathProfile::RescaleSolution() {
  Scalar rescale = Scalar(1);
  for (int i_samp = 0; i_samp < num_samples_; i_samp++) {
    const Constraint &c = constraints_[i_samp];
    constr_work_ =
        c.a_coefficient() * sdd_[i_samp] + c.b_coefficient() * sd2_[i_samp];
    for (int i_const = 0; i_const < num_constraints_; i_const++) {
      if ((constr_work_[i_const] > c.upper(i_const) + kTiny) &&
          (std::abs(constr_work_[i_const]) > kTiny)) {
        const Scalar temp = c.upper(i_const) / constr_work_[i_const];
        DebugLog(kAll,
                 "Constraint violation at i_samp= %d, i_const= %d, upper= "
                 "%.18e, work= "
                 "%.18e scale= %.18e\n",
                 i_samp, i_const, c.upper(i_const), constr_work_[i_const],
                 temp);
        if ((temp < rescale) && (temp > 0)) {
          rescale = temp;
        }
      }
      if ((constr_work_[i_const] < c.lower(i_const) - kTiny) &&
          (std::abs(constr_work_[i_const]) > kTiny)) {
        const Scalar temp = c.lower(i_const) / constr_work_[i_const];
        DebugLog(kAll,
                 "Constraint violation at i_samp= %d, i_const= %d, lower= "
                 "%.18e, work= "
                 "%.18e, scale= %.18e\n",
                 i_samp, i_const, c.upper(i_const), constr_work_[i_const],
                 temp);
        if ((temp < rescale) && (temp > 0)) {
          rescale = temp;
        }
      }
    }
  }

  if (rescale < Scalar(1)) {
    const Scalar sd2_0 = sd2_[0];
    sdd_ *= rescale;
    sd2_ *= rescale;
    DebugLog(kAll, "Rescaled: sd2_0: %.18e, sd2[0]: %.18e (rescale= %.18e)\n",
             sd2_0, sd2_[0], rescale);
  }
  return true;
}

bool TimeOptimalPathProfile::AreDerivativesValid(const int idx,
                                                 const Scalar sdd,
                                                 const Scalar sd2) const {
  assert(idx < static_cast<int>(constraints_.size()));
  const Constraint &c = constraints_[idx];
  for (int i = 0; i < num_constraints_; i++) {
    const Scalar v = c.a_coefficient(i) * sdd + c.b_coefficient(i) * sd2;
    if (v + kTiny < c.lower(i) || v - kTiny > c.upper(i)) {
      return false;
    }
  }
  return true;
}

TimeOptimalPathProfile::Scalar TimeOptimalPathProfile::FindSddMax(
    const int idx, const Scalar sd2) const {
  assert(idx < static_cast<int>(constraints_.size()));
  Scalar sdd = std::numeric_limits<Scalar>::lowest();
  const Constraint &c = constraints_[idx];

  for (int i = 0; i < num_constraints_; i++) {
    const Scalar &A = c.a_coefficient(i);
    const Scalar &B = c.b_coefficient(i);
    const Scalar &lower = c.lower(i);
    const Scalar &upper = c.upper(i);

    if (!IsTiny(A)) {
      Scalar sddi = (lower - B * sd2) / A;
      if ((sddi > sdd) && AreDerivativesValid(idx, sddi, sd2)) {
        sdd = sddi;
      }
      sddi = (upper - B * sd2) / A;
      if ((sddi > sdd) && AreDerivativesValid(idx, sddi, sd2)) {
        sdd = sddi;
      }
    }
  }

  if (sdd == std::numeric_limits<Scalar>::lowest()) {
    sdd = 0;
  }
  return sdd;
}

TimeOptimalPathProfile::Scalar TimeOptimalPathProfile::FindSddMin(
    const int idx, const Scalar sd2) const {
  assert(idx < static_cast<int>(constraints_.size()));
  Scalar sdd = std::numeric_limits<Scalar>::max();
  const Constraint &constr = constraints_[idx];

  for (int i = 0; i < num_constraints_; i++) {
    const Scalar &A = constr.a_coefficient(i);
    const Scalar &B = constr.b_coefficient(i);
    const Scalar &lower = constr.lower(i);
    const Scalar &upper = constr.upper(i);

    if (!IsTiny(A)) {
      Scalar sddi = (lower - B * sd2) / A;
      if ((sddi < sdd) && AreDerivativesValid(idx, sddi, sd2)) {
        sdd = sddi;
      }
      sddi = (upper - B * sd2) / A;
      if ((sddi < sdd) && AreDerivativesValid(idx, sddi, sd2)) {
        sdd = sddi;
      }
    }
  }
  if (sdd == std::numeric_limits<Scalar>::max()) {
    sdd = 0;
  }
  return sdd;
}

int TimeOptimalPathProfile::NextCriticalPoint(const int idx_lo,
                                              const int idx_hi) const {
  assert(idx_lo <= idx_hi);
  assert(idx_hi < boundary_.type.size());
  int crit = -1;
  for (int idx = idx_lo + 1; idx <= idx_hi; idx++) {
    if (crit < 0) {
      if ((boundary_.type[idx] & Boundary::kSource) ||
          (boundary_.type[idx] & Boundary::kTrajectory)) {
        crit = idx;
      }
    } else {
      // Check for isolated points
      if (boundary_.sd2_max[idx] == boundary_.sd2_max_for_sdd0[0]) {
        crit = idx;
      }
    }
    // valid trajectory piece found? ++> done
    if ((crit > 0) && (!std::isnan(sd2_[idx]))) {
      return crit;
    }
  }
  return -1;
}

void TimeOptimalPathProfile::ComputeSddAtIntersection(int index,
                                                      const char *msg) {
  // Find a valid sdd = d^2(s)/dt^2 by trying different finite difference
  // approximations (forward, backward, symmetric).
  // The finite differences are applied to sd(s) = (d(s)/dt)(s).
  // As we need sdd, not d(sd^2)/ds, and because d((ds/dt)^2)/ds = 2*d^2(s)/dt^2
  // (eq (6) in the https://doi.org/10.1109/ROBOT.1986.1087500), there is an
  // additional factor of 1/2 to consider.
  std::vector<Scalar> sdd_candidates;
  sdd_candidates.reserve(3);
  if (index > 0 && index < sdd_.size() - 1) {
    sdd_candidates.push_back(0.25 / ds_ * (sd2_[index + 1] - sd2_[index - 1]));
  }
  if (index < sdd_.size() - 1) {
    sdd_candidates.push_back(0.5 / ds_ * (sd2_[index + 1] - sd2_[index]));
  }
  if (index > 0) {
    sdd_candidates.push_back(0.5 / ds_ * (sd2_[index] - sd2_[index - 1]));
  }
  sdd_[index] = Scalar{0};
  for (const auto &sdd : sdd_candidates) {
    if (AreDerivativesValid(index, sdd, sd2_[index])) {
      sdd_[index] = sdd;
      return;
    }
  }
  // TODO If none of the derivative candidates is valid, the
  // solution can still violate constraints. If that becomes an issue, the
  // sd2 values might need to be modified outward from the intersection.
}

std::pair<TimeOptimalPathProfile::Scalar, TimeOptimalPathProfile::Scalar>
TimeOptimalPathProfile::OneForwardExtremalStep(int index, Scalar sd2) {
  const Scalar sdd = FindSddMax(index, sd2);
  // 2.0*ds_, because d((ds/dt)^2)/ds = 2*d^2(s)/dt^2 (eq (6) in the
  // https://doi.org/10.1109/ROBOT.1986.1087500).
  return {sdd, sd2 + 2.0 * ds_ * sdd};
}

std::pair<TimeOptimalPathProfile::Scalar, TimeOptimalPathProfile::Scalar>
TimeOptimalPathProfile::OneBackwardExtremalStep(int index, Scalar sd2) {
  const Scalar sdd = FindSddMin(index, sd2);
  // 2.0*ds_, because d((ds/dt)^2)/ds = 2*d^2(s)/dt^2 (eq (6) in the
  // https://doi.org/10.1109/ROBOT.1986.1087500).
  return {sdd, sd2 - 2.0 * ds_ * sdd};
}

int TimeOptimalPathProfile::AddForwardExtremal(const int idx_lo) {
  int idx = idx_lo;
  Scalar sd2tmp = std::numeric_limits<Scalar>::quiet_NaN();
  Scalar sddtmp = std::numeric_limits<Scalar>::quiet_NaN();

  for (; idx < num_samples_ - 2; idx++) {
    assert(!std::isnan(sd2_[idx]));
    assert((boundary_.type[idx] != Boundary::kNone) || (idx == 0) ||
           (idx == num_samples_ - 1));
    const bool on_boundary = IsTiny(sd2_[idx] - boundary_.sd2_max[idx]);
    if (on_boundary && ((boundary_.type[idx] & Boundary::kTrajectory) &&
                        (idx < num_samples_ - 1) &&
                        (boundary_.type[idx + 1] & Boundary::kTrajectory))) {
      DebugLog(kExtremalDetail,
               "forward-extremal: idx= %d ; on boundary, this & next "
               "boundary point class TRAJ\n",
               idx);
      sd2tmp = boundary_.sd2_max[idx + 1];
      sddtmp = 0.5 * (sd2tmp - sd2_[idx]) / ds_;
    } else {
      std::tie(sddtmp, sd2tmp) = OneForwardExtremalStep(idx, sd2_[idx]);
      DebugLog(kExtremalDetail,
               "forward-extremal: idx= %d ; not on boundary, normal forward "
               "integration case, sdd = %e\n",
               idx, sddtmp);
    }

    assert(!std::isnan(sd2tmp));

    // Check for intersection.
    if (!std::isnan(sd2_[idx + 1]) && (sd2_[idx + 1] < sd2tmp)) {
      ComputeSddAtIntersection(idx, "forward-extremal");
      return num_samples_ - 1;
    }

    // Check if we are exceeding boundary
    if (sd2tmp > boundary_.sd2_max[idx + 1]) {
      const Scalar sdd_bound =
          0.5 * (boundary_.sd2_max[idx + 1] - sd2_[idx]) / ds_;
      const bool deriv_invalid =
          !AreDerivativesValid(idx, sdd_bound, boundary_.sd2_max[idx]);
      const bool type_invalid = boundary_.type[idx + 1] & Boundary::kSink;
      if (type_invalid || deriv_invalid) {
        DebugLog(kExtremalControl,
                 "forward-extremal: stopping forward extremal at idx= %d\n",
                 idx + 1);
        DebugLog(kExtremalControl,
                 "forward-extremal: sd2tmp= %e, bound= %e, sdd_bound= %e, "
                 "type= %d "
                 "type_invalid= %d deriv_invalid= %d\n",
                 sd2tmp, boundary_.sd2_max[idx + 1], sdd_bound,
                 boundary_.type[idx + 1], type_invalid, deriv_invalid);
        return idx;
      } else {
        // source or trajectory, and snapping to boundary produces valid
        // velocity/acceleration point
        DebugLog(kExtremalDetail,
                 "forward-extremal: idx= %d, boundary class SOURCE or TRAJ: "
                 "snapping "
                 "to boundary sd2= %e\n",
                 idx, sd2tmp);
        sd2tmp = boundary_.sd2_max[idx + 1];
        sddtmp = sdd_bound;
      }
    }

    // Check for negative values -- this probably indicates that we don't have
    // a fine enough discretization
    if (sd2tmp < 0) {
      LOGI(
          "Negative sd2 at idx= %d -- increase number of samples (sd2= "
          "%.18e)?",
          idx, sd2tmp);
      sd2tmp = 0.0;
      if (idx <= 1) {
        sddtmp = 0.0;
      } else {
        sddtmp = -sd2_[idx - 1] / ds_;
      }
    }

    DebugLog(kExtremalDetail, "forward-extremal-point: %d %e\n", idx + 1,
             sd2tmp);
    sd2_[idx + 1] = sd2tmp;
    solution_type_[idx + 1] = kForward;
    sdd_[idx] = sddtmp;
  }
  return num_samples_ - 1;
}

int TimeOptimalPathProfile::AddBackwardExtremal(const int idx_hi) {
  int idx = idx_hi;
  Scalar sd2tmp = std::numeric_limits<Scalar>::quiet_NaN();
  Scalar sddtmp = std::numeric_limits<Scalar>::quiet_NaN();

  for (; idx > 1; idx--) {
    assert(!std::isnan(sd2_[idx]));
    assert((boundary_.type[idx] != Boundary::kNone) || (idx == 0) ||
           (idx == num_samples_ - 1));
    const bool on_boundary = IsTiny(sd2_[idx] - boundary_.sd2_max[idx]);
    if (on_boundary &&
        ((boundary_.type[idx] & Boundary::kTrajectory) && (idx > 0) &&
         (boundary_.type[idx - 1] & Boundary::kTrajectory))) {
      DebugLog(kExtremalDetail,
               "backward-extremal: idx= %d ; on boundary, this & next  "
               "boundary point class TRAJ\n",
               idx);
      sd2tmp = boundary_.sd2_max[idx - 1];
      sddtmp = 0.5 * (sd2_[idx] - sd2tmp) / ds_;
    } else {
      std::tie(sddtmp, sd2tmp) = OneBackwardExtremalStep(idx, sd2_[idx]);
      DebugLog(kExtremalDetail,
               "backward-extremal: idx= %d ; not on boundary, normal backward "
               "integration "
               "case, sdd= %e\n",
               idx, sddtmp);
    }

    assert(!std::isnan(sd2tmp));

    // Check for intersection
    if (!std::isnan(sd2_[idx - 1]) && (sd2_[idx - 1] < sd2tmp)) {
      ComputeSddAtIntersection(idx, "backward-extremal");
      return 0;
    }

    // Check if we are exceeding boundary
    if (sd2tmp > boundary_.sd2_max[idx - 1]) {
      const Scalar sdd_bound =
          0.5 * (sd2_[idx] - boundary_.sd2_max[idx - 1]) / ds_;
      const bool deriv_invalid =
          !AreDerivativesValid(idx, sdd_bound, sd2_[idx]);
      const bool type_invalid = boundary_.type[idx - 1] & Boundary::kSource;
      // Only stop backward extremal if not connecting to critical point.
      // If connecting to a critical point, there exists a backward extremal
      // to the previous trajectory segment and exceeding the velocity
      // boundary is caused by discretization errors. So in that case snap to
      // boundary curve and accept the associated constraint violation (which
      // is related to the discretization)
      const bool is_connecting = (idx_hi != (num_samples_ - 1));
      if ((type_invalid || deriv_invalid) && !is_connecting) {
        DebugLog(kExtremalControl,
                 "backward-extremal: stopping extremal backward idx= %d\n",
                 idx - 1);
        DebugLog(kExtremalControl,
                 "backward-extremal: sd2tmp= %e, bound= %e, delta= %e, "
                 "sdd_bound= %e "
                 "type= %d type_invalid= %d deriv_invalid= %d\n",
                 sd2tmp, boundary_.sd2_max[idx - 1],
                 boundary_.sd2_max[idx - 1] - sd2tmp, sdd_bound,
                 boundary_.type[idx - 1], type_invalid, deriv_invalid);
        return idx;
      } else {
        // sink or trajectory, and snapping to  boundary produces valid
        // velocity/acceleration point
        DebugLog(kExtremalDetail,
                 "backward-extremal: idx= %d, boundary class SINK or "
                 "TRAJ, snapping to boundary sd2= %e\n",
                 idx, sd2tmp);
        sd2tmp = boundary_.sd2_max[idx - 1];
        sddtmp = sdd_bound;
      }
    }

    // Check for negative values -- this probably indicates that we don't have
    // a fine enough discretization.
    if (sd2tmp < 0) {
      LOGI("Negative sd2 idx= %d -- increase number of samples?\n", idx);
      sd2tmp = 0.0;
      if (idx < sd2_.size() - 1) {
        sddtmp = sd2_[idx + 1] / ds_;
      } else {
        sddtmp = 0.0;
      }
    }

    DebugLog(kExtremalDetail, "backward-extremal-point: %d %e\n", idx - 1,
             sd2tmp);
    sd2_[idx - 1] = sd2tmp;
    solution_type_[idx - 1] = kBackward;
    sdd_[idx] = sddtmp;
  }
  return 0;
}

bool TimeOptimalPathProfile::Intersect(const Scalar &A1, const Scalar &B1,
                                       const Scalar &e1, const Scalar &A2,
                                       const Scalar &B2, const Scalar &e2,
                                       Scalar *sdd, Scalar *sp2) const {
  assert(nullptr != sdd);
  assert(nullptr != sp2);
  const Scalar det = A1 * B2 - B1 * A2;
  if (IsTiny(det)) {
    // linearly dependent
    if (IsTiny(A1)) {  // reason: acceleration independence. choose sdd=0
      *sdd = 0;
      if (IsTiny(B1)) {  // singular equation, all parameters zero
        return false;
      }
      *sp2 = e1 / B1;
      return true;
    } else {
      // parallel constraints : no finite maximum
      // (can also happen for very small coefficients, when using very dense
      // sampling)
      return false;
    }
  }
  const Scalar inv_det = Scalar(1) / det;
  *sdd = (B2 * e1 - B1 * e2) * inv_det;
  *sp2 = (-A2 * e1 + A1 * e2) * inv_det;
  return true;
}

void TimeOptimalPathProfile::PrintFindMaxSd2MathematicaDebugCode(
    const Constraint &constr) const {
  LOGI("RegionPlot[");
  for (int c = 0; c < num_constraints_; c++) {
    LOGI("%.18f*sdd %+f*sp2<=%.18f && %.18f*sdd %+f*sp2>=%.18f ",
         constr.a_coefficient(c), constr.b_coefficient(c), constr.upper(c),
         constr.a_coefficient(c), constr.b_coefficient(c), constr.lower(c));
    if (c < num_constraints_ - 1) {
      LOGI(" && ");
    }
  }
  LOGI(
      ", {sp2, 0, 2}, {sdd, -2, 2}, PlotLegends -> All, FrameLabel -> "
      "{sd2, sdd}]\n");

  LOGI("Maximize[x,");
  for (int c = 0; c < num_constraints_; c++) {
    LOGI("%.18f*y %+f*x<=%.18f && %.18f*y %+f*x>=%.18f ",
         constr.a_coefficient(c), constr.b_coefficient(c), constr.upper(c),
         constr.a_coefficient(c), constr.b_coefficient(c), constr.lower(c));
    if (c < num_constraints_ - 1) {
      LOGI(" && ");
    }
  }
  LOGI(", {x, y}]\n");
}

void TimeOptimalPathProfile::FindMaxSd2BruteForce(const Constraint &constr,
                                                  Scalar *sd2max,
                                                  Scalar *sddmax,
                                                  Scalar *sd2zero) {
  assert(nullptr != sd2zero);
  assert(nullptr != sddmax);
  assert(nullptr != sd2zero);

  // this is a very simple, brute-force implementation ..
  // room for performance optimizations.
  *sd2max = 0;
  *sddmax = 0;
  // this prints Mathematica code for visualizing viable parameters and
  // finding the minimum
  if (debug_ >= kAll) {
    PrintFindMaxSd2MathematicaDebugCode(constr);
  }
  *sd2zero = kMaxSd2;
  for (int c = 0; c < num_constraints_; c++) {
    assert(c < constr.size());
    if (constr.b_coefficient(c) > kTiny) {
      assert(c < constr.size());
      const Scalar tmp = constr.upper(c) / constr.b_coefficient(c);
      if (tmp < *sd2zero) {
        *sd2zero = tmp;
      }
    } else if (constr.b_coefficient(c) < -kTiny) {
      assert(c < constr.size());
      const Scalar tmp = constr.lower(c) / constr.b_coefficient(c);
      if (tmp < *sd2zero) {
        *sd2zero = tmp;
      }
    }
  }

  {
    // brute force: compute all intersections;
    for (int c1 = 0; c1 < num_constraints_; c1++) {
      for (int c2 = c1 + 1; c2 < num_constraints_; c2++) {
        Scalar sd2, sdd;
        // upper/upper
        if (Intersect(constr.a_coefficient(c1), constr.b_coefficient(c1),
                      constr.upper(c1), constr.a_coefficient(c2),
                      constr.b_coefficient(c2), constr.upper(c2), &sdd, &sd2)) {
          if ((sd2 > *sd2max) &&
              ConstraintIsValid(constr, sdd, sd2, &constr_work_)) {
            *sd2max = sd2;
            *sddmax = sdd;
          }
        }
        // upper/lower
        if (Intersect(constr.a_coefficient(c1), constr.b_coefficient(c1),
                      constr.upper(c1), constr.a_coefficient(c2),
                      constr.b_coefficient(c2), constr.lower(c2), &sdd, &sd2)) {
          if ((sd2 > *sd2max) &&
              ConstraintIsValid(constr, sdd, sd2, &constr_work_)) {
            *sd2max = sd2;
            *sddmax = sdd;
          }
        }

        // lower/upper
        if (Intersect(constr.a_coefficient(c1), constr.b_coefficient(c1),
                      constr.lower(c1), constr.a_coefficient(c2),
                      constr.b_coefficient(c2), constr.upper(c2), &sdd, &sd2)) {
          if ((sd2 > *sd2max) &&
              ConstraintIsValid(constr, sdd, sd2, &constr_work_)) {
            *sd2max = sd2;
            *sddmax = sdd;
          }
        }

        // lower/lower
        if (Intersect(constr.a_coefficient(c1), constr.b_coefficient(c1),
                      constr.lower(c1), constr.a_coefficient(c2),
                      constr.b_coefficient(c2), constr.lower(c2), &sdd, &sd2)) {
          if ((sd2 > *sd2max) &&
              ConstraintIsValid(constr, sdd, sd2, &constr_work_)) {
            *sd2max = sd2;
            *sddmax = sdd;
          }
        }
      }
    }
    if (0 == *sd2max || *sd2max > kMaxSd2) {
      *sd2max = kMaxSd2;
      *sddmax = 0;
    }
  }

  if (0 == *sd2zero) {
    *sd2zero = kMaxSd2;
  }
}

bool TimeOptimalPathProfile::IsOptimal(const Constraint &constr,
                                       const int first_index,
                                       const int second_index,
                                       const ConstraintType first_type,
                                       const ConstraintType second_type) {
  // Returns true if the optimality conditions (KKT) for
  // sd2 -> max
  // A1*sdd + B1*sd2 <> limit1
  // A2*sdd + B2*sd2 <> limit2
  // Augmented lagrangian
  // L= sd2 +
  //   lambda1*(A1*sdd+B1*sd2 - limit1) +
  //   lambda2*(A2*sdd+B2*sd2 -
  // The following solves for lambda1, lambda2 and checks for lambda > 0 or <0
  // depending on the type of inequality.

  const Scalar denom =
      constr.a_coefficient(second_index) * constr.b_coefficient(first_index) -
      constr.a_coefficient(first_index) * constr.b_coefficient(second_index);

  if (std::abs(denom) < kTiny) {
    // Parallel constaints: not optimal.
    return false;
  }
  if (first_type == kUpper) {
    if (second_type == kUpper) {
      // upper/upper
      return denom * constr.a_coefficient(first_index) <= 0 &&
             denom * (-constr.a_coefficient(second_index)) <= 0;
    }
    // upper/lower
    return denom * constr.a_coefficient(first_index) >= 0 &&
           denom * (-constr.a_coefficient(second_index)) <= 0;
  }
  if (second_type == kUpper) {
    // lower/upper
    return denom * constr.a_coefficient(first_index) <= 0 &&
           denom * (-constr.a_coefficient(second_index)) >= 0;
  }
  // lower/lower
  return denom * constr.a_coefficient(first_index) >= 0 &&
         denom * (-constr.a_coefficient(second_index)) >= 0;
}

void TimeOptimalPathProfile::FindMaxSd2Simplex(const Constraint &constr,
                                               Scalar *sd2max, Scalar *sddmax,
                                               Scalar *sd2zero) {
  // Solve the LP:
  // sd2 -> max
  // sd2 >= 0
  // lower <= A*sdd + B*sd2 <= upper
  // Using a simplex approach:
  // 1) Start at point (0,0), which is always feasible (since lower < upper)
  // 2) Search along sdd=0 for maximum feasible sd2, record active constraint
  //    set there.
  // 3) Check active set for optimality and terminate if yes.
  // 4) Search for highest admissible sd2 along constraint in active set with
  //    lowest slope.
  // 5) Add search direction to new active set. Goto 3)

  // Consider choosing minimum sdd in cases where the optimum is at a constraint
  // orthogonal to grad(sd2).
  assert(nullptr != sd2max);
  assert(nullptr != sddmax);
  assert(nullptr != sd2zero);
  *sd2max = 0;
  *sddmax = 0;

  // Put all constraints into constraint set.
  constraint_set_.resize(2 * num_constraints_);
  for (int constraint = 0; constraint < num_constraints_; constraint++) {
    constraint_set_[2 * constraint].first = constraint;
    constraint_set_[2 * constraint].second = kUpper;
    constraint_set_[2 * constraint + 1].first = constraint;
    constraint_set_[2 * constraint + 1].second = kLower;
  }

  // This is step 2)
  active_set_.clear();
  Scalar sd2 = std::numeric_limits<Scalar>::max();
  Scalar sdd = 0.0;
  for (int idx = 0; idx < num_constraints_; idx++) {
    if (std::abs(constr.b_coefficient(idx)) < kTiny) {
      continue;
    }
    // Checks either lower or upper for intersection with sdd = 0, depending
    // on sign and update active set.
    if (constr.b_coefficient(idx) > kTiny) {
      const Scalar invB = Scalar(1) / constr.b_coefficient(idx);
      const Scalar tmp = constr.upper(idx) * invB;
      if (tmp < (sd2 + kTiny) && tmp > 0) {
        if (tmp < sd2 - kTiny) {
          active_set_.clear();
        }
        active_set_.push_back(ActiveConstraint{
            idx, kUpper, std::abs(constr.a_coefficient(idx) * invB)});
        sd2 = tmp;
      }
    } else if (constr.b_coefficient(idx) < -kTiny) {
      const Scalar invB = Scalar(1) / constr.b_coefficient(idx);
      const Scalar tmp = constr.lower(idx) * invB;
      if (tmp < (sd2 + kTiny) && tmp > 0) {
        if (tmp < sd2 - kTiny) {
          active_set_.clear();
        }
        active_set_.push_back(ActiveConstraint{
            idx, kLower, std::abs(constr.a_coefficient(idx) * invB)});
        sd2 = tmp;
      }
    }
  }

  // Choose kMaxSd2 if problem is unbounded or solution is too large.
  if (sd2 > kMaxSd2 || active_set_.empty()) {
    *sd2zero = kMaxSd2;
    *sd2max = kMaxSd2;
    *sddmax = 0.0;
    return;
  }
  *sd2zero = sd2;

  // Step 3)
  // If active_set_ has >= 2 constraints, check if initial point is optimal
  // by checking all constraint pairs.
  ActiveConstraint search = active_set_.front();
  if (active_set_.size() >= 2) {
    for (size_t first = 0; first < active_set_.size(); first++) {
      // Choose lowest slope constraint (in case not optimal).
      if (active_set_[first].slope < search.slope) {
        search.index = active_set_[first].index;
        search.type = active_set_[first].type;
        search.slope = active_set_[first].slope;
      }
      for (size_t second = first + 1; second < active_set_.size(); second++) {
        if (IsOptimal(constr, active_set_[first].index,
                      active_set_[second].index, active_set_[first].type,
                      active_set_[second].type)) {
          *sd2max = sd2;
          *sddmax = 0.0;
          return;
        }
      }
    }
  }
  // Remove active set from constraint set, as already handled.
  for (auto &l : active_set_) {
    auto it = std::find(constraint_set_.begin(), constraint_set_.end(),
                        std::pair<int, int>{l.index, l.type});
    assert(it != constraint_set_.end());
    constraint_set_.erase(it);
  }

  // Limit main iteration to num_constraints_, which is more than the
  // remaining constraint_set, so this limit should never be reached.
  for (int loop = 0; loop < num_constraints_; loop++) {
    if (std::abs(constr.a_coefficient(search.index)) < kTiny) {
      // Done: constraint is orthogonal to cost function gradient.
      *sd2max = sd2;
      *sddmax = sdd;
      return;
    }
    // We're searching along the line sdd = a + b*sd2.
    const Scalar invA{Scalar(1) / constr.a_coefficient(search.index)};
    const Scalar b = -constr.b_coefficient(search.index) * invA;
    const Scalar a = search.type == kUpper ? constr.upper(search.index) * invA
                                           : constr.lower(search.index) * invA;

    // Step 4) Find new active set limiting sd2 among remaining
    // constraint_set.
    active_set_.clear();
    Scalar next_sd2 = std::numeric_limits<Scalar>::max();
    Scalar next_sdd = 0.0;
    for (auto it = constraint_set_.begin(); it != constraint_set_.end(); it++) {
      const int c = it->first;
      Scalar B{constr.a_coefficient(c) * b + constr.b_coefficient(c)};
      if (std::abs(B) < kTiny) {
        continue;
      }
      const Scalar invB = Scalar(1) / B;
      if (it->second == kUpper) {
        const Scalar tmp =
            (constr.upper(c) - constr.a_coefficient(c) * a) * invB;
        if (tmp < (next_sd2 + kTiny) && tmp > sd2) {
          if (tmp < next_sd2 - kTiny) {
            active_set_.clear();
          }
          active_set_.push_back(ActiveConstraint{
              c, kUpper, std::abs(constr.a_coefficient(c) * invB)});
          next_sd2 = tmp;
          next_sdd = a + b * next_sd2;
        }
      } else {
        const Scalar tmp =
            (constr.lower(c) - constr.a_coefficient(c) * a) * invB;
        if (tmp < (next_sd2 + kTiny) && tmp > sd2) {
          if (tmp < next_sd2 - kTiny) {
            active_set_.clear();
          }
          active_set_.push_back(ActiveConstraint{
              c, kLower, std::abs(constr.a_coefficient(c) * invB)});
          next_sd2 = tmp;
          next_sdd = a + b * next_sd2;
        }
      }
    }
    if (active_set_.empty()) {
      LOGI("Unbounded problem, choosing sd2.");
      *sd2max = *sd2zero;
      *sddmax = 0.0;
      return;
    }
    // Step 3):
    // Check {active_set_, search} for optimality and determine next search
    // direction.
    active_set_.push_back(search);
    search.index = -1;
    search.type = ConstraintType::kNotSet;
    search.slope = std::numeric_limits<Scalar>::max();
    for (size_t first = 0; first < active_set_.size(); first++) {
      // Record lowest slope in active set (but ignoring last search
      // direction).
      if (first != (active_set_.size() - 1) &&
          active_set_[first].slope < search.slope) {
        search.index = active_set_[first].index;
        search.type = active_set_[first].type;
        search.slope = active_set_[first].slope;
      }
      for (size_t second = first + 1; second < active_set_.size(); second++) {
        if (IsOptimal(constr, active_set_[first].index,
                      active_set_[second].index, active_set_[first].type,
                      active_set_[second].type)) {
          *sd2max = next_sd2;
          *sddmax = next_sdd;
          if (next_sd2 > kMaxSd2) {
            *sd2max = kMaxSd2;
            *sddmax = 0.0;
            LOGI("Maximum > kMaxS2d, saturating result.");
          }
          return;
        }
      }
    }
    // Remove last search (not in constraint_set anymore).
    active_set_.pop_back();
    for (auto &l : active_set_) {
      auto it = std::find(constraint_set_.begin(), constraint_set_.end(),
                          std::pair<int, int>{l.index, l.type});
      assert(it != constraint_set_.end());
      constraint_set_.erase(it);
    }

    sd2 = next_sd2;
    sdd = next_sdd;
  }
  // This should never happen, but avoid faulting.
  LOGE("No optimum after num_constraints itertions, using sddzero as sddmax.");
  *sd2max = *sd2zero;
  *sddmax = 0.0;
}

bool TimeOptimalPathProfile::CalculateBoundary() {
  for (int i = 0; i < num_samples_; i++) {
    Scalar sd2max = 0;
    Scalar sddmax = 0;
    Scalar sd2zero = 0;
    FindMaxSd2Simplex(constraints_[i], &sd2max, &sddmax, &sd2zero);
    boundary_.sd2_max[i] = sd2max;
    boundary_.sd2_max_for_sdd0[i] = sd2zero;
    boundary_.sdd_max_for_sd2_max[i] = FindSddMax(i, sd2max);
    boundary_.sdd_min_for_sd2_max[i] = FindSddMin(i, sd2max);
    boundary_.sd2_max_at_sdd0[i] =
        std::abs(boundary_.sd2_max[i] - boundary_.sd2_max_for_sdd0[i]) < kTiny;
  }

  boundary_.type[0] = Boundary::kNone;
  boundary_.type[num_samples_ - 1] = Boundary::kNone;
  index_value_work_.resize(0);
  for (int i = 1; i < num_samples_ - 1; i++) {
    // Find isolated critical points on the boundary curve.
    if (!boundary_.sd2_max_at_sdd0[i - 1] && boundary_.sd2_max_at_sdd0[i] &&
        !boundary_.sd2_max_at_sdd0[i + 1]) {
      boundary_.sd2_max[i - 1] = boundary_.sd2_max_for_sdd0[i - 1];
      boundary_.sdd_max_for_sd2_max[i - 1] =
          FindSddMax(i - 1, boundary_.sd2_max[i - 1]);
      boundary_.sdd_min_for_sd2_max[i - 1] =
          FindSddMin(i - 1, boundary_.sd2_max[i - 1]);
      boundary_.sd2_max[i + 1] = boundary_.sd2_max_for_sdd0[i + 1];
      boundary_.sdd_max_for_sd2_max[i + 1] =
          FindSddMax(i + 1, boundary_.sd2_max[i + 1]);
      boundary_.sdd_min_for_sd2_max[i + 1] =
          FindSddMax(i + 1, boundary_.sd2_max[i + 1]);
    }
    // Find critical points missed due to discretization.
    const Scalar sd2p =
        (boundary_.sd2_max[i + 1] - boundary_.sd2_max[i]) / (ds_);
    const Scalar sd2p_min = 2 * boundary_.sdd_min_for_sd2_max[i];
    const Scalar sd2p_max = 2 * boundary_.sdd_max_for_sd2_max[i];
    const bool boundary_is_sink_or_source =
        (sd2p < sd2p_min) || (sd2p > sd2p_max);

    const bool boundary_has_skipped_maximum_sdd =
        (boundary_.sdd_max_for_sd2_max[i] > 0) &&
        (boundary_.sdd_min_for_sd2_max[i + 1] < 0);

    const bool boundary_has_skipped_maximum_sd2 =
        (boundary_.sd2_max[i] > boundary_.sd2_max[i - 1] - kTiny) &&
        (boundary_.sd2_max[i] > boundary_.sd2_max[i + 1] - kTiny);

    // If the discrete boundary curve values around `i` indicate a maximum and
    // the boundary curve is a sink or source, reduce the boundary curve value
    // s.t. horizontal curves, as well as forward and backward extremals going
    // through it are valid. This is necessary, as discretization might miss
    // isolated critical points. In the case of false positives, the solution
    // might be sub-optimal (that is, slower than possible).
    if ((boundary_has_skipped_maximum_sd2 ||
         boundary_has_skipped_maximum_sdd) &&
        boundary_is_sink_or_source) {
      const auto [_, sd2_max_forw] =
          OneForwardExtremalStep(i - 1, boundary_.sd2_max[i - 1]);
      const auto [__, sd2_max_backw] =
          OneBackwardExtremalStep(i + 1, boundary_.sd2_max[i + 1]);
      const Scalar sd2_max =
          std::max(Scalar{0}, std::min({boundary_.sd2_max_for_sdd0[i],
                                        sd2_max_forw, sd2_max_backw}));
      index_value_work_.emplace_back(i, sd2_max);
    }
  }
  for (const auto& [index, value] : index_value_work_) {
    boundary_.sd2_max[index] = value;
    boundary_.sdd_max_for_sd2_max[index] = FindSddMax(index, value);
    boundary_.sdd_min_for_sd2_max[index] = FindSddMin(index, value);

    if (index > 0) {
      boundary_.sd2_max[index - 1] = boundary_.sd2_max_for_sdd0[index - 1];
      boundary_.sdd_max_for_sd2_max[index - 1] =
          FindSddMax(index - 1, boundary_.sd2_max[index - 1]);
      boundary_.sdd_min_for_sd2_max[index - 1] =
          FindSddMin(index - 1, boundary_.sd2_max[index - 1]);
    }
    if (index < boundary_.size() - 1) {
      boundary_.sd2_max[index + 1] = boundary_.sd2_max_for_sdd0[index + 1];
      boundary_.sdd_max_for_sd2_max[index + 1] =
          FindSddMax(index + 1, boundary_.sd2_max[index + 1]);
      boundary_.sdd_min_for_sd2_max[index + 1] =
          FindSddMin(index + 1, boundary_.sd2_max[index + 1]);
    }
  }

  for (int i = 1; i < num_samples_ - 1; i++) {
    const Scalar sd2p =
        (boundary_.sd2_max[i + 1] - boundary_.sd2_max[i]) / (ds_);
    const Scalar sd2p_min = 2 * boundary_.sdd_min_for_sd2_max[i];
    const Scalar sd2p_max = 2 * boundary_.sdd_max_for_sd2_max[i];
    DebugLog(
        kExtremalDetail,
        "velocity boundary curve %d: sd2p= %e min= %e max= %e delta= %e\n", i,
        sd2p, sd2p_min, sd2p_max, sd2p_max - sd2p_min);
    boundary_.type[i] = Boundary::kNone;
    if (sd2p < sd2p_min) {
      boundary_.type[i] = Boundary::kSink;
      DebugLog(
          kExtremalDetail,
          "velocity boundary curve point is class SINK (sd2p= %e; sd2p_min= "
          "%e)\n",
          sd2p, sd2p_min);
    } else if (sd2p > sd2p_max) {
      boundary_.type[i] = Boundary::kSource;
      DebugLog(kExtremalDetail,
               "velocity boundary curve point is class SOURCE (sd2p= %e; "
               "sd2p_max= %e)\n",
               sd2p, sd2p_max);
    }
    if ((sd2p <= sd2p_max) && (sd2p >= sd2p_min)) {
      boundary_.type[i] = Boundary::kTrajectory;
      DebugLog(kExtremalDetail,
               "velocity boundary curve point is class TRAJ (sd2p= %e; "
               "sd2p_min= %e; sd2p_max= %e)\n",
               sd2p, sd2p_min, sd2p_max);
    }
  }

  return true;
}

void TimeOptimalPathProfile::Plot(FILE *fp, const char *str, const int idx0,
                                  const int idx1) {
  for (int idx = idx0; idx <= idx1; idx++) {
    fprintf(fp, "%s: %d %.18e %.18e %.18e %.18e\n", str, idx, sd2_[idx],
            sdd_[idx], boundary_.sd2_max[idx], s_start_ + idx * ds_);
  }
}

int TimeOptimalPathProfile::SampleIndexFromTime(const Scalar t) const {
  // special-cases for out-of-range t
  if (t <= time_[0]) {
    return 0;
  }
  if (t >= time_[num_samples_ - 1]) {
    return num_samples_ - 2;
  }
  // binary search for interval
  int low = low_idx_;
  int high = high_idx_;
  int mid = (low + high) / 2;
  while (t < time_[mid] || t >= time_[mid + 1]) {
    if (t < time_[mid]) {
      high = mid;
    } else {
      low = mid;
    }
    mid = (low + high) / 2;
  }

  // if there are equal time samples, skip those.
  while ((time_[mid] == time_[mid + 1]) && (mid < num_samples_ - 1)) {
    LOGI("time_[%d] == time_[%d] ==> ++", mid, mid + 1);
    mid++;
  }
  return mid;
}

bool TimeOptimalPathProfile::ConstraintIsValid(const Constraint &constr,
                                               Scalar sdd, const Scalar sd2,
                                               Eigen::ArrayX<Scalar> *work) {
  assert(nullptr != work);
  assert(work->size() == constr.size());
  for (int i = 0; i < num_constraints_; i++) {
    const Scalar tmp =
        constr.a_coefficient(i) * sdd + constr.b_coefficient(i) * sd2;
    if (tmp + kTiny < constr.lower(i)) return false;
    if (tmp - kTiny > constr.upper(i)) return false;
  }
  return true;
}

TimeOptimalPathProfile::Scalar TimeOptimalPathProfile::GetMaxTimeIncrement()
    const {
  if (solver_state_ != SolverState::kProblemSolved) {
    LOGE("Error, solution not yet calculated!");
    return -1;
  }
  return dt_max_;
}

bool TimeOptimalPathProfile::GetPathParameterAndDerivatives(Scalar t, Scalar *s,
                                                            Scalar *sd,
                                                            Scalar *sdd) const {
  assert(nullptr != s);
  assert(nullptr != sd);
  assert(nullptr != sdd);

  if (solver_state_ != SolverState::kProblemSolved) {
    LOGE("Error, solution not yet calculated!");
    return false;
  }

  // 1. find interval
  if (t <= time_[0]) {
    t = time_[0];
    *s = s_start_;
    *sd = sd_[0];
    *sdd = 0.5 * inv_ds_ * (sd2_[1] - sd2_[0]);
    return true;
  }
  if (t >= time_[num_samples_ - 1]) {
    t = time_[num_samples_ - 1];
    *s = s_end_;
    *sd = sd_[num_samples_ - 1];
    *sdd = 0.0;
    return true;
  }
  const int k = SampleIndexFromTime(t);

  if (time_[k] == time_[k + 1]) {
    assert(k <= (time_.size() - 2));
    *s = s_[k + 1];
    *sd = sd_[k + 1];
    *sdd = 0.5 * inv_ds_ * (sd2_[k + 1] - sd2_[k]);
    return true;
  }

  const Scalar dt = t - time_[k];
  Scalar ds = 0;
  if (k > (num_samples_ - 2)) {
    LOGE("Unexpected index k= %d, num_samples_= %d", k, num_samples_);
    return false;
  }

  // 2. t calculated by integration using trapezoidal rule,
  //   so use same rule here to get to specified t from time[k],
  //   then solve for path parameter s (distance ds from s_[k]).
  const Scalar &sda = sd_[k];
  const Scalar &sdb = sd_[k + 1];
  const Scalar &sd2a = sd2_[k];
  const Scalar &sd2b = sd2_[k + 1];
  if (sda > 0 || sdb > 0) {
    ds = sda * dt + dt * dt * 0.25 * inv_ds_ * (sd2b - sd2a);
    if (ds > ds_) {
      ds = ds_;
    }
    if (dt < 0 || ds < 0) {
      LOGE("Got dt= %e; ds= %e; t= %e tk= %e tk+1= %e", dt, ds, t, time_[k],
           time_[k + 1]);
      return false;
    }
    // Saturate increment to s_[k+1] to avoid going beyond s_end_ because of
    // floating point rounding reasons.
    *s = std::min(s_[k] + ds, s_[k + 1]);
    const Scalar sqrt_arg = sd2a + ds * inv_ds_ * (sd2b - sd2a);
    assert(sqrt_arg >= 0);
    *sd = std::sqrt(sqrt_arg);
    *sdd = 0.5 * inv_ds_ * (sd2b - sd2a);
  } else {
    assert(time_[k + 1] != time_[k]);
    *s = s_[k] + (s_[k + 1] - s_[k]) * dt / (time_[k + 1] - time_[k]);
    *sd = 0.0;
    *sdd = 0.0;
  }

  assert(*s <= s_end_);

  return true;
}

int TimeOptimalPathProfile::GetPreviousIndex(const Scalar t) const {
  if (solver_state_ != SolverState::kProblemSolved) {
    LOGE("Error, solution not yet calculated!");
    return -1;
  }

  if (t < time_[0]) {
    LOGE("Got time < initial time (%e < %e: time-time_[0]= %e).", t, time_[0],
         t - time_[0]);
    return -1;
  }
  if (t > time_[num_samples_ - 1]) {
    return num_samples_ - 1;
  }

  return SampleIndexFromTime(t);
}

bool TimeOptimalPathProfile::GetPreviousDiscreteValues(const Scalar t,
                                                       Scalar *sk, Scalar *sdk,
                                                       Scalar *sddk,
                                                       Scalar *tk) const {
  assert(nullptr != sk);
  assert(nullptr != sdk);
  assert(nullptr != sddk);
  assert(nullptr != tk);

  const int k = GetPreviousIndex(t);

  if (k < 0) return false;

  assert(k >= 0);
  assert(k < num_samples_);
  *sk = s_[k];
  *sdk = sd_[k];
  *sddk = sdd_[k];
  *tk = time_[k];

  return true;
}

}  // namespace trajectory_planning

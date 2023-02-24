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

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <utility>
#include <valarray>

#include "absl/log/log.h"
#include "absl/log/check.h"
#include "absl/strings/substitute.h"


namespace trajectory_planning {

constexpr size_t kMaxNumPoints = 1000000;
constexpr size_t kMinNumPoints = 4;

namespace {
// Solve linear band-diagonal system using Thomas algorithm (spec. for spline)
// see, eg., "Numerical Mathematics," Quateroni et al.
// note: overwrites lo vector with temporary values.
struct SolveThomasArguments {
  // Lower diagonal, overwritten as scratch memory.
  std::valarray<double>* lower = nullptr;
  // Upper diagonal.
  const std::valarray<double>& upper;
  // Right hand side.
  const std::valarray<double>& rhs;
  // Solution.
  std::valarray<double>* output = nullptr;
};

void SolveThomas(SolveThomasArguments args) {
  CHECK(nullptr != args.output);
  CHECK(nullptr != args.lower);
  std::valarray<double>& lo = *args.lower;
  const std::valarray<double>& up = args.upper;
  std::valarray<double>& out = *args.output;
  const std::valarray<double>& rhs = args.rhs;
  constexpr double kDiag = 1.0;  // Value of diagonal values in matrix.
  const size_t n = rhs.size();
  double bet = kDiag;
  // Decomposition & forward substitution.
  out[0] = rhs[0] / bet;
  for (int j = 1; j < n; j++) {
    // Cache lo[j] so we can use lo as temporary work array.
    const double loj = lo[j];
    lo[j] = up[j - 1] / bet;
    bet = kDiag - loj * lo[j];
    // This should never happen in this use-case (for strictly increasing knot
    // vector).
    CHECK(0.0 != bet);
    out[j] = (rhs[j] - loj * out[j - 1]) / bet;
  }

  // Backsubstitution.
  for (int j = n - 2; j >= 0; j--) {
    out[j] -= lo[j + 1] * out[j + 1];
  }
}
}  // namespace

bool CubicSpline::Init(size_t num_points) {
  if (num_points > kMaxNumPoints) {
    LOG(ERROR) << absl::Substitute("Number of points must be < $0, got $1",
                                   kMaxNumPoints, num_points);
    return false;
  }
  if (num_points < kMinNumPoints) {
    LOG(ERROR) << absl::Substitute("Number of points must be > $0, got $1",
                                   kMinNumPoints, num_points);
    return false;
  }

  num_points_ = num_points;

  u_.resize(num_points);
  u_ = 0.0;

  a_.resize(num_points - 1);
  b_.resize(num_points);
  c_.resize(num_points - 1);
  p_.resize(num_points);

  a_ = 0.0;
  b_ = 0.0;
  c_ = 0.0;
  p_ = 0.0;

  du_.resize(num_points);
  dp_.resize(num_points);
  db_.resize(num_points);
  lse_lo_.resize(num_points);
  lse_up_.resize(num_points);
  lse_rhs_.resize(num_points);

  db_ = 0.0;
  du_ = 0.0;
  dp_ = 0.0;
  lse_lo_ = 0.0;
  lse_up_ = 0.0;
  lse_rhs_ = 0.0;
  initialized_ = true;
  return true;
}

void CubicSpline::SetBoundaryConditions(const BoundaryCond& start,
                                        const BoundaryCond& end) {
  bound_start_ = start;
  bound_end_ = end;
}

size_t CubicSpline::Index(const double u) const {
  if (u >= u_[num_points_ - 1]) {
    return num_points_ - 2;
  }
  if (u < u_[0]) {
    return 0;
  }
  // binary search for interval
  size_t low = 0;
  size_t high = num_points_ - 1;
  size_t mid = (low + high) / 2;
  while (u < u_[mid] || u >= u_[mid + 1]) {
    if (u < u_[mid]) {
      high = mid;
    } else {
      low = mid;
    }
    mid = (low + high) / 2;
  }
  return mid;
}

bool CubicSpline::CalculateParameters() {
  if (!initialized_) {
    LOG(ERROR) << "Call init() first.";
    return false;
  }

  std::adjacent_difference(std::begin(u_), std::end(u_), std::begin(du_));
  std::adjacent_difference(std::begin(p_), std::end(p_), std::begin(dp_));

  for (size_t idx = 0; idx < num_points_ - 2; idx++) {
    // Strict ordering on u_ assured, so devision is safe.
    const double inv_denom = 0.5 / (du_[idx + 1] + du_[idx + 2]);
    lse_up_[idx + 1] = du_[idx + 2] * inv_denom;
    lse_lo_[idx + 1] = du_[idx + 1] * inv_denom;
    lse_rhs_[idx + 1] =
        3.0 * inv_denom *
        (dp_[idx + 2] / du_[idx + 2] - dp_[idx + 1] / du_[idx + 1]);
  }

  // set boundary conditions
  switch (bound_start_.type) {
    case BoundaryCond::kAcceleration:
      lse_up_[0] = 0.0;
      // initialized to zero: lse_lo_[0] = 0.0;
      lse_rhs_[0] = bound_start_.value * 0.5;
      break;
    case BoundaryCond::kVelocity:
      lse_up_[0] = 0.5;
      // initialized to zero: lse_lo_[0] = 0.0;
      lse_rhs_[0] = 1.5 / du_[1] * (dp_[1] / du_[1] - bound_start_.value);
      break;
    default:
      CHECK(false) << "Invalid boundary type " << bound_start_.type;
  }
  switch (bound_end_.type) {
    case BoundaryCond::kAcceleration:
      // initialized to zero: lse_up_[num_points_ - 1] = 0.0;
      lse_lo_[num_points_ - 1] = 0.0;
      lse_rhs_[num_points_ - 1] = bound_end_.value / 2.0;
      break;
    case BoundaryCond::kVelocity:
      // initialized to zero: lse_up_[num_points_-1] = 0.0;
      lse_lo_[num_points_ - 1] = 0.5;
      lse_rhs_[num_points_ - 1] =
          (bound_end_.value - dp_[num_points_ - 1] / du_[num_points_ - 1]) *
          1.5 / du_[num_points_ - 1];
      break;
    default:
      CHECK(false) << "Invalid boundary type " << bound_start_.type;
  }

  // Note: The one missing common spline variant is the "not-a-knot spline"
  // (continuity of third derivative at first and last internal knot). This is
  // what many commercial tools do as a default (e.g., Mathematica and Matlab).
  // Add this here if it is needed.

  // Solve for b-parameters (corresponds to accelerations at knots).
  SolveThomas(
      {.lower = &lse_lo_, .upper = lse_up_, .rhs = lse_rhs_, .output = &b_});

  // compute remaining coefficients
  std::adjacent_difference(std::begin(b_), std::end(b_), std::begin(db_));
  for (size_t idx = 0; idx < num_points_ - 1; idx++) {
    a_[idx] = db_[idx + 1] / (3.0 * du_[idx + 1]);
    c_[idx] = dp_[idx + 1] / du_[idx + 1] -
              du_[idx + 1] * (b_[idx + 1] + 2.0 * b_[idx]) / 3.0;
  }

  calculated_ = true;
  return true;
}

bool CubicSpline::EvalCurve(double u, double* value) const {
  if (!initialized_ || !calculated_) {
    LOG(ERROR) << "Call init() and calculateParameters() first.";
    return false;
  }

  if (u < umin_) {
    switch (out_of_bound_policy_) {
      case OutOfBoundPolicy::kUseBound:
        *value = p_[0];
        return true;
      case OutOfBoundPolicy::kExtrapolate:
        break;
      case OutOfBoundPolicy::kError:
      default:
        LOG(ERROR) << absl::Substitute(
            "Parameter out of range: u= $0, should be in [$1, $2].", u, umin_,
            umax_);
        return false;
    }
  }

  if (u > umax_) {
    switch (out_of_bound_policy_) {
      case OutOfBoundPolicy::kUseBound:
        *value = p_[p_.size() - 1];
        return true;
      case OutOfBoundPolicy::kExtrapolate:
        break;
      case OutOfBoundPolicy::kError:
      default:
        LOG(ERROR) << absl::Substitute(
            "Parameter out of range: u= $0, should be in [$1, $2]", u, umin_,
            umax_);
        return false;
    }
  }

  int idx = Index(u);
  const double du = u - u_[idx];

  *value = a_[idx];
  *value = *value * du + b_[idx];
  *value = *value * du + c_[idx];
  *value = *value * du + p_[idx];

  return true;
}

bool CubicSpline::EvalCurveAndDerivatives(double u, double* val, double* dval,
                                          double* ddval) const {
  if (!initialized_ || !calculated_) {
    LOG(ERROR) << "Call init() and calculateParameters() first.";
    return false;
  }

  if (u < umin_) {
    switch (out_of_bound_policy_) {
      case OutOfBoundPolicy::kUseBound:
        *val = p_[0];
        *dval = 3.0 * a_[0];
        *ddval = 2.0 * b_[0];
        return true;
      case OutOfBoundPolicy::kExtrapolate:
        break;
      case OutOfBoundPolicy::kError:
      default:
        LOG(ERROR) << absl::Substitute(
            "Parameter out of range: u= $0, should be in [$1, $2].", u, umin_,
            umax_);
        return false;
    }
  }

  if (u > umax_) {
    switch (out_of_bound_policy_) {
      case OutOfBoundPolicy::kUseBound:
        *val = p_[p_.size() - 1];
        *dval = 3.0 * a_[num_points_ - 2];
        *dval = *dval * du_[num_points_ - 1] + 2.0 * b_[num_points_ - 2];
        *dval = *dval * du_[num_points_ - 1] + c_[num_points_ - 2];
        *ddval = 6.0 * a_[num_points_ - 2] * du_[num_points_ - 1] +
                 2.0 * b_[num_points_ - 2];
        return true;
      case OutOfBoundPolicy::kExtrapolate:
        break;
      case OutOfBoundPolicy::kError:
      default:
        LOG(ERROR) << absl::Substitute(
            "Parameter out of range: u= $0, should be in [$1, $2].", u, umin_,
            umax_);
        return false;
    }
  }

  int idx = Index(u);

  const double du = u - u_[idx];

  *val = a_[idx];
  *val = *val * du + b_[idx];
  *val = *val * du + c_[idx];
  *val = *val * du + p_[idx];

  *dval = 3.0 * a_[idx];
  *dval = *dval * du + 2.0 * b_[idx];
  *dval = *dval * du + c_[idx];

  *ddval = 6.0 * a_[idx] * du + 2.0 * b_[idx];

  return true;
}

bool CubicSpline::SetKnotVector(const double* knots, const size_t num_knots,
                                const double umin, const double umax) {
  if (!initialized_) {
    LOG(ERROR) << "Call init() first.";
    return false;
  }
  if (num_knots != num_points_) {
    LOG(ERROR) << absl::Substitute(
        "Wrong number of knots: got $0, should be $1.", num_knots, num_points_);
    return false;
  }
  // verify ordering
  for (size_t idx = 0; idx < num_knots - 1; idx++) {
    if (knots[idx + 1] <= knots[idx]) {
      LOG(ERROR) << absl::Substitute(
          "Knot must be strictly increasing, but knots[$0]=$1 and "
          "knots[$2]=$3.",
          idx + 1, knots[idx + 1], idx, knots[idx]);
      return false;
    }
  }

  std::copy(knots, knots + num_points_, std::begin(u_));
  if (!std::isfinite(umin)) {
    umin_ = knots[0];
  } else {
    if (umin >= knots[0]) {
      umin_ = umin;
    } else {
      LOG(WARNING) << "Got umin<knots[0], setting to knots[0].";
      umin_ = knots[0];
    }
  }
  if (!std::isfinite(umax)) {
    umax_ = knots[num_knots - 1];
  } else {
    if (umax <= knots[num_knots - 1]) {
      umax_ = umax;
    } else {
      umax_ = knots[num_knots - 1];
      LOG(WARNING) << "Got umax<knots[0], setting to knots[0].";
    }
  }

  return true;
}

bool CubicSpline::SetControlPoints(const double* points,
                                   const size_t num_points) {
  if (!initialized_) {
    LOG(ERROR) << absl::Substitute("Call init() first.");
    return false;
  }
  if (num_points != num_points_) {
    LOG(ERROR) << absl::Substitute(
        "Wrong number of points: got $0, should be $1.", num_points,
        num_points_);
    return false;
  }
  std::copy(points, points + num_points_, std::begin(p_));
  return true;
}
}  // namespace trajectory_planning

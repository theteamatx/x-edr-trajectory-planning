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

#include "trajectory_planning/splines/bspline_base.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"

namespace trajectory_planning {
constexpr size_t kMaxDegree = 128;
constexpr size_t kMinNumKnots = 3;

absl::Status BSplineBase::Init(size_t degree, size_t max_num_knots) {
  if (max_num_knots < kMinNumKnots) {
    return absl::OutOfRangeError(absl::Substitute(
        "max_num_knots must be >= 3, but got $0", max_num_knots));
  }

  if (degree > kMaxDegree) {
    return absl::OutOfRangeError(absl::Substitute(
        "degree too large, increase kMaxDegree (got $0)", degree));
  }

  // p = degree
  // m+1: number of knots (0,.., m) = num_knots
  // n+1: number of basis functions/points,
  //     n+1 = m-p-1, n=m-p-2
  //     ==> m_num_points = (num_knots-1)-degree-1 num_knots-degree-2
  degree_ = degree;

  num_knots_ = 0;
  num_points_ = 0;
  if (NumPoints(max_num_knots, degree) < 1) {
    return absl::InvalidArgumentError(
        absl::Substitute("Degree ($0) and knot capacity $1 inconsistent",
                         degree, max_num_knots));
  }

  knots_.resize(max_num_knots);
  knots_ = 0.0;

  basis_.resize(degree + 1);
  basis_ = 0.0;

  left_.resize(degree + 1);
  left_ = 0.0;
  right_.resize(degree + 1);
  right_ = 0.0;

  basis_ders_.resize(degree + 1, degree + 1);
  basis_du_.resize(degree + 1, degree + 1);
  a_.resize(2, (degree + 1));
  a_.setZero();

  tmp_.resize(degree + 1);
  tmp_ = 0.0;

  return absl::OkStatus();
}

absl::Status BSplineBase::SetUniformKnotVector(size_t num_knots) {
  umin_ = 0;
  umax_ = 1.0;

  if (num_knots > knots_.size()) {
    return absl::OutOfRangeError(absl::Substitute(
        "num_knots > max_num_knots ($0 > $1)", num_knots, knots_.size()));
  }
  const size_t min_num_knots = (degree_ + 1) * 2;
  if (num_knots < min_num_knots) {
    return absl::OutOfRangeError(absl::Substitute(
        "Knot vector too short ($0 < $1)", num_knots, min_num_knots));
  }

  knots_.segment(0, degree_ + 1).setZero();
  knots_.segment(num_knots - 1 - degree_, degree_ + 1).setConstant(1);
  double spacing = 1.0 / (num_knots - 2 * (degree_ + 1) + 1);
  for (int idx = degree_ + 1; idx < num_knots - degree_ - 1; idx++) {
    knots_[idx] = knots_[idx - 1] + spacing;
  }
  num_knots_ = num_knots;

  return absl::OkStatus();
}

absl::Status BSplineBase::SetKnotVector(absl::Span<const double> knots,
                                        const double umin, const double umax) {
  if (knots.size() > knots_.size()) {
    return absl::OutOfRangeError(absl::Substitute(
        "knots.size() > max_num_knots ($0 > $1)", knots.size(), knots_.size()));
  }
  const size_t min_num_knots = (degree_ + 1) * 2;
  if (knots.size() < min_num_knots) {
    return absl::OutOfRangeError(absl::Substitute(
        "Knot vector too short ($0 < $1)", knots.size(), min_num_knots));
  }

  // Verify non-decreasing knots.
  for (size_t i = 1; i < knots.size(); i++) {
    if (knots[i - 1] > knots[i]) {
      return absl::InvalidArgumentError("knot vector not increasing");
    }
  }
  // Verify knot multiplicity at beginning and end.
  for (int i = 1; i < degree_; ++i) {
    if (knots[i] != knots.front()) {
      return absl::InvalidArgumentError("First degree_+1 knots must be equal.");
    }
    if (knots[knots.size() - degree_ - 1 + i] != knots.back()) {
      return absl::InvalidArgumentError("Last degree_+1 knots must be equal.");
    }
  }

  num_knots_ = knots.size();
  absl::c_copy(knots, std::begin(knots_));
  num_points_ = NumPoints(num_knots_, degree_);

  if (!std::isfinite(umin)) {
    umin_ = knots[0];
  } else {
    if (umin >= knots[0]) {
      umin_ = umin;
    } else {
      LOG(WARNING) << "Got umin<knots[0], setting to knots[0]";
      umin_ = knots[0];
    }
  }
  if (!std::isfinite(umax)) {
    umax_ = knots.back();
  } else {
    if (umax <= knots.back()) {
      umax_ = umax;
    } else {
      umax_ = knots.back();
      LOG(WARNING) << "Got umin<knots[0], setting to knots[0]";
    }
  }

  return absl::OkStatus();
}

absl::Span<const double> BSplineBase::GetKnotVector() const {
  return absl::MakeConstSpan(knots_.data(), num_knots_);
}

absl::Status BSplineBase::CanInsertKnot(double knot, int multiplicity) {
  // Note: This doesn't check if the multiplicity of the knot is > degree_+1
  // after insertion because knot coincides with an existing internal knot
  // multiplicity. This should not break anything, but it might be desirable to
  // avoid this if we start using curves with internal knot multiplicity.
  if (multiplicity > degree_ + 1) {
    // There is no point in supporting this, so return an error.
    return absl::InvalidArgumentError(
        "Knot multiplicity > degree + 1 not supported.");
  }
  // Check capacity. If knot capacity is sufficient, point capacity should be
  // sufficient as well.
  const int required_knot_capacity = num_knots_ + multiplicity;
  if (required_knot_capacity > knots_.size()) {
    return absl::FailedPreconditionError(
        absl::Substitute("Knot capacity too small, required: $0, actual: $1",
                         required_knot_capacity, knots_.size()));
  }
  if (num_knots_ < 2) {
    return absl::FailedPreconditionError("Set initial knot vector first.");
  }
  // Check knot value.
  if (knot <= knots_[0] || knot >= knots_[num_knots_ - 1]) {
    return absl::InvalidArgumentError(absl::Substitute(
        "knot not in range of current knots, knot=$0, range=[$1,$2]", knot,
        knots_[0], knots_[num_knots_ - 1]));
  }

  return absl::OkStatus();
}

void BSplineBase::InsertKnotIntoKnotVector(double knot, int multiplicity,
                                           int knot_span) {
  CHECK_LE(num_knots_ + multiplicity, knots_.size());
  CHECK_LE(knot_span, num_points_ - 1);
  CHECK_EQ(num_knots_, NumKnots(num_points_, degree_));

  // 0,.., knot_span is unchanged.
  //  Shift tail of knot vector by 'multiplicity'.
  // Avoid temporaries and potential aliasing in the Eigen assignment operator.
  std::copy_backward(knots_.data() + knot_span, knots_.data() + num_knots_,
                     knots_.data() + num_knots_ + multiplicity);

  // knot_span+1 .. knot_span+multiplicity = the_knot
  knots_.segment(knot_span + 1, multiplicity).array() = knot;

  num_knots_ += multiplicity;
}

// This is alg. 2.1 from the NURBS book.
// Implements a binary search for the knot span index i in which u lies:
// u \in [knots_[i], knots_[i+1])
size_t BSplineBase::KnotSpan(double u) const {
  // spline not initialized ==> return 0;
  if (num_knots_ == 0) {
    return 0;
  }
  // Special case (upper bound): return last span to avoid special case in other
  // functions.
  // Return largest possible knot span index.
  // The spline values are computed by
  // \sum{basis[i] * points_[span - degree + i], i={0 ... degree}},
  // so the largest value is num_points_ - 1.
  if (u == knots_[num_knots_ - 1]) {
    return num_points_ - 1;
  }

  // Assert that u is in the knot range.
  CHECK(u >= knots_[0]) << absl::Substitute("u: $0; knots_[0]= $1", u,
                                            knots_[0]);
  CHECK(u <= knots_[num_knots_ - 1]) << absl::Substitute(
      "u: $0; knots_[$1]= $2", u, num_knots_ - 1, knots_[num_knots_ - 1]);

  // Find the (inclusive) lower bound.
  const size_t one_past_lower_bound =
      std::lower_bound(knots_.begin() + degree_,
                       knots_.begin() + num_knots_ - degree_, u,
                       [](const auto a, const auto b) { return a <= b; }) -
      knots_.begin();
  return one_past_lower_bound - 1;
}

// This is alg. 2.2 from the NURBS book.
const Eigen::ArrayXd& BSplineBase::UpdateBasis(size_t knot_span_idx, size_t p,
                                               double u) {
  basis_[0] = 1.0;
  basis_.segment(p + 1, basis_.size() - p - 1).setZero();
  left_.segment(1, p) = u - knots_(Eigen::seqN(knot_span_idx, p, -1));
  right_.segment(1, p) = knots_.segment(knot_span_idx + 1, p) - u;
  for (size_t j = 1; j <= p; j++) {
    double saved = 0.0;
    for (size_t r = 0; r < j; r++) {
      double tmp = basis_[r] / (right_[r + 1] + left_[j - r]);
      basis_[r] = saved + right_[r + 1] * tmp;
      saved = left_[j - r] * tmp;
    }
    basis_[j] = saved;
  }
  return basis_;
}

// Corresponds to algorithm 2.3 from the NURBS book.
const Eigen::MatrixXd& BSplineBase::UpdateBasisAndDerivatives(
    size_t knot_span_idx, size_t p, size_t der, double u) {
  CHECK(der <= p);
  CHECK(p <= degree_);

  // Note: replacing this loop with eigen templates using slices does not
  // speed the code up, but arguably makes it much harder to read.
  // In addition, some more more complex expressions apparently only compile
  // with '-c opt', but fail with '-c dbg'.
  basis_du_(0, 0) = 1.0;

  for (size_t j = 1; j <= p; j++) {
    left_[j] = u - knots_[knot_span_idx + 1 - j];
    right_[j] = knots_[knot_span_idx + j] - u;
    double saved = 0.0;
    for (size_t r = 0; r < j; r++) {
      // cache knot differences in lower triangle of basis_du_
      basis_du_(j, r) = right_[r + 1] + left_[j - r];
      const double tmp = basis_du_(r, j - 1) / basis_du_(j, r);
      // cache basis function in upper triangle of basis_du_
      basis_du_(r, j) = saved + right_[r + 1] * tmp;
      saved = left_[j - r] * tmp;
    }
    basis_du_(j, j) = saved;
  }
  // Copy basis functions to first row of basis_ders_.
  basis_ders_.row(0) = basis_du_.col(p);

  // Compute derivatives.
  int j1, j2;
  for (int r = 0; r <= p; r++) {
    int s1 = 0, s2 = 1;
    a_(0, 0) = 1.0;
    // kth derivative.
    for (int k = 1; k <= der; k++) {
      double d = 0.0;
      int rk = r - k;
      int pk = p - k;
      if (r >= k) {
        a_(s2, 0) = a_(s1, 0) / basis_du_(pk + 1, rk);
        d = a_(s2, 0) * basis_du_(rk, pk);
      }
      if (rk >= -1) {
        j1 = 1;
      } else {
        j1 = -rk;
      }
      if (r - 1 <= pk) {
        j2 = k - 1;
      } else {
        j2 = p - r;
      }
      for (size_t j = j1; j <= j2; j++) {
        a_(s2, j) = (a_(s1, j) - a_(s1, j - 1)) / basis_du_(pk + 1, rk + j);
        d += a_(s2, j) * basis_du_(rk + j, pk);
      }
      if (r <= pk) {
        a_(s2, k) = -a_(s1, k - 1) / basis_du_((pk + 1), r);
        d += a_(s2, k) * basis_du_(r, pk);
#ifdef DEBUG
        CHECK(std::isfinite(d));
#endif  // DEBUG
      }
      basis_ders_(k, r) = d;
#ifdef DEBUG
      CHECK(std::isfinite(d));
#endif  // DEBUG
      // Swap rows.
      std::swap(s1, s2);
    }
  }

  // Multiply by factors.
  tmp_ = p;
  for (size_t k = 1; k <= der; k++) {
    basis_ders_.row(k).array() *= tmp_;
    tmp_ *= (p - k);
  }

  return basis_ders_;
}

double UniformKnotSpacing(int num_knots, size_t degree) {
  const double denom = num_knots - 2.0 * (degree + 1.0) + 1.0;
  CHECK(denom > 0.0);
  return 1.0 / denom;
}

absl::Status BSplineBase::MakeUniformKnotVector(int num_points, size_t degree,
                                                double low_knot_value,
                                                double high_knot_value,
                                                absl::Span<double> knots) {
  if (high_knot_value <= low_knot_value) {
    return absl::InvalidArgumentError(
        "high_knot_value must be > low_knot_value");
  }
  if (num_points < MinNumPoints(degree)) {
    return absl::OutOfRangeError(absl::Substitute(
        "Too few points ($0 < $1)", num_points, MinNumPoints(degree)));
  }
  const auto nknots = NumKnots(num_points, degree);
  if (nknots != knots.size()) {
    return absl::OutOfRangeError(absl::Substitute(
        "knots has wrong size, should be $0, but is $1", nknots, knots.size()));
  }
  std::fill(knots.begin(), knots.begin() + degree + 1, low_knot_value);
  const double spacing =
      UniformKnotSpacing(nknots, degree) * (high_knot_value - low_knot_value);
  for (size_t idx = degree + 1; idx < nknots - degree - 1; idx++) {
    knots[idx] = knots[idx - 1] + spacing;
  }
  std::fill(knots.end() - degree - 1, knots.end(), high_knot_value);
  return absl::OkStatus();
}

absl::Status BSplineBase::MakeUniformKnotVector(int num_points,
                                                std::vector<double>* knots,
                                                size_t degree) {
  if (num_points < MinNumPoints(degree)) {
    return absl::OutOfRangeError(absl::Substitute(
        "Too few points ($0 < $1)", num_points, MinNumPoints(degree)));
  }
  const auto nknots = NumKnots(num_points, degree);
  if (nknots > knots->capacity()) {
    return absl::OutOfRangeError(absl::Substitute(
        "Too many knots ($0 < $1)", num_points, knots->capacity()));
  }
  knots->resize(nknots);

  return MakeUniformKnotVector(num_points, degree, 0.0, 1.0,
                               absl::MakeSpan(*knots));
}
}  // namespace trajectory_planning

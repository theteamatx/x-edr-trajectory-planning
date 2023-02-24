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

#ifndef TRAJECTORY_PLANNING_SPLINES_BSPLINE_BASE_H_
#define TRAJECTORY_PLANNING_SPLINES_BSPLINE_BASE_H_

#include <limits>
#include <vector>

#include "eigenmath/types.h"
#include "absl/status/status.h"
#include "absl/log/check.h"

namespace trajectory_planning {
// Base class for b-spline implementations.
// This class contains methods for calculating basis functions and derivatives,
// that a derived class can use to implement a curve in a desired space.
// For details on B-Splines see "The Nurbs Book" by Les Piegl, Wayne Tiller,
// which this implementation closely follows.
class BSplineBase {
 public:
  BSplineBase() = default;
  virtual ~BSplineBase() = default;

  // Initialize the spline. Not real-time safe, will allocate memory.
  // degree: polynomial degree
  // max_num_knots: maximum length of the knot vector
  absl::Status Init(size_t degree, size_t max_num_knots);

  // Returns the degree of the spline.LOG@
  double GetDegree() const { return degree_; }

  // Get number of points for a curve using this knot vector & degree
  // returns number of points
  int NumPoints() const { return num_points_; }
  // Get number maximum number of points in control polygon
  int MaxNumPoints() const { return NumPoints(knots_.size(), degree_); }
  // Set knot vector. Must be a non-decreasing sequence.
  // (multiple knots decrease continuity of the curve).
  // umin: minimum curve parameter, defaults to smallest in knot vector
  // umax: maximum curve parameter, defaults to largest in knot vector
  // return: true on success, false on error
  absl::Status SetKnotVector(
      absl::Span<const double> knots,
      const double umin = std::numeric_limits<double>::quiet_NaN(),
      const double umax = std::numeric_limits<double>::quiet_NaN());
  // Untility function that sets a uniform knot vector for curve parameter in
  // [0,1]
  // num_knots: size of the knot vector
  // return: true on success, false on error
  absl::Status SetUniformKnotVector(size_t num_knots);
  // Returns the knot vector.
  absl::Span<const double> GetKnotVector() const;
  // return number of control points for given knot length and degree
  static constexpr ssize_t NumPoints(size_t knots, size_t degree) {
    return knots - degree - 1;
  }
  // return number of knots for given control polygon and degree
  static constexpr size_t NumKnots(size_t points, size_t degree) {
    return points + degree + 1;
  }
  // return the minimum number of points for a given degree
  static constexpr size_t MinNumPoints(size_t degree) { return degree + 1; }
  // return the minimum number of knots for a given degree
  static constexpr size_t MinNumKnots(size_t degree) { return 2 * degree + 2; }
  // Fills `knots` with uniformly spaced knots, resizing the vector if
  // necessary.
  static absl::Status MakeUniformKnotVector(int num_points,
                                      std::vector<double>* knots,
                                      size_t degree);
  // Same as above, but requires knots to be appropriately sized and uses
  // the given low and high values for the knot vector.
  static absl::Status MakeUniformKnotVector(int num_points, size_t degree,
                                      double low_knot_value,
                                      double high_knot_value,
                                      absl::Span<double> knots);

  // Find and return knot span index with non-vanishing basis functions.
  // Asserts if u is not inside min/max knot values.
  // u: function/curve parameter
  size_t KnotSpan(double u) const;

  // Compute and return non-vanishing basis functions \f$N_{i-p,p}(u), \dots,
  // N_{i,p}(u)\f$.
  // This sets basis_[j]=N_{i-j,p}(u)
  // i: knot span index
  // p: degree of basis functions
  // u: curve/function parameter
  const Eigen::ArrayXd& UpdateBasis(size_t knot_span_idx, size_t p, double u);

  // Compute and return non-vanishing basis functions and derivatives
  // \f$\frac{partial^k N_{i-p,p}(u)}{partial u^k}, \dots \f$.
  // This sets basis_deriv_[j][k]=d^kN_{i-j,p}(u)/du^k
  // i: knot span index
  // p: degree of basis functions
  // u: curve/function parameter
  // der: highest derivative to calculate
  const eigenmath::MatrixXd& UpdateBasisAndDerivatives(size_t knot_span_idx,
                                                       size_t p, size_t der,
                                                       double u);

 protected:
  // Inserts `knot` `multiplicity` times.
  // The `knot` must be in the range of [knot[0], knot[n]] and the spline's
  // capacity allocated in Init must be sufficient.
  // Note: The calling this without updating control points in a derived class
  // puts the spline into an invalid state!
  // This function should only be called if CanInsertKnot() == Okabsl::Status();
  void InsertKnotIntoKnotVector(double knot, int multiplicity, int knot_span);
  absl::Status CanInsertKnot(double knot, int multiplicity);

  bool init_ = false;
  size_t degree_ = 0;
  size_t num_knots_ = 0;
  size_t num_points_ = 0;
  Eigen::ArrayXd knots_;
  double umin_ = 0;
  double umax_ = 0;

 private:
  Eigen::ArrayXd basis_;
  eigenmath::MatrixXd basis_ders_;
  eigenmath::MatrixXd basis_du_;
  eigenmath::MatrixXd a_;
  Eigen::ArrayXd tmp_;
  Eigen::ArrayXd left_;
  Eigen::ArrayXd right_;
};

}  // namespace trajectory_planning
#endif  // TRAJECTORY_PLANNING_SPLINES_BSPLINE_BASE_H_

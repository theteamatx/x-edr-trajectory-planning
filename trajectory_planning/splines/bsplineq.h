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

#ifndef TRAJECTORY_PLANNING_SPLINES_BSPLINEQ_H_
#define TRAJECTORY_PLANNING_SPLINES_BSPLINEQ_H_

#include <limits>
#include <valarray>
#include <vector>

#include "trajectory_planning/splines/bspline_base.h"
#include "eigenmath/types.h"
#include "absl/status/status.h"

namespace trajectory_planning {

// Quaternion B-Spline curve implementation.
// This implementation uses cumulative B-Spline basis functions to
// define the quaternion B-Spline as described in
// 'A General Construction Scheme for Unit Quaternion Curves with Simple High
// Order Derivatives' by Kim, Kim & Shin.
// Limitations: derivative calculations are not yet implemented.
class BSplineQ : public BSplineBase {
 public:
  using Quaterniond = ::eigenmath::Quaterniond;

  // Initialize the spline. Not real-time safe, will allocate memory.
  // degree: polynomial degree
  // max_num_knots: maximum length of the knot vector
  // return false on error, true on success
  absl::Status Init(size_t degree, size_t max_num_knots);
  // Initialize the spline and set knots & control points.
  // Not real-time safe, will allocate memory.
  absl::Status Init(size_t degree, absl::Span<const Quaterniond> points,
              absl::Span<const double> knots);
  // Set control points. This function will check if the number of points is
  // consistent with the knot vector, so call BSplineBase::setKnotVector first.
  // points: quaternion control points
  // return true on success, false on error
  absl::Status SetControlPoints(absl::Span<const Quaterniond> points);
  // Returns the control points.
  absl::Span<const Quaterniond> GetControlPoints() const;
  // Inserts `knot` `multiplicity` times.
  // On success, updates the knot and control point vector and returns
  // OkStatus().
  // The `knot` must be in the range of (knot[0], knot[n]) and the spline's
  // capacity allocated in Init must be sufficient.
  // The value of `multiplicity` must be >= 0, where 0 is a noop.
  // This does not change the curve's shape.
  // This function allocates inside Eigen's linear solver, if degree_ > 2.
  // Limitations:
  //  - For degree_ == 1, the curve's shape is not changed at all.
  //  - For higher order curves, knot insertion creates successively larger
  //    changes of the curve relative to the original shape.
  absl::Status InsertKnotAndUpdateControlPoints(double knot, int multiplicity);
  // Modifies knots and control points such that the curve ends at `u_end` and
  // is otherwise unchanged.
  // If u_end >= max, the spline is not modified, while u_end <= u_min results
  // in an empty curve.
  absl::Status TruncateSplineAt(double u_end);
  // Extends the current spline using the given control points.
  // The existing spline is unchanged. One additional control point is generated
  // such that the resulting curve is C^1-smooth.
  // Only implemented for 2nd order splines.
  // Returns non-ok on error.
  absl::Status ExtendWithControlPoints(absl::Span<const Quaterniond> points);
  // Evaluate b-spline curve
  // u: curve parameter
  // quat: curve at u
  // return true on success, false on error
  absl::Status EvalCurve(double u, Quaterniond& quat);

  // Evaluate first curve derivative.
  absl::Status EvalCurveAndDerivative(double u, Quaterniond& quat,
                                Quaterniond& derivative);

  // Exposed for testing purposes only.
  void UpdateCumulativeBasisAndDerivative(double u);
  void UpdateCumulativeBasis(double u);

  Eigen::ArrayXd GetCumulativeBasis() const { return cumulative_basis_; }
  Eigen::ArrayXd GetCumulativeBasisDerivative() const {
    return cumulative_basis_derivative_;
  }

 private:
  // Compute all cumulative basis functions Nc of degree p at u.
  void UpdateCumulativeBasis(size_t span, double u);
  void UpdateCumulativeBasisAndDerivative(size_t span, double u);
  // Insert knot once (that is, multiplicity == 1).
  absl::Status InsertKnotAndUpdateControlPoints(double knot);

  std::vector<Quaterniond> points_;
  Eigen::ArrayXd cumulative_basis_;
  Eigen::ArrayXd cumulative_basis_derivative_;

  // Resized to points_dim_*degree_ for knot insertion in Init.
  eigenmath::VectorXd spline_reference_values_;
  eigenmath::VectorXd spline_values_;

  // Resized to degree_ for knot insertion in Init.
  std::vector<Quaterniond> points_scratch_;
  // Storage for products of quaternion power terms used in
  // EvalCurveAndDerivative.
  std::vector<Quaterniond> quat_powers_;
  // Resized to degree_ for knot insertion in Init.
  std::vector<double> knots_scratch_;
};

namespace bsplineq_details {
eigenmath::Quaterniond QuatLog(const eigenmath::Quaterniond& quat_in);
eigenmath::Quaterniond QuatExp(const eigenmath::Quaterniond& quat_in);

// Returns quat^power.
eigenmath::Quaterniond QuatPower(const eigenmath::Quaterniond& quat,
                                 const double power);
// Returns d(quat^power)/dpower. This is an implementation detail of BSplineQ,
// but exposed here for testing.
eigenmath::Quaterniond QuatPowerDerivative(const eigenmath::Quaterniond& quat,
                                           const double power);
}  // namespace bsplineq_details

}  // namespace trajectory_planning

#endif  // TRAJECTORY_PLANNING_SPLINES_BSPLINEQ_H_

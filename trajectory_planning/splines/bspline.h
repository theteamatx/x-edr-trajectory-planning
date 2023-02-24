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

#ifndef TRAJECTORY_PLANNING_SPLINES_BSPLINE_H_
#define TRAJECTORY_PLANNING_SPLINES_BSPLINE_H_

#include <algorithm>
#include <limits>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "eigenmath/scalar_utils.h"
#include "eigenmath/types.h"
#include "trajectory_planning/splines/bspline_base.h"

namespace trajectory_planning {

#define RETURN_IF_ERROR(status) \
  if (!status.ok()) {           \
    return status;              \
  }

// B-Spline curve implementation.
// The dimension of the space the curve is in can be configured using the
// Traits
// template argument.
// It is expected to define:
// - Point type supporting [], +, *, += operations
// - a IsSizeOk function to check point dimensions
// - a size() function to query point size
// - a SetPointZero(Point) function to initialize points
// - a Resize(size, Point&) function to initialize dynamically sized points.
// - a kPointDim element defining point dimensions (-1 for variable size types)
// The order of the spline is also configurable.
// BSplines have some desirable advantages, such as
// - Local support
// - C^k smoothness, where k= degree-1
// - Coordinate system invariant
// - The curve transforms with its control points
// Note, however, that control points are not interpolated.
// For details on B-Splines see "The Nurbs Book" by Les Piegl, Wayne Tiller,
// which this implementation closely follows.
template <typename Traits>
class BSplineT : public BSplineBase {
 public:
  using Point = typename Traits::Point;

  BSplineT() = default;
  ~BSplineT() override = default;

  // Initialize the spline. Not real-time safe, will allocate memory.
  // degree: polynomial degree
  // max_num_knots: maximum length of the knot vector
  // points_dim dimension: of the space this curve is in
  absl::Status Init(size_t degree, size_t max_num_knots,
                    size_t points_dim = Traits::kPointDim);
  // Initialize the spline and set knots & control points.
  // Not real-time safe, will allocate memory.
  absl::Status Init(size_t degree, absl::Span<const Point> points,
                    absl::Span<const double> knots);
  // Set control points.
  // This function will check if the number of points is consistent with the
  // knot vector,
  // so call BSplineBase::setKnotVector first.
  // points: control points
  // return: true on success, false on error
  absl::Status SetControlPoints(absl::Span<const Point> points);
  // Returns the vector of control points.
  absl::Span<const Point> GetControlPoints() const;
  // Inserts `knot` `multiplicity` times.
  // On success, updates the knot and control point vector and returns
  // absl::OkStatus().
  // The `knot` must be in the range of (knot[0], knot[n]) and the spline's
  // capacity allocated in Init must be sufficient.
  // This does not change the curve's shape.
  // This function allocates inside Eigen's linear solver.
  absl::Status InsertKnotAndUpdateControlPoints(double knot, int multiplicity);
  // Same as above, but exploiting the structure of the standard linear
  // b-spline.
  // This function does not allocate.
  absl::Status InsertKnotAndUpdateControlPointsRef(double knot,
                                                   int multiplicity);
  // Modifies knots and control points such that the curve ends at `u_end` and
  // is otherwise unchanged.
  // If u_end >= max, the spline is not modified, while u_end <= u_min results
  // in an empty curve.
  absl::Status TruncateSplineAt(double u_end);
  // Extends the current spline using the given control points.
  // The existing spline is unchanged. One additional control point is generated
  // such that the resulting curve is C^1-smooth.
  // Only implemented for 3rd order splines.
  // Returns non-ok on error.
  absl::Status ExtendWithControlPoints(absl::Span<const Point> points);
  // Evaluate b-spline curve.
  // u: curve parameter
  // value: curve at u
  // return: true on success, false on error
  absl::Status EvalCurve(double u, Point& value);
  // Evaluate b-spline curve and derivatives.
  // u: curve parameter
  // values: curve and derivative values.
  //        length of values determines number of derivaties
  //        to calculate
  // return: true on success, false on error
  absl::Status EvalCurveAndDerivatives(double u, absl::Span<Point> values);
  // Returns the dimension of the underlying point space.
  int GetPointsDim() const { return points_dim_; }

 private:
  // Insert knot once.
  absl::Status InsertKnotAndUpdateControlPoints(double knot);
  // Insert knot once using reference implementation.
  absl::Status InsertKnotAndUpdateControlPointsRef(double knot);

  int points_dim_;
  std::vector<Point> points_;
  // Resized to points_dim_*degree_ for knot insertion in Init.
  eigenmath::VectorXd spline_reference_values_;
  // Allocated to points_dim_*degree_ x points_dim_*degree_ values for knot
  // insertion in Init.
  eigenmath::MatrixXd spline_gradient_;
  // Resized to points_dim_*degree_ for knot insertion in Init.
  eigenmath::VectorXd new_control_points_;
  // Resized to degree_ for knot insertion in Init.
  std::vector<Point> points_scratch_;
  // Resized to degree_ for knot insertion in Init.
  std::vector<double> knots_scratch_;
};

template <typename Traits>
absl::Status BSplineT<Traits>::Init(size_t degree, size_t max_num_knots,
                                    size_t points_dim) {
  init_ = false;
  RETURN_IF_ERROR(BSplineBase::Init(degree, max_num_knots));
  if (!Traits::IsSizeOk(points_dim)) {
    return absl::InvalidArgumentError(
        absl::Substitute("Point size not valid ($0)", points_dim));
  }
  points_dim_ = points_dim;

  const int max_num_points = NumPoints(max_num_knots, degree);
  points_.resize(max_num_points);
  for (auto& p : points_) {
    Traits::Resize(points_dim, p);
    Traits::SetPointZero(p);
  }

  points_scratch_.resize(degree_);
  for (auto& p : points_scratch_) {
    Traits::Resize(points_dim, p);
    Traits::SetPointZero(p);
  }
  knots_scratch_.resize(degree_, 0.0);

  spline_reference_values_.resize(points_dim_ * degree_);
  new_control_points_.resize(points_dim_ * degree_);
  spline_gradient_.resize(spline_reference_values_.size(),
                          spline_reference_values_.size());

  init_ = true;
  return absl::OkStatus();
}

template <typename Traits>
absl::Status BSplineT<Traits>::Init(size_t degree,
                                    absl::Span<const Point> points,
                                    absl::Span<const double> knots) {
  if (points.empty()) {
    return absl::InvalidArgumentError("Control point vector empty.");
  }
  RETURN_IF_ERROR(Init(degree, knots.size(), points.front().size()));
  RETURN_IF_ERROR(SetKnotVector(knots));
  RETURN_IF_ERROR(SetControlPoints(points));
  return absl::OkStatus();
}

template <typename Traits>
absl::Status BSplineT<Traits>::SetControlPoints(
    absl::Span<const Point> points) {
  if (!init_) {
    return absl::FailedPreconditionError("Call Init first.");
  }
  if (points.size() > points_.capacity()) {
    return absl::InvalidArgumentError(
        absl::Substitute("Too many points: points.size: $0 capacity: $1",
                         points.size(), points_.capacity()));
  }
  if (points.size() != NumPoints(num_knots_, degree_)) {
    return absl::InvalidArgumentError(
        absl::Substitute("points.size() inconsistent, got: $0 expected: $1",
                         points.size(), NumPoints(num_knots_, degree_)));
  }
  num_points_ = points.size();
  for (size_t i = 0; i < num_points_; i++) {
    if (Traits::size(points[i]) != points_dim_) {
      return absl::InvalidArgumentError(absl::Substitute(
          "Invalid point size at index $0, got $1, expected $2", i,
          Traits::size(points[i]), points_dim_));
    }
  }
  absl::c_copy(points, points_.begin());
  return absl::OkStatus();
}

template <typename Traits>
absl::Span<const typename BSplineT<Traits>::Point>
BSplineT<Traits>::GetControlPoints() const {
  return absl::MakeConstSpan(points_.data(), num_points_);
}

template <typename Traits>
absl::Status BSplineT<Traits>::InsertKnotAndUpdateControlPoints(
    double knot, int multiplicity) {
  RETURN_IF_ERROR(CanInsertKnot(knot, multiplicity));
  for (int i = 0; i < multiplicity; ++i) {
    RETURN_IF_ERROR(InsertKnotAndUpdateControlPoints(knot));
  }
  return absl::OkStatus();
}

template <typename Traits>
absl::Status BSplineT<Traits>::InsertKnotAndUpdateControlPointsRef(
    double knot, int multiplicity) {
  RETURN_IF_ERROR(CanInsertKnot(knot, multiplicity));
  for (int i = 0; i < multiplicity; ++i) {
    RETURN_IF_ERROR(InsertKnotAndUpdateControlPointsRef(knot));
  }
  return absl::OkStatus();
}

template <typename Traits>
absl::Status BSplineT<Traits>::InsertKnotAndUpdateControlPointsRef(
    double knot) {
  // This function implements a stripped down version of Alg. 5.1 from the NURBS
  // book, with slight modifications to avoid some temporary arrays, and without
  // direct handling of knot multiplicity.

  // Double-check point capacity. This should never trigger, as point capacity
  // should be sufficient if knot capacity is, which is checked in
  // CanInsertKnot.
  CHECK(num_points_ + 1 <= points_.size()) << absl::Substitute(
      "Point capacity ($0) smaller than required capacity ($1), "
      "but knot capacity sufficient?",
      points_.size(), num_points_ + 1);

  const int knot_span = KnotSpan(knot);
  // Shift unchanged points after affected local spline region.
  std::copy_backward(points_.data() + knot_span, points_.data() + num_points_,
                     points_.data() + num_points_ + 1);
  // Compute new `degree` number of control points for new knot vector.
  for (int i = 0; i < degree_; ++i) {
    const int k = knot_span + i - degree_ + 1;
    const double alpha = (knot - knots_[k]) / (knots_[k + degree_] - knots_[k]);
    points_scratch_[i] = alpha * points_[k] + (1.0 - alpha) * points_[k - 1];
  }

  std::copy(points_scratch_.data(), points_scratch_.data() + degree_,
            points_.data() + knot_span - degree_ + 1);
  // Update knot vector.
  BSplineBase::InsertKnotIntoKnotVector(knot, 1, knot_span);

  num_points_ += 1;

  return absl::OkStatus();
}

template <typename Traits>
absl::Status BSplineT<Traits>::InsertKnotAndUpdateControlPoints(double knot) {
  // Allocates in colPivHouseholderQr, see comment there.

  // Implementation of knot insertion based on directly solving a system of
  // equations without exploiting linearity of the b-spline or orthogonality
  // of basis functions.
  // This is to make the basic approach generalizable to the non-linear
  // BSplineQ-case.

  // Double-check point capacity. This should never trigger, as point capacity
  // should be sufficient if knot capacity is, which is checked in
  // CanInsertKnot.
  CHECK(num_points_ + 1 <= points_.size()) << absl::Substitute(
      "Point capacity ($0) smaller than required capacity ($1), "
      "but knot capacity sufficient?",
      points_.size(), num_points_ + 1);

  if (degree_ < 1) {
    return absl::UnimplementedError("Not implemented for splines of degree 0.");
  }

  const int knot_span = KnotSpan(knot);

  // Control points that affect the section of the spline that is influenced by
  // the modified knot vector are recomputed by solving the system of equations:
  // spline_before_change(ui) = spline_after_change(ui).
  // Here the ui=`knots_scratch_` are uniformly spaced within the relevant
  // range of parameter values. The concrete choice influences the condition of
  // the system of equations and could likely be optimized.
  const double one_over_degree_plus_one = 1.0 / (degree_ + 1);
  for (int i = 0; i < degree_; ++i) {
    knots_scratch_[i] =
        knot + i * one_over_degree_plus_one * (knots_[knot_span + 1] - knot);
  }

  for (int ui = 0; ui < degree_; ++ui) {
    const double u = knots_scratch_[ui];
    RETURN_IF_ERROR(EvalCurve(u, points_scratch_[0]));
    spline_reference_values_.segment(points_dim_ * ui, points_dim_) =
        points_scratch_[0];
  }

  // Update knot vector.
  BSplineBase::InsertKnotIntoKnotVector(knot, 1, knot_span);
  num_points_ += 1;

  // Shift and recompute control points using new knot vector and reference
  // curve values.
  // Unchanged control points: 0 .. knot_span - degree_.
  // Modified control points:  knot_span - degree_ + 1 .. knot_span.
  // Shifted control points:   knot_span .. num_points_ - 2.
  // Shift unchanged points after affected local spline region.
  std::copy_backward(points_.data() + knot_span,
                     points_.data() + num_points_ - 1,
                     points_.data() + num_points_);

  // Recompute control points so the curve doesn't change:
  const int point_offset = knot_span - degree_ + 1;
  for (int i = 0; i < degree_; ++i) {
    const int k = point_offset + i;
    Traits::SetPointZero(points_[k]);
  }

  // The b-spline is linear in the control points (see EvalCurve), so can be
  // written as offset+gradient*points.
  // The equation to solve then is: spline_reference_values_ = offsets +
  // gradient*points. Below the offsets are directly subtracted from the
  // reference values stored in `spline_reference_values_`.
  for (int ui = 0; ui < degree_; ++ui) {
    const double u = knots_scratch_[ui];
    RETURN_IF_ERROR(EvalCurve(u, points_scratch_[0]));
    spline_reference_values_.segment(points_dim_ * ui, points_dim_) -=
        points_scratch_[0];
  }

  // Compute the gradient w.r.t. the unknown points. The finite differences with
  // unit steps used below are exact because of the b-spline's linearity in the
  // control points.
  struct PointAndDimension {
    int point;
    int dimension;
  };
  const auto get_indices =
      [&point_offset, this](const int variable_index) -> PointAndDimension {
    return {.point = variable_index / points_dim_ + point_offset,
            .dimension = variable_index % points_dim_};
  };
  for (int k = 0; k < degree_ * points_dim_; ++k) {
    const PointAndDimension indices = get_indices(k);
    points_[indices.point](indices.dimension) = 1.0;
    // Evaluate functions for all equations.
    for (int ui = 0; ui < degree_; ++ui) {
      const double u = knots_scratch_[ui];
      RETURN_IF_ERROR(EvalCurve(u, points_scratch_[0]));
      spline_gradient_.block(points_dim_ * ui, k, points_dim_, 1) =
          points_scratch_[0];
    }
    // Restore zero value.
    points_[indices.point](indices.dimension) = 0.0;
    // Evaluate functions for all equations.
    for (int ui = 0; ui < degree_; ++ui) {
      const double u = knots_scratch_[ui];
      RETURN_IF_ERROR(EvalCurve(u, points_scratch_[0]));
      spline_gradient_.block(points_dim_ * ui, k, points_dim_, 1) -=
          points_scratch_[0];
    }
  }

  // This is where allocation happens.
  // If we have a need to use this function without allocation, there's likely
  // a way to pre-allocate storage for the decomposition in Init.
  new_control_points_ =
      spline_gradient_.colPivHouseholderQr().solve(spline_reference_values_);

  for (int k = 0; k < degree_; ++k) {
    points_[point_offset + k] =
        new_control_points_.segment(points_dim_ * k, points_dim_);
  }

  return absl::OkStatus();
}

template <typename Traits>
absl::Status BSplineT<Traits>::TruncateSplineAt(double u_end) {
  // Nothing to do.
  if (u_end >= umax_) {
    return absl::OkStatus();
  }
  // Clear the curve.
  if (u_end <= umin_) {
    umin_ = std::numeric_limits<double>::infinity();
    umax_ = -std::numeric_limits<double>::infinity();
    num_knots_ = 0;
    num_points_ = 0;
    return absl::OkStatus();
  }

  // Insert degree+1 control points, which effectively decouples the curve at
  // u_end.
  RETURN_IF_ERROR(InsertKnotAndUpdateControlPoints(u_end, degree_ + 1));

  // Discard second curve section.
  const int span = KnotSpan(u_end);
  num_knots_ = span + 1;
  num_points_ = NumPoints(num_knots_, degree_);
  umax_ = u_end;
  return absl::OkStatus();
}

template <typename Traits>
absl::Status BSplineT<Traits>::ExtendWithControlPoints(
    absl::Span<const Point> points) {
  // This is a special-case implementation that only works for degree_ = 2.
  if (degree_ != 2) {
    return absl::UnimplementedError("Only implemented for 2nd order splines.");
  }
  const int new_num_points = num_points_ + points.size();
  const int added_knots = NumKnots(points.size() + 1, degree_) - 2 * degree_;
  const int new_num_knots = num_knots_ + added_knots;

  if (num_knots_ < MinNumKnots(degree_)) {
    return absl::FailedPreconditionError("Spline is empty or invalid.");
  }

  if (new_num_points > points_.size()) {
    return absl::FailedPreconditionError(
        "Point capacity too small to append all points.");
  }
  if (new_num_knots > knots_.size()) {
    return absl::FailedPreconditionError(
        "Knot capacity too small to append all points.");
  }
  if (points.size() < 2) {
    // With one control point, we can't ensure the existing curve isn't modified
    // by adding in internal duplicate knot.
    return absl::UnimplementedError("Only implemented for >= 2 points.");
  }

  // The knot value where the old and new curve segments are joined.
  const double u_join = knots_[num_knots_ - 1];

  // Generate a uniform knot distribution for new control points with a density
  // proportional to that of the existing knots.
  const double old_knot_range = knots_[num_knots_ - 1] - knots_[0];
  const int old_inner_knot_count = num_knots_ - 2 * degree_ - 1;
  const int new_inner_knot_count = new_num_knots - 2 * degree_ - 1;
  const double new_knot_range =
      (old_knot_range * new_inner_knot_count) / old_inner_knot_count;
  const int linspace_upper_bound = new_num_knots - degree_;
  const int linspace_start_index = num_knots_ - degree_ - 1;
  const int linspace_size = linspace_upper_bound - linspace_start_index;
  knots_.segment(linspace_start_index, linspace_size)
      .setLinSpaced(old_knot_range, new_knot_range);

  knots_.segment(new_num_knots - degree_ - 1, degree_ + 1)
      .setConstant(knots_[0] + new_knot_range);
  num_knots_ = new_num_knots;
  umax_ = knots_[new_num_knots - 1];

  // Knot span for the spline parameter `u_join` at which curves are joined.
  const int u_joint_span = KnotSpan(u_join);
  // The index of the point that is modified to assure smoothness.
  const int modified_point_index = num_points_ - 1;
  const auto& basis = UpdateBasis(u_joint_span, degree_, u_join);
  // Sanity check, the value should be 0.5 by construction.
  CHECK(basis[1] > 0) << absl::Substitute(
      "basis[1]= $0, but should be > 0 (0.5).", basis[1]);

  // If the old control points are not modified, the curve's value at `u_join`
  // will change due to the modified knot values.
  // Because of knot multiplicity, the old end point is
  // `points_[modified_point_index]`.
  // After extending the curve, the curve value is (after updating the basis
  // function values):
  //   basis[0] * points_[modified_point_index - 1] +
  //   basis[1] * points_[modified_point_index] +
  //   basis[2] * points_[modified_point_index + 1]
  //   == points_[modified_point_index].
  // The value of basis[2] == 0.0 at the knot, so the modified point is:
  points_[modified_point_index] =
      1.0 / (basis[1]) *
      (points_[modified_point_index] -
       basis[0] * points_[modified_point_index - 1]);
  for (int idx = 0; idx < points.size(); ++idx) {
    points_[idx + num_points_] = points[idx];
  }

  num_points_ = new_num_points;

  return absl::OkStatus();
}

// Corresponds to algorithm 3.1 in NURBS Book.
template <typename Traits>
absl::Status BSplineT<Traits>::EvalCurve(double u, Point& value) {
  if (!init_) {
    return absl::FailedPreconditionError("Call Init first.");
  }
  if (u < umin_ || u > umax_) {
    return absl::OutOfRangeError(absl::Substitute(
        "Spline parameter $0, valid range [$1, $2]", u, umin_, umax_));
  }

  size_t span = KnotSpan(u);
  const auto& basis = UpdateBasis(span, degree_, u);
  Traits::SetPointZero(value);
  for (int i = 0; i <= degree_; i++) {
#ifdef DEBUG
    CHECK(span - degree_ + i < points_.size()) << absl::Substitute(
        "span= $0, degree_= $1, i= $2, num_points_= $3; u= $4", span, degree_,
        i, num_points_, u);
#endif
    value += basis[i] * points_[span - degree_ + i];
  }
  return absl::OkStatus();
}

// Corresponds to algorithm 3.2 in NURBS Book.
template <typename Traits>
absl::Status BSplineT<Traits>::EvalCurveAndDerivatives(
    double u, absl::Span<Point> values) {
  if (values.empty()) {
    return absl::InvalidArgumentError("values must not be empty.");
  }

  if (!init_) {
    return absl::FailedPreconditionError("Call Init first.");
  }
  if (u < umin_ || u > umax_) {
    return absl::OutOfRangeError(absl::Substitute(
        "Spline parameter $0, valid range [$1, $2]", u, umin_, umax_));
  }

  const size_t der = values.size() - 1;
  for (size_t k = degree_ + 1; k < der; k++) {
    Traits::SetPointZero(values[k]);
  }
  size_t du = std::min(der, degree_);
  size_t span = KnotSpan(u);
  const auto& basis_ders = UpdateBasisAndDerivatives(span, degree_, der, u);
  for (int k = 0; k <= du; k++) {
    Traits::SetPointZero(values[k]);
    for (int j = 0; j <= degree_; j++) {
      values[k] += basis_ders(k, j) * points_[span - degree_ + j];
    }
  }
  return absl::OkStatus();
}

// Implementations for commonly used point types.

// Traits for scalar b-spline, using double "points"
struct SplineTraits1d {
  typedef double Point;
  static constexpr int kPointDim = 1;
  static constexpr bool IsSizeOk(const int sz) { return sz == 1; }
  static void Resize(const int sz, Point& point) {}
  static constexpr int size(const Point& point) { return 1; }
  static void SetPointZero(Point& point) { point = 0.0; }
};

// b-spline using Vector2d points.
typedef BSplineT<SplineTraits1d> BSpline1d;

// Traits for b-spline in R^2, using Vector2d points.
struct SplineTraits2d {
  typedef eigenmath::Vector2d Point;
  static constexpr int kPointDim = 2;
  static constexpr bool IsSizeOk(const int sz) { return sz == 2; }
  static void Resize(const int sz, Point& point) {}
  static constexpr int size(const Point& point) { return 2; }
  static void SetPointZero(Point& point) { point.setZero(); }
};

// b-spline using Vector2d points.
typedef BSplineT<SplineTraits2d> BSpline2d;

// Traits for b-spline in R^3, using eigenmath::Vector3d points.
struct SplineTraits3d {
  typedef eigenmath::Vector3d Point;
  static constexpr int kPointDim = 3;
  static constexpr bool IsSizeOk(const int sz) { return sz == 3; }
  static void Resize(const int sz, Point& point) {}
  static constexpr int size(const Point& point) { return 3; }
  static void SetPointZero(Point& point) { point.setZero(); }
};

// b-spline using eigenmath::Vector3d points.
typedef BSplineT<SplineTraits3d> BSpline3d;

// Traits for b-spline in R^n, using eigenmath::VectorNd points.
struct SplineTraitsNd {
  typedef eigenmath::VectorNd Point;
  static constexpr int kPointDim = -1;
  static constexpr bool IsSizeOk(const int sz) {
    return sz > 0 && sz < eigenmath::kMaxEigenVectorCapacity;
  }
  static void Resize(const int sz, Point& point) {}
  static int size(const Point& point) { return point.rows(); }
  static void SetPointZero(Point& point) { point.setZero(); }
};

// b-spline using eigenmath::VectorNd points.
typedef BSplineT<SplineTraitsNd> BSplineNd;

// A B-Spline on eigenmath::VectorXd.
struct SplineTraitsXd {
  typedef eigenmath::VectorXd Point;
  static constexpr int kPointDim = -1;
  static constexpr bool IsSizeOk(const int sz) { return sz > 0; }
  static void Resize(const int sz, Point& point) { point.resize(sz); }
  static int size(const Point& point) { return point.rows(); }
  static void SetPointZero(Point& point) { point.setZero(); }
};
typedef BSplineT<SplineTraitsXd> BSplineXd;

}  // namespace trajectory_planning
#endif  // TRAJECTORY_PLANNING_SPLINES_BSPLINE_H_

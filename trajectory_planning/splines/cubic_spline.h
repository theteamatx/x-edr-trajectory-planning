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

#ifndef TRAJECTORY_PLANNING_SPLINES_CUBIC_SPLINE_H_
#define TRAJECTORY_PLANNING_SPLINES_CUBIC_SPLINE_H_

#include <limits>
#include <valarray>

namespace trajectory_planning {
// Interpolating cubic spline, ie, an interpolating piecewise cubic
// polynomial with C^2 continuity.
// Exploits details of this type of spline to efficiently compute
// polynomial coefficients.
// Details see e.g. Numerical Mathematics, Quateroni et al.
class CubicSpline {
 public:
  // Structure for defining boundary conditions.
  struct BoundaryCond {
    // Type of boundary condition
    enum Type {
      // Set acceleration at the boundary.
      kAcceleration,
      // Set velocity at the boundary.
      kVelocity
    };
    BoundaryCond() {}
    BoundaryCond(const Type type_, const double value_)
        : type(type_), value(value_) {}
    // boundary condition type
    Type type = kAcceleration;
    // boundary condition value
    double value = 0.0;
  };
  // Policies for handling out-of-bounds parameters.
  enum class OutOfBoundPolicy {
    // Treat as an error.
    kError,
    // Use boundary values.
    kUseBound,
    // Extrapolate.
    kExtrapolate
  };

  CubicSpline() = default;
  ~CubicSpline() = default;

  // Initialize the spline. Not real-time safe, will allocate memory.
  // num_points: number of control points
  // returns: false on error, true on success
  bool Init(size_t num_points);
  // Set control points
  // VecType a vector type with size() function and element operator
  // points control points for the spline
  // return true on success, false on error
  template <typename VecType>
  bool SetControlPoints(const VecType& points);
  // Set boundary conditions for spline. Default is natural spline
  // (accelerations zero)
  // start: boundary contition at the start
  // end: boundary contition at the end
  void SetBoundaryConditions(const BoundaryCond& start,
                             const BoundaryCond& end);
  // Set policy for treating out of bound parameters.
  // Default is treating as an error
  void SetOutOfBoundPolicy(OutOfBoundPolicy policy) {
    out_of_bound_policy_ = policy;
  }
  // Set knot vector.
  // VecType: a vector type with size() function and element operator
  // knots: knots for the spline
  // umin: lower bound for function parameter (defaults to knots[0])
  // umax: upper bound for function parameter (defaults to
  // knots[knots.size()-1])
  // return: true on success, false on error
  template <typename VecType>
  bool SetKnotVector(
      const VecType& knots,
      const double umin = std::numeric_limits<double>::quiet_NaN(),
      const double umax = std::numeric_limits<double>::quiet_NaN());
  // Calculate parameters.
  // return false on error, true on success
  bool CalculateParameters();
  // Evaluate spline at u.
  // u: curve parameter
  // value: curve at u
  // return: true on success, false on error
  bool EvalCurve(double u, double* value) const;
  // Evaluate b-spline curve and derivatives.
  // u: curve parameter
  // val: function value at u
  // dval: function value derivative at u
  // ddval: second function value derivative at u
  // return: true on success, false on error
  bool EvalCurveAndDerivatives(double u, double* val, double* dval,
                               double* ddval) const;
  // Get number of control points.
  size_t NumPoints() const { return num_points_; }

 private:
  bool SetKnotVector(const double* knots, const size_t num_knots,
                     const double umin, const double umax);
  bool SetControlPoints(const double* points, const size_t num_points);
  size_t Index(const double t) const;
  bool initialized_ = false;
  bool calculated_ = false;
  size_t num_points_ = 0;
  double umin_, umax_;
  // coefficients: curve is a[i]*(u-u[i])^3+b[i]*(u-u[i])^2+c[i]*(u-u[i])+p[i],
  // for u in [u[i], u[i+1]]
  std::valarray<double> a_, b_, c_, p_;
  // temporaries for parameter calculation
  std::valarray<double> lse_lo_, lse_up_, lse_rhs_;
  std::valarray<double> du_, dp_;
  std::valarray<double> db_;

  // knot vector; curve values at t_[i] are d_[i]
  std::valarray<double> u_;

  BoundaryCond bound_start_;
  BoundaryCond bound_end_;
  OutOfBoundPolicy out_of_bound_policy_ = OutOfBoundPolicy::kError;
};

// ---- Implementation
template <typename VecType>
bool CubicSpline::SetKnotVector(const VecType& knots, const double umin,
                                const double umax) {
  return SetKnotVector(&knots[0], knots.size(), umin, umax);
}
template <typename VecType>
bool CubicSpline::SetControlPoints(const VecType& points) {
  return SetControlPoints(&points[0], points.size());
}
}  // namespace trajectory_planning
#endif  // TRAJECTORY_PLANNING_SPLINES_CUBIC_SPLINE_H_

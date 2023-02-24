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

#include "trajectory_planning/splines/bsplineq.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "absl/algorithm/container.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "eigenmath/manifolds.h"
#include "eigenmath/scalar_utils.h"
#include "eigenmath/so3.h"
#include "eigenmath/types.h"

namespace trajectory_planning {
namespace {
constexpr int kQuatDim = 4;
using ::eigenmath::Quaterniond;

#define RETURN_IF_ERROR(status) \
  if (!status.ok()) {           \
    return status;              \
  }

// Calculates the Jacobian of `func` using finite differences.
void ForwardFiniteDifferences(
    absl::FunctionRef<eigenmath::VectorXd(const eigenmath::VectorXd&)> func,
    const eigenmath::VectorXd& value, const double increment,
    eigenmath::MatrixXd& output) {
  eigenmath::VectorXd arg = value;
  output.colwise() = -func(arg);
  for (int i = 0; i < arg.size(); ++i) {
    arg[i] += increment;
    output.col(i) += func(arg);
    arg[i] = value[i];
  }
  output /= increment;
}

struct SolveOptions {
  double max_residual = std::numeric_limits<double>::epsilon() * 100;
  int max_iter = 10;
  double max_step = 1e-2;
  double regularization = 1e-3;
  double finite_difference = 1e-4;
};

// Finds the root of `func` with the Newton-Raphson method.
// Gradients are computed by finite differences according to `options`,
// which also control damping and maximum step size.
absl::Status Solve(
    const SolveOptions& options,
    absl::FunctionRef<eigenmath::VectorXd(const eigenmath::VectorXd&)> func,
    eigenmath::VectorXd& argument) {
  eigenmath::VectorXd residual;
  eigenmath::MatrixXd gradient;
  eigenmath::VectorXd argument_delta =
      eigenmath::VectorXd::Constant(argument.size(), 1.0);
  for (int i = 0; i < options.max_iter; ++i) {
    residual = func(argument);
    if (residual.lpNorm<Eigen::Infinity>() < options.max_residual ||
        argument_delta.lpNorm<Eigen::Infinity>() < options.max_residual) {
      return absl::OkStatus();
    }

    gradient.resize(residual.rows(), argument.rows());
    ForwardFiniteDifferences(func, argument, options.finite_difference,
                             gradient);
    gradient.diagonal().array() += options.regularization;

    argument_delta = gradient.colPivHouseholderQr().solve(-residual);

    const double delta_norm = argument_delta.lpNorm<Eigen::Infinity>();
    if (delta_norm > options.max_step) {
      argument_delta *= options.max_step / delta_norm;
    }
    argument += argument_delta;
  }

  return absl::AbortedError("Solver didn't converge.");
}

void NormalizeIfNecessaryAndEnsurePositiveReal(Quaterniond& quat) {
  if (quat.w() < 0) {
    quat.coeffs() *= -1.0;
  }

  constexpr double kEpsilon = Eigen::NumTraits<double>::dummy_precision();

  if (std::abs(quat.squaredNorm() - 1.0) > kEpsilon) {
    quat.normalize();
  }
}
}  // namespace

namespace bsplineq_details {
Quaterniond QuatLog(const Quaterniond& quat_in) {
  const double squared_norm_q = quat_in.squaredNorm();
  const double norm_v = quat_in.vec().stableNorm();
  Quaterniond quat_out;
  quat_out.w() = 0.5 * std::log(squared_norm_q);
  if (norm_v > Eigen::NumTraits<double>::dummy_precision()) {
    quat_out.vec() =
        quat_in.vec().stableNormalized() * std::atan2(norm_v, quat_in.w());
  } else {
    quat_out.vec() = quat_in.vec();
  }
  return quat_out;
}

Quaterniond QuatExp(const Quaterniond& quat_in) {
  const double norm_v = quat_in.vec().stableNorm();
  Quaterniond quat_out;
  quat_out.w() = std::cos(norm_v);
  quat_out.vec() = quat_in.vec().stableNormalized() * std::sin(norm_v);
  quat_out.coeffs() *= std::exp(quat_in.w());
  return quat_out;
}

Quaterniond QuatPower(const Quaterniond& quat, const double power) {
  Quaterniond quat_in(quat);
  NormalizeIfNecessaryAndEnsurePositiveReal(quat_in);

  Quaterniond quat_out = QuatLog(quat_in);
  // q^p = exp(p * log(q))
  quat_out.coeffs() *= power;
  quat_out = QuatExp(quat_out);

  return quat_out;
}

Quaterniond QuatPowerDerivative(const Quaterniond& quat, const double power) {
  Quaterniond quat_in(quat);
  NormalizeIfNecessaryAndEnsurePositiveReal(quat_in);

  // q^p = exp(p * log(q))
  // d(q^p)/dp= d(exp(p*log(q)))/dp = exp(p*log(q))*log(q)
  Quaterniond log_quat = QuatLog(quat_in);
  Quaterniond quat_out = log_quat;
  quat_out.coeffs() *= power;
  quat_out = QuatExp(quat_out) * log_quat;
  return quat_out;
}
}  // namespace bsplineq_details

absl::Status BSplineQ::Init(size_t degree, size_t max_num_knots) {
  init_ = false;
  if (auto init_status = BSplineBase::Init(degree, max_num_knots);
      !init_status.ok()) {
    return init_status;
  }

  num_points_ = 0;
  points_.resize(NumPoints(max_num_knots, degree));
  for (auto& p : points_) {
    p.setIdentity();
  }
  cumulative_basis_.resize(degree_);
  cumulative_basis_derivative_.resize(degree_);

  points_scratch_.resize(degree_);
  for (auto& p : points_scratch_) {
    p.setIdentity();
  }
  quat_powers_.resize(degree_ + 1);
  for (auto& p : quat_powers_) {
    p.setIdentity();
  }
  knots_scratch_.resize(degree_, 0.0);
  spline_reference_values_.resize(kQuatDim * degree_);
  spline_values_.resize(kQuatDim * degree_);

  init_ = true;
  return absl::OkStatus();
}

absl::Status BSplineQ::Init(size_t degree, absl::Span<const Quaterniond> points,
                            absl::Span<const double> knots) {
  RETURN_IF_ERROR(Init(degree, knots.size()));
  RETURN_IF_ERROR(SetKnotVector(knots));
  RETURN_IF_ERROR(SetControlPoints(points));
  return absl::OkStatus();
}

absl::Status BSplineQ::SetControlPoints(absl::Span<const Quaterniond> points) {
  if (!init_) {
    return absl::FailedPreconditionError("Call init first.");
  }
  if (points.size() > points_.capacity()) {
    return absl::OutOfRangeError(absl::Substitute(
        "Too many points ($0 > $1)", points.size(), points_.capacity()));
  }
  if (points.size() != NumPoints(num_knots_, degree_)) {
    return absl::InvalidArgumentError(
        absl::Substitute("Wrong number of control points ($0 != $1)",
                         points.size(), NumPoints(num_knots_, degree_)));
  }
  num_points_ = points.size();
  absl::c_copy(points, points_.begin());
  return absl::OkStatus();
}

absl::Span<const Quaterniond> BSplineQ::GetControlPoints() const {
  return absl::MakeConstSpan(points_.data(), num_points_);
}

absl::Status BSplineQ::EvalCurve(double u, Quaterniond& quat) {
  using bsplineq_details::QuatPower;

  if (!init_) {
    return absl::FailedPreconditionError("Call init first.");
  }
  if (u < umin_ || u > umax_) {
    return absl::OutOfRangeError(
        absl::Substitute("Parameter $0 not in [$1, $2]", u, umin_, umax_));
  }
  const int span_index = KnotSpan(u);
  UpdateCumulativeBasis(span_index, u);
  quat = points_[span_index - degree_];
  for (int i = 0; i < degree_; ++i) {
    quat *= QuatPower(points_[span_index - degree_ + i].inverse() *
                          points_[span_index - degree_ + i + 1],
                      cumulative_basis_[i]);
  }
  NormalizeIfNecessaryAndEnsurePositiveReal(quat);

  return absl::OkStatus();
}

absl::Status BSplineQ::EvalCurveAndDerivative(double u, Quaterniond& quat,
                                              Quaterniond& derivative) {
  using bsplineq_details::QuatPower;
  using bsplineq_details::QuatPowerDerivative;

  if (!init_) {
    return absl::FailedPreconditionError("Call init first.");
  }
  if (u < umin_ || u > umax_) {
    return absl::OutOfRangeError(
        absl::Substitute("Parameter $0 not in [$1, $2]", u, umin_, umax_));
  }

  const int span_index = KnotSpan(u);
  UpdateCumulativeBasisAndDerivative(span_index, u);

  // Compute products of relative control quaternion powers used to compute
  // curve and curve derivative values below.
  // quat_powers_[0] is set to identity during initialization.
  for (int k = 0; k < degree_; ++k) {
    quat_powers_[k + 1] =
        quat_powers_[k] *
        QuatPower(points_[span_index - degree_ + k].inverse() *
                      points_[span_index - degree_ + k + 1],
                  cumulative_basis_[k]);
  }

  // Compute the curve value.
  quat = points_[span_index - degree_] * quat_powers_.back();

  // Sum over the contributions of the derivative of the product of quaternion
  // powers w.r.t. each factor.
  // After the loop, `derivative` is the derivative of the curve without the
  // leading multiplication by points_[span_index - degree_].
  derivative.coeffs().setZero();
  for (int k = 0; k < degree_; ++k) {
    //  The derivative of the k-th factor w.r.t. u.
    Quaterniond dkthfactor_du =
        // d(delta_qk*Bck)/dBck
        QuatPowerDerivative(points_[span_index - degree_ + k].inverse() *
                                points_[span_index - degree_ + k + 1],
                            cumulative_basis_[k]);
    // .. * dBck/du.
    dkthfactor_du.coeffs() *= cumulative_basis_derivative_[k];
    // Add contribution of spline curve derivative when differentiating the
    // k-th factor.
    derivative.coeffs() += (quat_powers_[k] * dkthfactor_du *
                            quat_powers_[k + 1].inverse() * quat_powers_.back())
                               .coeffs();
  }

  // Add missing leading quaternion multiplication.
  derivative = points_[span_index - degree_] * derivative;

  // Normalize signs consistently for quaternion and it's derivative.
  if (quat.w() < 0) {
    quat.coeffs() *= -1.0;
    quat.normalize();
    derivative.coeffs() *= -1.0;
  }
  return absl::OkStatus();
}

void BSplineQ::UpdateCumulativeBasis(size_t span_index, double u) {
  const auto& basis = UpdateBasis(span_index, degree_, u);
  // Compute cumulative basis functions that are not identically 1 or 0
  // (see Sec. 4.2. in the paper cited in the header).
  cumulative_basis_[degree_ - 1] = basis[degree_];
  for (int i = degree_ - 2; i >= 0; --i) {
    cumulative_basis_[i] = cumulative_basis_[i + 1] + basis[i + 1];
  }
}

void BSplineQ::UpdateCumulativeBasisAndDerivative(size_t span_index, double u) {
  const auto& basis_and_derivative =
      UpdateBasisAndDerivatives(span_index, degree_, 1, u);
  // Compute cumulative basis functions that are not identically 1 or 0
  // (see Sec. 4.2. in the paper cited in the header).
  cumulative_basis_[degree_ - 1] = basis_and_derivative(0, degree_);
  cumulative_basis_derivative_[degree_ - 1] = basis_and_derivative(1, degree_);
  for (int i = degree_ - 2; i >= 0; --i) {
    cumulative_basis_[i] =
        cumulative_basis_[i + 1] + basis_and_derivative(0, i + 1);
    cumulative_basis_derivative_[i] =
        cumulative_basis_derivative_[i + 1] + basis_and_derivative(1, i + 1);
  }
}

void BSplineQ::UpdateCumulativeBasisAndDerivative(double u) {
  const double clamped_u = std::clamp(u, umin_, umax_);
  const int span_index = KnotSpan(clamped_u);
  UpdateCumulativeBasisAndDerivative(span_index, clamped_u);
}

void BSplineQ::UpdateCumulativeBasis(double u) {
  const double clamped_u = std::clamp(u, umin_, umax_);
  const int span_index = KnotSpan(clamped_u);
  UpdateCumulativeBasis(span_index, clamped_u);
}

absl::Status BSplineQ::InsertKnotAndUpdateControlPoints(double knot,
                                                        int multiplicity) {
  RETURN_IF_ERROR(CanInsertKnot(knot, multiplicity));
  for (int i = 0; i < multiplicity; ++i) {
    RETURN_IF_ERROR(InsertKnotAndUpdateControlPoints(knot));
  }
  return absl::OkStatus();
}

absl::Status BSplineQ::InsertKnotAndUpdateControlPoints(double knot) {
  // Allocates in colPivHouseholderQr if degree_ > 2, see comment there.

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
  // Recompute control points so the curve doesn't change:
  const int point_offset = knot_span - degree_ + 1;
  knots_scratch_.resize(degree_, 0.0);
  const double one_over_degree_plus_one = 1.0 / (degree_ + 1);
  for (int i = 0; i < degree_; ++i) {
    knots_scratch_[i] =
        knot + i * one_over_degree_plus_one * (knots_[knot_span + 1] - knot);
  }

  // Evaluate curve at knots_scratch_[i]. These are used for numerically
  // adjusting the control points if degree_ > 2.
  if (degree_ > 2) {
    spline_reference_values_.resize(kQuatDim * knots_scratch_.size());
    for (int ui = 0; ui < knots_scratch_.size(); ++ui) {
      const double u = knots_scratch_[ui];
      Quaterniond quat;
      CHECK_OK(EvalCurve(u, quat));
      spline_reference_values_.segment(kQuatDim * ui, kQuatDim) = quat.coeffs();
    }
  }

  // For degree_ = 1:
  //   q0*Exp(Bc0*Log(q0^-1*q1)) == hat(q0)*Exp(hat(Bc0)*Log(hat(q0)^1*hat(q1)))
  //                             = hat(q0) // hat(Bc0) == 0 at `knot`.
  //   That is, the new control point is the curve value at `knot`,
  //   which is the same as slerp interpolation.
  // For degree_ = 2, the equations are nonlinear and the solution not obvious:
  // q0*Exp(Bc0*Log(q0^-1*q1))*Exp(Bc1*Log(q1^-1*q2)) ==
  // hat(q0)*Exp(hat(Bc0)*Log(hat(q0)^-1*hat(q1)))
  //  *Exp(hat(Bc1)*Log(hat(q1)^-1*hat(q2))
  //   = hat(q0)*Exp(hat(Bc0)*Log(hat(q0)^-1*hat(q1))) // Bc1 == 0 at `knot`.
  // It happes that using slerp interpolation analogously to knot insertion for
  // the linear case gives a good approximation for higher order splines as
  // well.
  for (int i = 0; i < degree_; ++i) {
    const int k = knot_span + i - degree_ + 1;
    const double alpha = (knot - knots_[k]) / (knots_[k + degree_] - knots_[k]);
    points_scratch_[i] = points_[k - 1].slerp(alpha, points_[k]);
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

  std::copy(points_scratch_.data(), points_scratch_.data() + degree_,
            points_.data() + knot_span - degree_ + 1);

  if (degree_ <= 2) {
    // For degree_ == 1, the linear approximation is exact, for degree_ == 2, it
    // is close enough.
    return absl::OkStatus();
  }

  // If degree_ > 2, we need to solve a set of nonlinear equations iteratively.
  // The Log(SO3) values of the relative quaternions used to compute curve
  // values in the exponential expressions are chosen as unknowns.
  //
  // Note: This doesn't work as well as it should.
  constexpr int kLogSo3Dim = 3;
  eigenmath::VectorXd unknown_points(degree_ * kLogSo3Dim);
  for (int k = 0; k < degree_; ++k) {
    unknown_points.segment(kLogSo3Dim * k, kLogSo3Dim) = eigenmath::LogSO3(
        eigenmath::SO3d(points_[point_offset + k - 1].inverse() *
                        points_[point_offset + k]));
  }
  // Function to evaluate the spline curve using relative log SO3 values.
  const auto curve_values =
      [&,
       this](const eigenmath::VectorXd& point_values) -> eigenmath::VectorXd {
    CHECK_EQ(point_values.size(), kLogSo3Dim * degree_);
    eigenmath::VectorXd curve_vals(spline_reference_values_.size());
    for (int k = 0; k < degree_; ++k) {
      // Compute control point quaternion values from `unknown_points`, roughly:
      // segment = LogSO3(point[k-1]^(-1)*point[k])
      // -> Exp(segment) = point[k-1]^(-1)*point[k])
      // -> point[k] = point[k-1]*Exp(segment).
      points_[point_offset + k] =
          (eigenmath::SO3d(points_[point_offset + k - 1]) *
           eigenmath::ExpSO3(eigenmath::Vector3d(
               point_values.segment(kLogSo3Dim * k, kLogSo3Dim))))
              .quaternion();
    }
    // Evaluate curve at knots_scratch_[i].
    for (int ui = 0; ui < knots_scratch_.size(); ++ui) {
      const double u = knots_scratch_[ui];
      Quaterniond quat;
      CHECK_OK(EvalCurve(u, quat));
      curve_vals.segment(kQuatDim * ui, kQuatDim) = quat.coeffs();
    }

    return curve_vals;
  };

  const auto f =
      [&, this](const eigenmath::VectorXd& values) -> eigenmath::VectorXd {
    CHECK_EQ(values.size(), kLogSo3Dim * degree_);
    return spline_reference_values_ - curve_values(values);
  };

  RETURN_IF_ERROR(Solve(SolveOptions{.max_residual = 1e-6,
                                     .max_iter = 20,
                                     .max_step = 0.1,
                                     .regularization = 0.0,
                                     .finite_difference = 1e-5},
                        f, unknown_points));

  // Update points with last value.
  for (int k = 0; k < degree_; ++k) {
    points_[point_offset + k] =
        (eigenmath::SO3d(points_[point_offset + k - 1]) *
         eigenmath::ExpSO3(eigenmath::Vector3d(
             unknown_points.segment(kLogSo3Dim * k, kLogSo3Dim))))
            .quaternion();
  }

  return absl::OkStatus();
}

absl::Status BSplineQ::TruncateSplineAt(double u_end) {
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

absl::Status BSplineQ::ExtendWithControlPoints(
    absl::Span<const Quaterniond> points) {
  if (degree_ != 2) {
    return absl::UnimplementedError("Only implemented for 2rd order splines.");
  }
  const int new_num_points = num_points_ + points.size();
  const int added_knots = NumKnots(points.size() + 1, degree_) - 2 * degree_;
  const int new_num_knots = num_knots_ + added_knots;

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
    return absl::UnimplementedError("Only implemented for >= 2 points");
  }

  const double u_join = knots_[num_knots_ - 1];
  Quaterniond quat_join;
  EvalCurve(u_join, quat_join).IgnoreError();
  // Generate uniform knot distribution for new control points with a density
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

  const int imod = num_points_ - 1;
  const int span_index = KnotSpan(u_join);
  UpdateCumulativeBasis(span_index, u_join);
  CHECK_GT(cumulative_basis_[0], 0.0);

  Quaterniond quatrel = points_[imod - 1].inverse() * quat_join;
  Quaterniond log_quatrel = bsplineq_details::QuatLog(quatrel);
  log_quatrel.coeffs() /= cumulative_basis_[0];
  if (log_quatrel.w() < 0) {
    log_quatrel.coeffs() *= -1;
  }
  points_[imod] = points_[imod - 1] * bsplineq_details::QuatExp(log_quatrel);

  absl::c_copy(points, points_.begin() + num_points_);

  num_points_ = new_num_points;
  num_knots_ = new_num_knots;
  umax_ = knots_[num_knots_ - 1];

  return absl::OkStatus();
}

}  // namespace trajectory_planning

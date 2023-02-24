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

// Utility classes to compute time derivatives by finite differences.
#ifndef TRAJECTORY_PLANNING_SPLINES_FINITE_DIFFERENCE_TEST_UTILS_H_
#define TRAJECTORY_PLANNING_SPLINES_FINITE_DIFFERENCE_TEST_UTILS_H_

#include <string>

#include "eigenmath/types.h"
#include "absl/strings/string_view.h"

namespace trajectory_planning {
namespace test {

namespace finite_difference_details {

// Converts a DiffType (finite differences) to a ValueType. This is for angular
// velocity calculations via finite differences.
template <typename ValueType, typename DiffType>
DiffType ToDiffType(const ValueType &fd, const ValueType &val);

// Computes the norm of a value.
template <typename DiffType>
double Norm(const DiffType &val);

template <>
inline eigenmath::Vector3d ToDiffType(const eigenmath::Vector3d &fd,
                                      const eigenmath::Vector3d &val) {
  return fd;
}

template <>
inline eigenmath::VectorNd ToDiffType(const eigenmath::VectorNd &fd,
                                      const eigenmath::VectorNd &val) {
  return fd;
}

template <>
inline eigenmath::Vector3d ToDiffType(const eigenmath::Matrix3d &fd,
                                      const eigenmath::Matrix3d &val) {
  // Spin tensor.
  eigenmath::Matrix3d omega_tilde = fd * val.transpose();
  // Extract vector from spin tensor.
  eigenmath::Vector3d omega;
  omega(0) = 0.5 * (omega_tilde(2, 1) - omega_tilde(1, 2));
  omega(1) = 0.5 * (omega_tilde(0, 2) - omega_tilde(2, 0));
  omega(2) = 0.5 * (omega_tilde(1, 0) - omega_tilde(0, 1));
  return omega;
}

template <>
inline double ToDiffType(const double &fd, const double &val) {
  return fd;
}

template <>
inline double Norm(const eigenmath::Vector3d &val) {
  return val.norm();
}

template <>
inline double Norm(const eigenmath::VectorNd &val) {
  return val.norm();
}

template <>
inline double Norm(const double &val) {
  return std::abs(val);
}

// Prints values to stdout.
template <typename DiffType>
void PrintToStdout(const DiffType &val);

template <>
inline void PrintToStdout(const eigenmath::Vector3d &val) {
  printf("%e %e %e", val[0], val[1], val[2]);
}

template <>
inline void PrintToStdout(const eigenmath::VectorNd &val) {
  for (int i = 0; i < val.size(); i++) {
    printf("%e ", val[i]);
  }
}

template <>
inline void PrintToStdout(const double &val) {
  printf("%e ", val);
}

}  // namespace finite_difference_details

// Calculates a finite difference approximation of time derivatives and compares
// it to an analytical solution.
// DiffType and ValueType can be different, to allow comparing angular velocity
// vectors and orientations given as transform matrices.
template <typename ValueType, typename DiffType>
class FiniteDifferenceTest {
 public:
  FiniteDifferenceTest()
      : dt_(0.0),
        num_updates_(0),
        max_error_(0.0),
        max_value_(0.0),
        valid_fd_(false) {}

  // name: string to identify object in messages.
  // dt: timestep at which update() is called.
  void Init(absl::string_view name, double dt) {
    name_ = name;
    dt_ = dt;
    num_updates_ = 0;
    max_error_ = 0.0;
    max_value_ = 0.0;
    avg_error_ = 0.0;
    valid_fd_ = false;
  }

  // Adds new data and update finite difference approximations
  // val: new value
  // true_diff: new (analytical) derivative wrt. time. This is what the
  // finite differences will be compared to.
  void Update(const ValueType &val, const DiffType &true_diff) {
    val_ = val;
    if (num_updates_ > 2) {
      // Second order finite difference approximation for d(value)/dt.
      ValueType diff_value_fd = (val - older_val_) / (2.0 * dt_);
      // Convert to analytical diff type. This is for angular velocities.
      diff_fd_ = finite_difference_details::ToDiffType<ValueType, DiffType>(
          diff_value_fd, old_val_);
      // now, calculate the error
      double error =
          finite_difference_details::Norm<DiffType>(diff_fd_ - old_true_diff_);
      if (error > max_error_) {
        max_error_ = error;
      }

      const double value =
          finite_difference_details::Norm<DiffType>(old_true_diff_);
      if (value > max_value_) {
        max_value_ = value;
      }

      valid_fd_ = true;

      avg_error_ =
          (avg_error_ * (num_updates_ - 3) + error) / (num_updates_ - 2);
    }
    older_val_ = old_val_;
    old_val_ = val_;

    old_true_diff_ = true_diff;
    num_updates_++;
    time_ += dt_;
  }

  void PrintMaxError() {
    printf(
        "%s: max_error: %e avg_error= %e dt= %e max_value= %e fraction= %e\n",
        name_.c_str(), max_error_, avg_error_, dt_, max_value_,
        max_value_ > 0.0 ? max_error_ / max_value_ : 0.0);
  }

  void PrintCurrent() {
    if (valid_fd_) {
      // note: old_true_diff_ already equals true_diff_ here, so values are
      // not aligned.
      //     (but error calculation takes this into account)
      printf("%s time: %e fd: ", name_.c_str(), time_);
      finite_difference_details::PrintToStdout<DiffType>(diff_fd_);
      printf(" true: ");
      finite_difference_details::PrintToStdout<DiffType>(old_true_diff_);
      printf("\n");
    }
  }

  // Returns the maximum error (difference between finite difference and
  // reference value).
  double GetMaxError() const { return max_error_; }

  // Returns the average error (difference between finite difference and
  // reference value).
  double GetAvgError() const { return avg_error_; }

  // Returns the maximum function value passed to Update().
  double GetMaxValue() const { return max_value_; }

 private:
  double dt_;
  ValueType val_;
  ValueType old_val_;
  ValueType older_val_;
  DiffType old_true_diff_;
  DiffType diff_fd_;
  int num_updates_;
  double max_error_;
  double avg_error_;
  double max_value_;
  double time_;
  std::string name_;
  bool valid_fd_;
};
}  // namespace test
}  // namespace trajectory_planning

#endif  // TRAJECTORY_PLANNING_SPLINES_FINITE_DIFFERENCE_TEST_UTILS_H_

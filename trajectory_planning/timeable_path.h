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

#ifndef TRAJECTORY_PLANNING_TIMEABLE_PATH_H_
#define TRAJECTORY_PLANNING_TIMEABLE_PATH_H_

#include <vector>

#include "eigenmath/types.h"
#include "trajectory_planning/time_optimal_path_timing.h"
#include "absl/status/status.h"
#include "absl/types/span.h"

namespace trajectory_planning {

// Wrap an eigen vector <double> in an absl::Span.
template <typename EigenVector>
absl::Span<const double> ToSpan(const EigenVector& vec) {
  return absl::Span<const double>(vec.array().data(), vec.array().size());
}

// Map an absl::Span to an eigenmath::VectorXd.
// Assigning directly to an eigenmath::VectorXd will make a copy, e.g.:
//   eigenmath::VectorXd vector_copy = FromSpan(span)
// To avoid a copy, use inside an expression or assign to a map, e.g.:
//   Eigen::Map<const eigenmath::VectorXd> vector = FromSpan(span)
inline Eigen::Map<const eigenmath::VectorXd> FromSpan(
    absl::Span<const double> span) {
  return Eigen::Map<const eigenmath::VectorXd>(span.data(), span.size());
}

// Common options for timeable paths.
template <typename DerivedOptions>
class PathOptions {
 public:
  PathOptions() {
    static_assert(
        std::is_base_of<PathOptions<DerivedOptions>, DerivedOptions>::value,
        "Template parameter for PathOptions must be a derived class.");
  }
  double constraint_safety() const { return constraint_safety_; }
  DerivedOptions& set_constraint_safety(double safety) {
    constraint_safety_ = safety;
    return static_cast<DerivedOptions&>(*this);
  }
  double rounding() const { return rounding_; }
  DerivedOptions& set_rounding(double rounding) {
    rounding_ = rounding;
    return static_cast<DerivedOptions&>(*this);
  }
  size_t num_dofs() const { return num_dofs_; }
  DerivedOptions& set_num_dofs(size_t dofs) {
    num_dofs_ = dofs;
    return static_cast<DerivedOptions&>(*this);
  }
  size_t num_path_samples() const { return num_path_samples_; }
  DerivedOptions& set_num_path_samples(size_t num_samples) {
    num_path_samples_ = num_samples;
    return static_cast<DerivedOptions&>(*this);
  }
  double delta_parameter() const { return delta_parameter_; }
  DerivedOptions& set_delta_parameter(double delta_parameter) {
    delta_parameter_ = delta_parameter;
    return static_cast<DerivedOptions&>(*this);
  }

 private:
  // Safety factor for path constraints.
  double constraint_safety_ = 0.8;
  // Rounding of corners when converting waypoints to splines.
  double rounding_ = 0.2;
  // Number of joints / degrees of freedom.
  size_t num_dofs_ = 0;
  // Number of samples along the planning horizon (uniformly spaced along the
  // path).
  size_t num_path_samples_ = 500;
  // Distance between the path path parameter values of two path samples.
  double delta_parameter_ = 0.005;
};

class TimeablePath {
 public:
  enum class State {
    // No path yet or Reset was called.
    kNoPath,
    // New path waypoints were set, but path was not yet sampled.
    kNewPath,
    // An existing path was modified, and not yet re-sampled.
    kModifiedPath,
    // The path has not been modified since SamplePath() was last called.
    kPathWasSampled
  };

  virtual ~TimeablePath() = default;

  // Set maximum admissible joint velocity.
  virtual absl::Status SetMaxJointVelocity(
      absl::Span<const double> max_velocity) = 0;
  // Set maximum admissible joint acceleration.
  virtual absl::Status SetMaxJointAcceleration(
      absl::Span<const double> max_acceleration) = 0;
  // Returns maximum joint velocity settings.
  virtual const eigenmath::VectorXd& GetMaxJointVelocity() const = 0;
  // Returns maximum joint acceleration settings.
  virtual const eigenmath::VectorXd& GetMaxJointAcceleration() const = 0;
  // Set an initial joint velocity to be used as initial condition when solving
  // the trajectory planning problem. The default is zero.
  // Non-zero initial velocity may be unsupported by some implementations.
  // The provided initial velocity must be parallel to the initial path tangent.
  virtual absl::Status SetInitialVelocity(
      absl::Span<const double> max_acceleration) = 0;
  // Returns the requested initial velocity. Implementations that do not support
  // SetInitialVelocity return a zero vector.
  virtual const eigenmath::VectorXd& GetInitialVelocity() const = 0;
  // Returns true if path(parameter) is close to the path end.
  virtual bool CloseToEnd(double parameter) const = 0;
  // Returns the current State.
  virtual State GetState() const = 0;
  // Generate path samples from path_start to path_start+path_horizon
  virtual absl::Status SamplePath(double path_start) = 0;
  virtual int GetNumPathSamples() const = 0;
  virtual double GetPathSamplingDistance() const = 0;
  // Compute path constraints for profile optimization.
  virtual absl::Status ConstraintSetup() = 0;
  virtual const std::vector<TimeOptimalPathProfile::Constraint>&
  GetConstraints() const = 0;
  virtual size_t NumConstraints() const = 0;
  virtual size_t NumDofs() const = 0;
  virtual size_t NumPathSamples() const = 0;
  // Resets the generator and clears the waypoint queue.
  // Trajectories planned after calling this will start at the first
  // waypoint added after the call.
  virtual void Reset() = 0;
  // Return the first active waypoint. Asserts on empty waypoint queue.
  virtual const eigenmath::VectorXd& GetPathStart() const = 0;
  // Return the last active waypoint. Asserts on empty waypoint queue.
  virtual const eigenmath::VectorXd& GetPathEnd() const = 0;
  // Return the start of the path parameter range used in the previous call
  // to SamplePath(). Returns NaN if unavailable.
  virtual double GetParameterStart() const = 0;
  // Return the end of the path parameter range used in the previous call
  // to SamplePath(). Returns NaN if unavailable.
  virtual double GetParameterEnd() const = 0;

  // Returns the n-th sampled path position. Asserts if n is out of range.
  virtual const eigenmath::VectorXd& GetPathPositionAt(size_t n) const = 0;
  // Returns the n-th sample of the first path derivative. Asserts if n is out
  // of range.
  virtual const eigenmath::VectorXd& GetFirstPathDerivativeAt(
      size_t n) const = 0;
  // Returns the n-th sample of the second path derivative. Asserts if n is out
  // of range.
  virtual const eigenmath::VectorXd& GetSecondPathDerivativeAt(
      size_t n) const = 0;
};

// Returns a string representation of `state`, or a string indicating an error
// if `state` is invalid.
inline constexpr const char* ToString(const TimeablePath::State state) {
#define HANDLE_CASE(x)         \
  case TimeablePath::State::x: \
    return #x
  switch (state) {
    HANDLE_CASE(kNoPath);
    HANDLE_CASE(kNewPath);
    HANDLE_CASE(kModifiedPath);
    HANDLE_CASE(kPathWasSampled);
  }
  return "Invalid TimeablePath::State enum value";
#undef HANDLE_CASE
}

}  // namespace trajectory_planning

#endif  // TRAJECTORY_PLANNING_TIMEABLE_PATH_H_

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

#ifndef TRAJECTORY_PLANNING_TIME_OPTIMAL_PATH_TIMING_H_
#define TRAJECTORY_PLANNING_TIME_OPTIMAL_PATH_TIMING_H_

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "Eigen/Dense"

namespace trajectory_planning {

// For a given path and an unknown path parameter as a function of time
// [s(t)], this class computes the minimum time profile under a set of velocity
// and acceleration (or torque) constraints.
// Naming convention: s: path parameter, sd=ds/dt, sdd= d^2s/dt^2, sd2=(sd)^2,
// sd2p = d(sd2)/ds.
// This is an implementation of the approach described by Pfeiffer & Johanni
// in: "A concept for manipulator trajectory planning", ICRA 1986
// https://doi.org/10.1109/ROBOT.1986.1087500
class TimeOptimalPathProfile {
 public:
  // Scalar type to use for computations.
  // NOTE: This must currently be double, the implementation doesn't work for
  // single precision.
  using Scalar = double;

  // Verbosity levels for debug messages
  enum DebugVerbosity {
    // No debug output.
    kNoOutput = 0,
    // Log high level messages, stage of main algorithm.
    kMainAlgorithm,
    // Log status in main extremal construction loop.
    kExtremalLoop,
    // Log rough extremal construction info (stopping reason, etc.).
    kExtremalControl,
    // Log detailed extremal construction info (at each integration
    // step).
    kExtremalDetail,
    // Log all available debug messages, e.g. mathematica code for solving lp
    // subproblems.
    kAll
  };

  // Constraint definition for timing profile.
  // This defines the set of constraints that are applied to a sample
  // point along the path, i.e., are valid for a fixed path parameter s.
  // All constraints have the form: lower <= A*sdd+B*sd^2 <= upper
  class Constraint {
   public:
    void resize(int size) {
      data_.resize(size, kColumns);
      data_ = 0;
    }
    int size() const { return data_.rows(); }
    auto lower() { return data_.col(kLowerIndex); }
    const auto lower() const { return data_.col(kLowerIndex); }
    Scalar &lower(int index) { return data_(index, kLowerIndex); }
    const Scalar &lower(int index) const { return data_(index, kLowerIndex); }
    auto upper() { return data_.col(kUpperIndex); }
    const auto upper() const { return data_.col(kUpperIndex); }
    Scalar &upper(int index) { return data_(index, kUpperIndex); }
    const Scalar &upper(int index) const { return data_(index, kUpperIndex); }
    auto a_coefficient() { return data_.col(kAIndex); }
    const auto a_coefficient() const { return data_.col(kAIndex); }
    Scalar &a_coefficient(int index) { return data_(index, kAIndex); }
    const Scalar &a_coefficient(int index) const {
      return data_(index, kAIndex);
    }
    auto b_coefficient() { return data_.col(kBIndex); }
    const auto b_coefficient() const { return data_.col(kBIndex); }
    Scalar &b_coefficient(int index) { return data_(index, kBIndex); }
    const Scalar &b_coefficient(int index) const {
      return data_(index, kBIndex);
    }

   private:
    enum {
      kAIndex = 0,
      kBIndex = 1,
      kLowerIndex = 2,
      kUpperIndex = 3,
      kColumns = 4
    };
    Eigen::Array<Scalar, Eigen::Dynamic, kColumns> data_;
  };

  TimeOptimalPathProfile() = default;
  ~TimeOptimalPathProfile() = default;

  // set the level of debug output.
  // level: debug level (corresponding to enum DebugVerbosity).
  static void SetDebugVerbosity(int level);

  // Print description of problem setup for debugging.
  void PrintProblemDebugInfo();

  // Initialize solver & allocate memory; non real-time safe.
  // num_samples_: number of samples to use (along the path)
  // num_constraints_: number of constraints at each sample point for the
  // problem.
  bool InitSolver(int num_samples_, int num_constraints_);

  // Set constraints and other problem parameters.
  // This also tells the solver that problem setup is done.
  // constraints: a vector of constraints for the problem at each sample point.
  // s_start: initial path parameter value
  // s_end: final path parameter value
  // sd_start: initial path parameter derivative
  // sdd_start: initial second path parameter derivative.
  //            If (sd_start, sdd_start) is invalid, the largest valid sdd_start
  //            will be used.
  // time_start: initial time
  bool SetupProblem(const std::vector<Constraint> &constraints, Scalar s_start,
                    Scalar s_end, Scalar sd_start, Scalar sdd_start,
                    Scalar time_start);

  // Set/change maximum admissible number of iterations in the solver.
  // num_loops: maximum number of loops in the solver.
  void SetMaxNumSolverLoops(int num_loops);

  // Write data for forward and backward extremals to a text file.
  // For debugging only, not real-time safe, will allocate.
  // filename: file path to save extremals to.
  bool PlotAllExtremals(const char *filename);

  // Write data for optimal solution to file.
  // For debugging only, not real-time safe, will allocate.
  // filename: file path to save extremals to
  bool PlotSolution(const char *filename);

  // Calculate time optimal path parameter trajectory.
  // Returns false on error, true on success
  bool OptimizePathParameter();

  // Get vector of sampling times for path parameter samples.
  const Eigen::ArrayX<Scalar> &GetTimeSamples() const { return time_; }
  // get Eigen::ArrayX<Scalar> of path parameter.
  const Eigen::ArrayX<Scalar> &GetPathParameter() const { return s_; }
  // get vector of path parameter velocities.
  const Eigen::ArrayX<Scalar> &GetPathVelocity() const { return sd_; }
  // get vector of path parameter accelerations.
  const Eigen::ArrayX<Scalar> &GetPathAcceleration() const { return sdd_; }

  // Get path parameter and derivative by interpolating discrete parameter
  // (time) solution.
  // t: time.
  // s: path parameter for input t.
  // sd: path parameter speed for input t.
  // sdd: path parameter acceleration for input t.
  // Returns true on success, false on error.
  bool GetPathParameterAndDerivatives(Scalar t, Scalar *s, Scalar *sd,
                                      Scalar *sdd) const;

  // Returns the total duration of timing profile.
  Scalar GetTotalDuration() const { return time_[time_.size() - 1] - time_[0]; }

  // Returns the end time for the timing profile.
  Scalar GetEndTime() const { return time_[time_.size() - 1]; }

  // Returns the starting time for the timing profile.
  Scalar GetStartTime() const { return time_[0]; }

  // Get values from discretization point before t.
  // t:    time value.
  // sk:   path parameter at previous sampling point.
  // sdk:  path parameter velocity at previous sampling point.
  // sddk: path parameter acceleration at previous sampling point.
  // tk:   time value at previous sampling point.
  // Returns true on success, false on error.
  bool GetPreviousDiscreteValues(Scalar t, Scalar *sk, Scalar *sdk,
                                 Scalar *sddk, Scalar *tk) const;

  // Get index for samples before t.
  // return -1 if out of range.
  int GetPreviousIndex(Scalar t) const;

  // Get the largest time increment for the solution.
  // Return negative value on error.
  Scalar GetMaxTimeIncrement() const;

  // Find maximum admissible sd2 ((ds/dt)^2) for constr.
  // Only public for testability reasons.
  void FindMaxSd2BruteForce(const Constraint &constr, Scalar *sd2max,
                            Scalar *sddmax, Scalar *sd2zero);

  // Find maximum admissible sd2 ((ds/dt)^2) for constr using simplex approach.
  void FindMaxSd2Simplex(const Constraint &constr, Scalar *sd2max,
                         Scalar *sddmax, Scalar *sd2zero);
  void PrintFindMaxSd2MathematicaDebugCode(const Constraint &constr) const;

  // Returns the index where the backward extremal constituting the final
  // section of the solution begins.
  int GetLastExtremalIndex() const { return last_extremal_index_; }

  // Returns OkStatus() if all constraints are satisfied for the solution, and
  // an non-ok status otherwise.
  absl::Status SolutionSatisfiesConstraints();

 private:
  enum SolverState {
    kInvalidState = 0,
    kAllocated,
    kProblemDefined,
    kProblemSolved
  };
  const char *ToString(const SolverState &state);

  struct Boundary {
    enum : uint8_t {
      kNone = 0,
      kSource = 1,
      kSink = 2,
      kTrajectory = 4,
    };
    Eigen::ArrayX<Scalar> sd2_max;  // max. velocity squared: the boundary curve
    Eigen::ArrayX<Scalar> sdd_max_for_sd2_max;  // max. acceleration at sd_max
    Eigen::ArrayX<Scalar> sdd_min_for_sd2_max;  // min. acceleration at sd_max
    Eigen::ArrayX<Scalar>
        sd2_max_for_sdd0;  // max. velocity squared at zero acceleration
    Eigen::ArrayX<bool>
        sd2_max_at_sdd0;          // max. velocity is at zero acceleration
    Eigen::ArrayX<uint8_t> type;  // classifiction of boundary

    void resize(int sz) {
      sd2_max.resize(sz);
      sdd_max_for_sd2_max.resize(sz);
      sdd_min_for_sd2_max.resize(sz);
      sd2_max_for_sdd0.resize(sz);
      sd2_max_at_sdd0.resize(sz);
      type.resize(sz);
      sd2_max = 0.0;
      sdd_max_for_sd2_max = 0.0;
      sdd_min_for_sd2_max = 0.0;
      type = Boundary::kNone;
    }
    int size() const { return sd2_max.size(); }
    static const char *ToString(uint8_t type);
  };

  enum SolutionType {
    // Not set.
    kNoSolution = 0,
    // Forward extremal.
    kForward,
    // Backward extremal.
    kBackward,
    // On boundary.
    kBoundary,
  };

  enum ConstraintType { kNotSet = 0, kUpper = 1, kLower = 2 };
  struct ActiveConstraint {
    int index;
    ConstraintType type;
    Scalar slope;
  };

  static constexpr Scalar kTiny = std::numeric_limits<Scalar>::epsilon() * 1e5;
  // Maximum velocity squared (used for unbounded cases).
  // This is currently set heuristically, but it would be better to compute it
  // in from discretization parameters.
  static constexpr Scalar kMaxSd2 = 1e6;

  bool IsTiny(const Scalar &value) const { return std::abs(value) < kTiny; }

  // Set i-th constraint.
  // constr: the constraint to set
  // i: index for the constraint
  bool SetIthConstraint(const Constraint &constr, int i);

  // Signal that setup is done. returns false on error.
  bool SetSetupDone();
  bool IsSetupValid() const;

  // Rescale solution to remove any constraint violation.
  bool RescaleSolution();
  bool AreDerivativesValid(int idx, Scalar sdd, Scalar sd2) const;
  Scalar FindSddMax(int idx, Scalar sd2) const;
  Scalar FindSddMin(int idx, Scalar sd2) const;
  int NextCriticalPoint(int idx_lo, int idx_hi) const;
  // Computes sdd at the extremal intersection at `index`.
  void ComputeSddAtIntersection(int index, const char *msg);
  // Returns the pair {sdd at index, sd2 at index+1} from one forward
  // integration step along an extremal.
  std::pair<Scalar, Scalar> OneForwardExtremalStep(int index, Scalar sd2);
  // Returns the pair {sdd at index, sd2 at index-1} from one backward
  // integration step along an extremal.
  std::pair<Scalar, Scalar> OneBackwardExtremalStep(int index, Scalar sd2);
  int AddForwardExtremal(int idx_lo);
  int AddBackwardExtremal(int idx_hi);
  // compute intersection of two constraint boundaries given by
  // A1*sdd+B1*sp2 == e1 && A2*sdd+B2*sp2 == e2
  bool Intersect(const Scalar &A1, const Scalar &B1, const Scalar &e1,
                 const Scalar &A2, const Scalar &B2, const Scalar &e2,
                 Scalar *sdd, Scalar *sp2) const;
  bool CalculateBoundary();
  void Plot(FILE *fp, const char *str, int idx0, int idx1);
  int SampleIndexFromTime(Scalar t) const;
  bool ConstraintIsValid(const Constraint &constr, Scalar sdd, Scalar sd2,
                         Eigen::ArrayX<Scalar> *work);
  bool OnBoundary(int idx, Scalar sd2) const {
    return IsTiny(sd2_[idx] - boundary_.sd2_max[idx]);
  }
  bool AboveBoundary(int idx, Scalar sd2) const {
    return sd2 > boundary_.sd2_max[idx];
  }
  bool IsOptimal(const Constraint &constr, int first_index, int second_index,
                 ConstraintType first_type, ConstraintType second_type);

  void DebugLog(DebugVerbosity log_level, const char *fmt, ...);

  SolverState solver_state_ = kInvalidState;
  int num_constraints_ = 0;
  int num_samples_ = 0;
  Scalar s_start_;
  Scalar sd_start_;
  Scalar sdd_start_;
  Scalar time_start_;
  Scalar s_end_;
  std::vector<Constraint> constraints_;
  Boundary boundary_;
  int max_num_loops_ = 100;
  Eigen::ArrayX<Scalar> time_;
  Eigen::ArrayX<Scalar> sd2_;
  Eigen::ArrayX<Scalar> sd_;
  Eigen::ArrayX<Scalar> s_;
  Eigen::ArrayX<Scalar> sdd_;
  std::vector<SolutionType> solution_type_;
  Scalar ds_;
  Scalar inv_ds_;
  Eigen::ArrayX<Scalar> constr_work_;
  std::vector<std::pair<int, Scalar>> index_value_work_;
  static DebugVerbosity debug_;
  Scalar dt_max_;
  int low_idx_;
  int high_idx_;
  std::vector<std::pair<int, int>> constraint_set_;
  std::vector<ActiveConstraint> active_set_;
  int last_extremal_index_;
};

}  // namespace trajectory_planning

#endif  // TRAJECTORY_PLANNING_TIME_OPTIMAL_PATH_TIMING_H_

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

#include "trajectory_planning/splines/bspline.h"

#include <array>
#include <initializer_list>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "eigenmath/matchers.h"
#include "eigenmath/types.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "trajectory_planning/splines/spline_utils.h"

ABSL_FLAG(int32_t, verbosity, 0, "verbosity level");

namespace trajectory_planning {
namespace {
using ArrayXd = ::Eigen::ArrayXd;
using Map = ::Eigen::Map<const ArrayXd>;
using ::eigenmath::Vector2d;
using ::eigenmath::testing::IsApprox;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;

#define ASSERT_OK(x) ASSERT_TRUE(x.ok());

ArrayXd MakeArrayXd(std::initializer_list<double> data) {
  return Map(data.begin(), data.size());
}

template <typename Traits>
class BSplineGoldenTest : public testing::Test {
 public:
  void SetUp() override {
    golden_data_x_ = Eigen::Map<const ArrayXd>(kGoldenDataX, GetNumSamples());
    golden_data_xp_ = Eigen::Map<const ArrayXd>(kGoldenDataXp, GetNumSamples());
    golden_data_xpp_ =
        Eigen::Map<const ArrayXd>(kGoldenDataXpp, GetNumSamples());
    golden_data_xppp_ =
        Eigen::Map<const ArrayXd>(kGoldenDataXppp, GetNumSamples());

    golden_data_y_ = Eigen::Map<const ArrayXd>(kGoldenDataY, GetNumSamples());
    golden_data_yp_ = Eigen::Map<const ArrayXd>(kGoldenDataYp, GetNumSamples());
    golden_data_ypp_ =
        Eigen::Map<const ArrayXd>(kGoldenDataYpp, GetNumSamples());
    golden_data_yppp_ =
        Eigen::Map<const ArrayXd>(kGoldenDataYppp, GetNumSamples());
  }

  static constexpr double GetGoldenEpsilon() {
    // This value is tuned to make test pass on ARM64.
    // (The "golden" dataset is generated on x86, and
    // kGoldenEpsilon=std::Numeric_limits<double>::epsilon() works there.)
    return 5e-14;
  }

  static constexpr int GetNumSamples() { return 101; }

  ArrayXd golden_data_x_;
  ArrayXd golden_data_xp_;
  ArrayXd golden_data_xpp_;
  ArrayXd golden_data_xppp_;
  ArrayXd golden_data_y_;
  ArrayXd golden_data_yp_;
  ArrayXd golden_data_ypp_;
  ArrayXd golden_data_yppp_;

  // Golden reference data for bsplines.
  // The data was generated using Mathematica's BSpline function, with the
  // following code:
  // f = BSplineFunction[{{1, 1}, {2, 3}, {3, -1}, {4, 1}, {5, 0}}]
  // Export["outX.dat", Table[f[t][[1]], {t, 0, 1, 0.01}]]
  // Export["outY.dat", Table[f[t][[2]], {t, 0, 1, 0.01}]]
  // Export["outXp.dat", Table[f'[t][[1]], {t, 0, 1, 0.01}]]
  // Export["outYp.dat", Table[f'[t][[2]], {t, 0, 1, 0.01}]]
  // etc.
  static constexpr double kGoldenDataX[GetNumSamples()] = {1.,
                                                           1.059404,
                                                           1.117632,
                                                           1.1747079999999999,
                                                           1.230656,
                                                           1.2855,
                                                           1.339264,
                                                           1.3919720000000002,
                                                           1.443648,
                                                           1.4943160000000002,
                                                           1.5440000000000005,
                                                           1.592724,
                                                           1.640512,
                                                           1.6873880000000003,
                                                           1.7333760000000002,
                                                           1.7784999999999995,
                                                           1.8227839999999997,
                                                           1.8662519999999998,
                                                           1.9089280000000004,
                                                           1.9508360000000002,
                                                           1.9920000000000004,
                                                           2.0324440000000004,
                                                           2.0721920000000003,
                                                           2.111268,
                                                           2.149696,
                                                           2.1875,
                                                           2.224704,
                                                           2.261332,
                                                           2.2974080000000003,
                                                           2.332956,
                                                           2.368,
                                                           2.402564,
                                                           2.4366719999999997,
                                                           2.4703479999999995,
                                                           2.503616,
                                                           2.5364999999999998,
                                                           2.5690239999999998,
                                                           2.601212,
                                                           2.633088,
                                                           2.664676,
                                                           2.6959999999999997,
                                                           2.727084,
                                                           2.757952,
                                                           2.788628,
                                                           2.819136,
                                                           2.8495000000000004,
                                                           2.8797439999999996,
                                                           2.909892,
                                                           2.9399680000000004,
                                                           2.969996,
                                                           3.,
                                                           3.0300040000000004,
                                                           3.060032,
                                                           3.090108,
                                                           3.1202559999999995,
                                                           3.1505,
                                                           3.180864,
                                                           3.211372,
                                                           3.242048,
                                                           3.272916,
                                                           3.3039999999999994,
                                                           3.3353240000000004,
                                                           3.3669119999999997,
                                                           3.3987880000000006,
                                                           3.430976,
                                                           3.4635000000000002,
                                                           3.496384,
                                                           3.5296519999999996,
                                                           3.563328,
                                                           3.597436,
                                                           3.6319999999999997,
                                                           3.667044,
                                                           3.702592,
                                                           3.7386679999999997,
                                                           3.7752960000000004,
                                                           3.8125,
                                                           3.8503040000000004,
                                                           3.888732,
                                                           3.9278079999999997,
                                                           3.967556,
                                                           4.008000000000001,
                                                           4.049164,
                                                           4.0910720000000005,
                                                           4.133748000000001,
                                                           4.177216,
                                                           4.2215,
                                                           4.266624,
                                                           4.312612,
                                                           4.359488000000001,
                                                           4.4072759999999995,
                                                           4.456,
                                                           4.505684,
                                                           4.556352,
                                                           4.608028,
                                                           4.660736,
                                                           4.7145,
                                                           4.769344,
                                                           4.825291999999999,
                                                           4.8823680000000005,
                                                           4.940595999999999,
                                                           5.};
  static constexpr double kGoldenDataY[GetNumSamples()] = {1.,
                                                           1.1152440000000001,
                                                           1.2211520000000002,
                                                           1.3179879999999997,
                                                           1.406016,
                                                           1.4855000000000003,
                                                           1.5567039999999999,
                                                           1.6198919999999999,
                                                           1.6753279999999997,
                                                           1.7232760000000003,
                                                           1.7640000000000005,
                                                           1.7977640000000001,
                                                           1.8248319999999998,
                                                           1.8454680000000003,
                                                           1.8599360000000003,
                                                           1.8684999999999998,
                                                           1.8714239999999995,
                                                           1.8689719999999996,
                                                           1.861408,
                                                           1.848996,
                                                           1.8320000000000003,
                                                           1.8106840000000002,
                                                           1.785312,
                                                           1.7561480000000003,
                                                           1.7234560000000003,
                                                           1.6875,
                                                           1.648544,
                                                           1.606852,
                                                           1.5626879999999999,
                                                           1.516316,
                                                           1.4680000000000002,
                                                           1.418004,
                                                           1.3665919999999996,
                                                           1.3140279999999998,
                                                           1.2605759999999995,
                                                           1.2064999999999992,
                                                           1.152064,
                                                           1.0975320000000002,
                                                           1.0431679999999999,
                                                           0.989236,
                                                           0.9359999999999998,
                                                           0.883724,
                                                           0.8326720000000003,
                                                           0.7831080000000001,
                                                           0.7352960000000001,
                                                           0.6895000000000002,
                                                           0.6459840000000001,
                                                           0.6050120000000001,
                                                           0.5668479999999998,
                                                           0.5317560000000002,
                                                           0.5,
                                                           0.471772,
                                                           0.4469759999999999,
                                                           0.425444,
                                                           0.4070079999999999,
                                                           0.39149999999999996,
                                                           0.37875199999999987,
                                                           0.368596,
                                                           0.360864,
                                                           0.355388,
                                                           0.3520000000000001,
                                                           0.350532,
                                                           0.350816,
                                                           0.35268400000000005,
                                                           0.35596799999999995,
                                                           0.3605,
                                                           0.366112,
                                                           0.37263599999999997,
                                                           0.379904,
                                                           0.38774800000000004,
                                                           0.3960000000000001,
                                                           0.40449199999999996,
                                                           0.41305600000000003,
                                                           0.4215239999999999,
                                                           0.42972799999999994,
                                                           0.4375,
                                                           0.44467200000000007,
                                                           0.45107600000000003,
                                                           0.45654399999999995,
                                                           0.46090800000000004,
                                                           0.464,
                                                           0.46565199999999995,
                                                           0.465696,
                                                           0.463964,
                                                           0.46028800000000003,
                                                           0.4545,
                                                           0.44643200000000005,
                                                           0.4359160000000001,
                                                           0.422784,
                                                           0.40686799999999995,
                                                           0.388,
                                                           0.3660119999999999,
                                                           0.3407359999999999,
                                                           0.31200399999999984,
                                                           0.2796479999999998,
                                                           0.24349999999999974,
                                                           0.20339200000000013,
                                                           0.15915600000000013,
                                                           0.1106240000000001,
                                                           0.05762800000000005,
                                                           0.};

  static constexpr double kGoldenDataXp[GetNumSamples()] = {6.,
                                                            5.8812,
                                                            5.7648,
                                                            5.650799999999999,
                                                            5.5392,
                                                            5.429999999999999,
                                                            5.323200000000001,
                                                            5.218799999999999,
                                                            5.1168000000000005,
                                                            5.017200000000001,
                                                            4.920000000000001,
                                                            4.825200000000001,
                                                            4.7328,
                                                            4.6428,
                                                            4.555199999999999,
                                                            4.470000000000001,
                                                            4.387199999999999,
                                                            4.306799999999999,
                                                            4.2288,
                                                            4.1532,
                                                            4.08,
                                                            4.0092,
                                                            3.9408000000000003,
                                                            3.874800000000001,
                                                            3.811200000000001,
                                                            3.75,
                                                            3.6912000000000007,
                                                            3.6348000000000003,
                                                            3.5807999999999995,
                                                            3.5291999999999994,
                                                            3.4800000000000004,
                                                            3.4331999999999994,
                                                            3.3888000000000003,
                                                            3.3468000000000004,
                                                            3.307199999999999,
                                                            3.269999999999999,
                                                            3.2352,
                                                            3.2028,
                                                            3.1727999999999996,
                                                            3.1452,
                                                            3.12,
                                                            3.0972,
                                                            3.0768,
                                                            3.0588,
                                                            3.0432000000000006,
                                                            3.0300000000000002,
                                                            3.0192000000000005,
                                                            3.0108,
                                                            3.0048000000000004,
                                                            3.0012000000000003,
                                                            3.,
                                                            3.0011999999999994,
                                                            3.0048,
                                                            3.0107999999999997,
                                                            3.0192,
                                                            3.030000000000001,
                                                            3.043200000000001,
                                                            3.058799999999999,
                                                            3.0768,
                                                            3.0972,
                                                            3.119999999999999,
                                                            3.1452,
                                                            3.1728,
                                                            3.2028000000000008,
                                                            3.2352,
                                                            3.270000000000001,
                                                            3.3072000000000004,
                                                            3.3468,
                                                            3.3888000000000007,
                                                            3.4332000000000003,
                                                            3.4800000000000004,
                                                            3.5292,
                                                            3.5808,
                                                            3.6348,
                                                            3.6912000000000007,
                                                            3.75,
                                                            3.8112000000000013,
                                                            3.8748000000000005,
                                                            3.9407999999999994,
                                                            4.0092,
                                                            4.08,
                                                            4.1532,
                                                            4.2288000000000014,
                                                            4.306800000000001,
                                                            4.3872,
                                                            4.469999999999997,
                                                            4.555199999999999,
                                                            4.642799999999998,
                                                            4.732800000000003,
                                                            4.825200000000001,
                                                            4.92,
                                                            5.017200000000001,
                                                            5.116800000000005,
                                                            5.218799999999998,
                                                            5.3232000000000035,
                                                            5.429999999999996,
                                                            5.5391999999999975,
                                                            5.6508,
                                                            5.764800000000005,
                                                            5.8812,
                                                            6.};
  static constexpr double kGoldenDataXpp[GetNumSamples()] = {
      -12.,
      -11.760000000000009,
      -11.520000000000003,
      -11.28000000000001,
      -11.039999999999997,
      -10.800000000000002,
      -10.559999999999995,
      -10.320000000000007,
      -10.080000000000005,
      -9.839999999999996,
      -9.599999999999998,
      -9.359999999999992,
      -9.120000000000003,
      -8.879999999999997,
      -8.640000000000004,
      -8.400000000000002,
      -8.160000000000007,
      -7.920000000000005,
      -7.679999999999996,
      -7.439999999999998,
      -7.199999999999999,
      -6.959999999999999,
      -6.720000000000001,
      -6.479999999999995,
      -6.239999999999998,
      -6.,
      -5.760000000000002,
      -5.52,
      -5.279999999999998,
      -5.040000000000003,
      -4.800000000000001,
      -4.560000000000002,
      -4.320000000000004,
      -4.080000000000002,
      -3.8400000000000034,
      -3.6000000000000014,
      -3.3599999999999994,
      -3.1200000000000045,
      -2.8799999999999955,
      -2.640000000000004,
      -2.399999999999995,
      -2.1600000000000037,
      -1.9200000000000017,
      -1.6800000000000033,
      -1.4400000000000013,
      -1.2000000000000028,
      -0.9600000000000009,
      -0.7200000000000024,
      -0.4800000000000004,
      -0.24000000000000554,
      0.,
      0.23999999999999488,
      0.4800000000000004,
      0.7199999999999989,
      0.9600000000000009,
      1.2000000000000028,
      1.4400000000000013,
      1.6800000000000015,
      1.9200000000000017,
      2.1599999999999966,
      2.3999999999999986,
      2.6400000000000006,
      2.879999999999999,
      3.120000000000001,
      3.3599999999999994,
      3.6000000000000014,
      3.8399999999999963,
      4.080000000000005,
      4.32,
      4.560000000000002,
      4.799999999999997,
      5.039999999999999,
      5.280000000000008,
      5.520000000000003,
      5.759999999999991,
      6.,
      6.240000000000009,
      6.480000000000018,
      6.719999999999992,
      6.960000000000001,
      7.200000000000003,
      7.4399999999999835,
      7.679999999999993,
      7.920000000000002,
      8.159999999999997,
      8.399999999999991,
      8.64,
      8.879999999999995,
      9.120000000000005,
      9.360000000000014,
      9.600000000000009,
      9.840000000000003,
      10.079999999999998,
      10.319999999999993,
      10.560000000000002,
      10.799999999999997,
      11.039999999999978,
      11.279999999999987,
      11.519999999999982,
      11.76000000000002,
      12.};
  static constexpr double kGoldenDataXppp[GetNumSamples()] = {
      24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.,
      24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.,
      24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.,
      24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.,
      24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.,
      24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.,
      24., 24., 24., 24., 24., 24., 24., 24., 24., 24., 24.};
  static constexpr double kGoldenDataYp[GetNumSamples()] = {
      12.,
      11.053200000000002,
      10.1328,
      9.2388,
      8.371200000000002,
      7.529999999999999,
      6.715200000000002,
      5.926799999999997,
      5.1648000000000005,
      4.4292000000000025,
      3.72,
      3.037200000000001,
      2.3807999999999994,
      1.7507999999999997,
      1.1471999999999998,
      0.5699999999999994,
      0.019199999999999717,
      -0.5052000000000005,
      -1.0032000000000008,
      -1.4747999999999997,
      -1.920000000000001,
      -2.338799999999999,
      -2.7311999999999994,
      -3.0972,
      -3.4368000000000003,
      -3.75,
      -4.036800000000001,
      -4.297200000000001,
      -4.5312,
      -4.7387999999999995,
      -4.919999999999998,
      -5.0748,
      -5.203199999999999,
      -5.305199999999999,
      -5.380800000000001,
      -5.43,
      -5.4528,
      -5.449200000000002,
      -5.4192,
      -5.362799999999999,
      -5.2799999999999985,
      -5.170799999999999,
      -5.035200000000001,
      -4.873200000000001,
      -4.684800000000001,
      -4.470000000000001,
      -4.2288,
      -3.9612,
      -3.6672,
      -3.346800000000001,
      -3.,
      -2.648399999999999,
      -2.3136,
      -1.9955999999999996,
      -1.6943999999999986,
      -1.4099999999999988,
      -1.1423999999999985,
      -0.8915999999999988,
      -0.6576000000000011,
      -0.44040000000000057,
      -0.24000000000000155,
      -0.05640000000000023,
      0.11039999999999961,
      0.26040000000000063,
      0.39360000000000017,
      0.5100000000000007,
      0.6096000000000004,
      0.6924000000000003,
      0.7584,
      0.8076000000000003,
      0.8399999999999999,
      0.8555999999999996,
      0.8543999999999998,
      0.8364000000000006,
      0.8016000000000004,
      0.75,
      0.6816000000000003,
      0.5964000000000004,
      0.4943999999999994,
      0.37559999999999916,
      0.23999999999999955,
      0.08759999999999901,
      -0.08160000000000145,
      -0.2676000000000016,
      -0.4703999999999996,
      -0.6899999999999996,
      -0.9263999999999999,
      -1.1795999999999998,
      -1.4496000000000002,
      -1.7364000000000006,
      -2.040000000000001,
      -2.360400000000001,
      -2.6976000000000013,
      -3.0516000000000023,
      -3.422400000000002,
      -3.8100000000000027,
      -4.214399999999998,
      -4.635599999999999,
      -5.073599999999999,
      -5.5283999999999995,
      -6.};
  static constexpr double kGoldenDataYpp[GetNumSamples()] = {
      -96.,
      -93.36000000000001,
      -90.72000000000001,
      -88.08000000000001,
      -85.44,
      -82.80000000000001,
      -80.16,
      -77.52000000000001,
      -74.88000000000001,
      -72.24000000000002,
      -69.60000000000001,
      -66.96000000000001,
      -64.32000000000001,
      -61.67999999999999,
      -59.04,
      -56.39999999999999,
      -53.760000000000005,
      -51.12,
      -48.480000000000004,
      -45.839999999999996,
      -43.199999999999996,
      -40.56000000000001,
      -37.92,
      -35.28,
      -32.63999999999999,
      -30.,
      -27.360000000000003,
      -24.71999999999999,
      -22.07999999999999,
      -19.44000000000001,
      -16.800000000000004,
      -14.160000000000004,
      -11.520000000000003,
      -8.879999999999995,
      -6.239999999999997,
      -3.5999999999999925,
      -0.9600000000000044,
      1.6799999999999997,
      4.32,
      6.960000000000003,
      9.600000000000007,
      12.240000000000007,
      14.879999999999995,
      17.52,
      20.160000000000004,
      22.800000000000004,
      25.440000000000005,
      28.08000000000001,
      30.71999999999999,
      33.36,
      36.,
      34.32,
      32.64,
      30.959999999999994,
      29.279999999999994,
      27.599999999999994,
      25.919999999999987,
      24.239999999999988,
      22.56000000000001,
      20.880000000000003,
      19.200000000000003,
      17.52,
      15.840000000000002,
      14.159999999999998,
      12.48,
      10.799999999999997,
      9.119999999999996,
      7.439999999999994,
      5.759999999999991,
      4.079999999999991,
      2.399999999999988,
      0.720000000000006,
      -0.9599999999999937,
      -2.639999999999997,
      -4.32,
      -6.,
      -7.68,
      -9.360000000000001,
      -11.040000000000006,
      -12.720000000000006,
      -14.400000000000006,
      -16.080000000000013,
      -17.760000000000012,
      -19.440000000000012,
      -21.119999999999997,
      -22.799999999999997,
      -24.48,
      -26.159999999999997,
      -27.840000000000003,
      -29.52,
      -31.200000000000003,
      -32.88,
      -34.56000000000001,
      -36.24000000000001,
      -37.920000000000016,
      -39.60000000000001,
      -41.28,
      -42.96,
      -44.64,
      -46.31999999999999,
      -48.};
  static constexpr double kGoldenDataYppp[GetNumSamples()] = {
      264.,  264.,  264.,  264.,  264.,  264.,  264.,  264.,  264.,  264.,
      264.,  264.,  264.,  264.,  264.,  264.,  264.,  264.,  264.,  264.,
      264.,  264.,  264.,  264.,  264.,  264.,  264.,  264.,  264.,  264.,
      264.,  264.,  264.,  264.,  264.,  264.,  264.,  264.,  264.,  264.,
      264.,  264.,  264.,  264.,  264.,  264.,  264.,  264.,  264.,  264.,
      -168., -168., -168., -168., -168., -168., -168., -168., -168., -168.,
      -168., -168., -168., -168., -168., -168., -168., -168., -168., -168.,
      -168., -168., -168., -168., -168., -168., -168., -168., -168., -168.,
      -168., -168., -168., -168., -168., -168., -168., -168., -168., -168.,
      -168., -168., -168., -168., -168., -168., -168., -168., -168., -168.,
      -168.};
};

TYPED_TEST_SUITE_P(BSplineGoldenTest);
TYPED_TEST_P(BSplineGoldenTest, CompareToReference) {
  using SplineType = TypeParam;

  ArrayXd sampled_x(this->GetNumSamples());
  ArrayXd sampled_y(this->GetNumSamples());

  SplineType spline;
  constexpr int kDegree = 3;
  constexpr int kPointDim = 2;
  ArrayXd knots = MakeArrayXd({0, 0, 0, 0, 0.5, 1, 1, 1, 1});
  ASSERT_EQ(spline.Init(kDegree, knots.size(), kPointDim).code(),
            absl::StatusCode::kOk);
  ASSERT_EQ(spline.SetKnotVector(knots).code(), absl::StatusCode::kOk);
  typename SplineType::Point point(kPointDim);
  std::vector<typename SplineType::Point> polygon(spline.NumPoints());

  for (auto &p : polygon) {
    p.resize(kPointDim);
  }
  polygon[0] << 1, 1;
  polygon[1] << 2, 3;
  polygon[2] << 3, -1;
  polygon[3] << 4, 1;
  polygon[4] << 5.0, 0.0;
  ASSERT_EQ(spline.SetControlPoints(polygon).code(), absl::StatusCode::kOk);
  ASSERT_EQ(this->golden_data_x_.size(), this->GetNumSamples());
  ASSERT_EQ(this->golden_data_y_.size(), this->GetNumSamples());
  if (absl::GetFlag(FLAGS_verbosity) >= 2) {
    for (int i = 0; i < polygon.size(); i++) {
      printf("debug-golden-polygon i= %d x= %f y= %f\n", i, polygon[i][0],
             polygon[i][1]);
    }
  }

  const double du =
      (knots[knots.size() - 1] - knots[0]) / (this->GetNumSamples() - 1);
  for (int idx = 0; idx < this->GetNumSamples(); idx++) {
    const double u = knots[0] + idx * du;
    ASSERT_EQ(spline.EvalCurve(u, point).code(), absl::StatusCode::kOk);
    sampled_x[idx] = point[0];
    sampled_y[idx] = point[1];
    if (absl::GetFlag(FLAGS_verbosity) >= 2) {
      printf("debug-golden-spline u= %.18e x= %.18e y= %.18e\n", u, point[0],
             point[1]);
    }
  }
  double yerr = (sampled_x - this->golden_data_x_).matrix().norm();
  double xerr = (sampled_y - this->golden_data_y_).matrix().norm();
  if (absl::GetFlag(FLAGS_verbosity) >= 1) {
    printf("GoldenCurve: xerr= %.18e, yerr= %.18e\n", xerr, yerr);
  }
  EXPECT_NEAR(0.0, xerr, this->GetGoldenEpsilon());
  EXPECT_NEAR(0.0, yerr, this->GetGoldenEpsilon());

  std::vector<typename SplineType::Point> pk(kDegree + 1);
  for (auto &p : pk) {
    p.resize(kPointDim);
  }
  ArrayXd sampled_xp(this->GetNumSamples());
  ArrayXd sampled_yp(this->GetNumSamples());
  ArrayXd sampled_xpp(this->GetNumSamples());
  ArrayXd sampled_ypp(this->GetNumSamples());
  ArrayXd sampled_xppp(this->GetNumSamples());
  ArrayXd sampled_yppp(this->GetNumSamples());
  for (int idx = 0; idx < this->GetNumSamples(); idx++) {
    const double u = knots[0] + idx * du;
    ASSERT_EQ(
        spline.EvalCurveAndDerivatives(u, absl::MakeSpan(pk.data(), pk.size()))
            .code(),
        absl::StatusCode::kOk);
    sampled_x[idx] = pk[0][0];
    sampled_xp[idx] = pk[1][0];
    sampled_xpp[idx] = pk[2][0];
    sampled_xppp[idx] = pk[3][0];
    sampled_y[idx] = pk[0][1];
    sampled_yp[idx] = pk[1][1];
    sampled_ypp[idx] = pk[2][1];
    sampled_yppp[idx] = pk[3][1];
    if (absl::GetFlag(FLAGS_verbosity) >= 2) {
      printf(
          "debug-golden-spline u= %.18e x= %.18e y= %.18e xp= %.18e yp= "
          "%.18e xp= "
          "%.18e yp= %.18e\n",
          u, pk[0][0], pk[0][1], pk[1][0], pk[1][1], pk[2][0], pk[2][1]);
    }
  }
  xerr = (sampled_x - this->golden_data_x_).matrix().squaredNorm();
  yerr = (sampled_y - this->golden_data_y_).matrix().squaredNorm();
  double xperr = (sampled_xp - this->golden_data_xp_).matrix().squaredNorm();
  double yperr = (sampled_yp - this->golden_data_yp_).matrix().squaredNorm();
  double xpperr = (sampled_xpp - this->golden_data_xpp_).matrix().squaredNorm();
  double ypperr = (sampled_ypp - this->golden_data_ypp_).matrix().squaredNorm();
  double xppperr =
      (sampled_xppp - this->golden_data_xppp_).matrix().squaredNorm();
  double yppperr =
      (sampled_yppp - this->golden_data_yppp_).matrix().squaredNorm();
  if (absl::GetFlag(FLAGS_verbosity) >= 1) {
    printf("GoldenCurve: xerr= %.18e, yerr= %.18e\n", xerr, yerr);
    printf("GoldenCurve: xperr= %.18e, yperr= %.18e\n", xperr, yperr);
    printf("GoldenCurve: xpperr= %.18e, ypperr= %.18e\n", xpperr, ypperr);
    printf("GoldenCurve: xppperr= %.18e, yppperr= %.18e\n", xppperr, yppperr);
  }
  EXPECT_NEAR(0.0, xerr, this->GetGoldenEpsilon());
  EXPECT_NEAR(0.0, yerr, this->GetGoldenEpsilon());
  EXPECT_NEAR(0.0, xperr, this->GetGoldenEpsilon());
  EXPECT_NEAR(0.0, yperr, this->GetGoldenEpsilon());
  EXPECT_NEAR(0.0, xpperr, this->GetGoldenEpsilon());
  EXPECT_NEAR(0.0, ypperr, this->GetGoldenEpsilon());
  EXPECT_NEAR(0.0, xppperr, this->GetGoldenEpsilon());
  EXPECT_NEAR(0.0, yppperr, this->GetGoldenEpsilon());
}

REGISTER_TYPED_TEST_SUITE_P(BSplineGoldenTest, CompareToReference);
typedef ::testing::Types<BSplineNd, BSpline2d> GoldenSplineTypes;
INSTANTIATE_TYPED_TEST_SUITE_P(TheGoldenSplineTest, BSplineGoldenTest,
                               GoldenSplineTypes);

template <typename Traits>
class SplineStatus : public testing::Test {};

TYPED_TEST_SUITE_P(SplineStatus);

TYPED_TEST_P(SplineStatus, Init) {
  using SplineType = typename TypeParam::SplineType;
  constexpr int kDimension = TypeParam::kDimension;
  {
    SplineType spline;
    ASSERT_EQ(spline.Init(3, 10, kDimension).code(), absl::StatusCode::kOk);
  }
  {
    SplineType spline;
    ASSERT_EQ(spline.Init(3, 3, kDimension).code(),
              absl::StatusCode::kInvalidArgument);
  }
  {
    SplineType spline;
    ASSERT_EQ(spline.Init(-3, 10, kDimension).code(),
              absl::StatusCode::kOutOfRange);
  }
}

TYPED_TEST_P(SplineStatus, ControlPoints) {
  using SplineType = typename TypeParam::SplineType;
  using Point = typename TypeParam::SplineType::Point;
  constexpr int kDimension = TypeParam::kDimension;

  SplineType spline;
  constexpr int kDegree = 2;
  std::vector<double> knots = {0, 0, 0, 1, 2, 3, 4, 5, 5, 5};
  ASSERT_EQ(spline.Init(kDegree, knots.size(), kDimension).code(),
            absl::StatusCode::kOk);
  std::vector<Point> polygon(spline.MaxNumPoints());
  std::vector<Point> polygon_small(spline.MaxNumPoints() - 1);
  std::vector<Point> polygon_large(spline.MaxNumPoints() + 1);
  Point point(kDimension);
  std::vector<Point> points(kDegree);

  for (int i = 0; i < polygon.size(); ++i) {
    if constexpr (std::is_same_v<double, Point>) {
      polygon[i] = i;
    } else {
      polygon[i].setConstant(i);
    }
  }

  ASSERT_EQ(spline.SetKnotVector(knots).code(), absl::StatusCode::kOk);
  ASSERT_EQ(spline.SetControlPoints(polygon_small).code(),
            absl::StatusCode::kInvalidArgument);
  ASSERT_EQ(spline.SetControlPoints(polygon_large).code(),
            absl::StatusCode::kInvalidArgument);
  ASSERT_EQ(spline.SetControlPoints(polygon).code(), absl::StatusCode::kOk);
  const auto control_points = spline.GetControlPoints();
  for (double u = knots[0]; u <= knots[knots.size() - 1]; u += 0.01) {
    ASSERT_EQ(spline.EvalCurve(u, point).code(), absl::StatusCode::kOk);
    ASSERT_EQ(spline
                  .EvalCurveAndDerivatives(
                      u, absl::MakeSpan(points.data(), points.size()))
                  .code(),
              absl::StatusCode::kOk);
  }

  EXPECT_THAT(control_points, ElementsAreArray(polygon));
}

TYPED_TEST_P(SplineStatus, KnotVector) {
  using SplineType = typename TypeParam::SplineType;
  using Point = typename TypeParam::SplineType::Point;
  constexpr int kDimension = TypeParam::kDimension;

  SplineType spline;
  constexpr int kDegree = 2;
  std::vector<double> knots = {0, 0, 0, 1, 2, 3, 4, 5, 5, 5};
  std::vector<double> knots_small = {0, 0, 0, 1, 2, 3, 3, 3};
  ASSERT_EQ(spline.Init(kDegree, knots.size(), kDimension).code(),
            absl::StatusCode::kOk);
  std::vector<Point> polygon(spline.MaxNumPoints());
  std::vector<Point> polygon_small(spline.MaxNumPoints() - 2);
  Point point(kDimension);
  std::vector<Point> points(kDegree);

  ASSERT_EQ(spline.SetKnotVector(knots).code(), absl::StatusCode::kOk);
  ASSERT_EQ(spline.SetControlPoints(polygon).code(), absl::StatusCode::kOk);
  for (double u = knots[0]; u <= knots[knots.size() - 1]; u += 0.01) {
    ASSERT_EQ(spline.EvalCurve(u, point).code(), absl::StatusCode::kOk);
    ASSERT_EQ(spline
                  .EvalCurveAndDerivatives(
                      u, absl::MakeSpan(points.data(), points.size()))
                  .code(),
              absl::StatusCode::kOk);
  }
  ASSERT_EQ(spline.SetKnotVector(knots_small).code(), absl::StatusCode::kOk);
  ASSERT_EQ(spline.SetControlPoints(polygon_small).code(),
            absl::StatusCode::kOk);
  for (double u = knots_small[0]; u <= knots_small[knots_small.size() - 1];
       u += 0.01) {
    ASSERT_EQ(spline.EvalCurve(u, point).code(), absl::StatusCode::kOk);
    ASSERT_EQ(spline
                  .EvalCurveAndDerivatives(
                      u, absl::MakeSpan(points.data(), points.size()))
                  .code(),
              absl::StatusCode::kOk);
  }
  ASSERT_EQ(spline.SetControlPoints(polygon).code(),
            absl::StatusCode::kInvalidArgument);
}

struct BSpline1Traits {
  using SplineType = BSpline1d;
  static constexpr int kDimension = 1;
};
struct BSpline2Traits {
  using SplineType = BSpline2d;
  static constexpr int kDimension = 2;
};
struct BSpline3Traits {
  using SplineType = BSpline3d;
  static constexpr int kDimension = 3;
};

REGISTER_TYPED_TEST_SUITE_P(SplineStatus, Init, ControlPoints, KnotVector);
typedef ::testing::Types<BSpline1Traits, BSpline2Traits, BSpline3Traits>
    InitiAndAllocationSplineTypes;

INSTANTIATE_TYPED_TEST_SUITE_P(TheSplineStatus, SplineStatus,
                               InitiAndAllocationSplineTypes);

// Check if:
// - functions catch errors
// - if code intended for real-time use allocates
TEST(BSplineNdTest, InputAndAllocation) {
  {  // valid case
    BSplineNd spline;
    ASSERT_EQ(spline.Init(3, 10, 1).code(), absl::StatusCode::kOk);
  }
  {  // too few knots
    BSplineNd spline;
    ASSERT_EQ(spline.Init(3, 3, 1).code(), absl::StatusCode::kInvalidArgument);
  }
  {  // negative degree
    BSplineNd spline;
    ASSERT_EQ(spline.Init(-3, 10, 1).code(), absl::StatusCode::kOutOfRange);
  }
  {  // set points function
    BSplineNd spline;
    constexpr int kDegree = 2;
    constexpr int kDim = 1;
    std::vector<double> knots = {0, 0, 0, 1, 2, 3, 4, 5, 5, 5};
    ASSERT_EQ(spline.Init(kDegree, knots.size(), kDim).code(),
              absl::StatusCode::kOk);
    std::vector<BSplineNd::Point> polygon(spline.MaxNumPoints());
    std::vector<BSplineNd::Point> polygon_small(spline.MaxNumPoints() - 1);
    std::vector<BSplineNd::Point> polygon_large(spline.MaxNumPoints() + 1);
    std::vector<BSplineNd::Point> polygon_size_error(spline.MaxNumPoints());
    BSplineNd::Point point(kDim);
    std::vector<BSplineNd::Point> points(kDegree);
    for (auto &p : points) {
      p.resize(kDim);
    }

    for (auto &p : polygon) {
      p.resize(kDim);
    }
    for (auto &p : polygon_small) {
      p.resize(kDim);
    }
    for (auto &p : polygon_large) {
      p.resize(kDim);
    }

    ASSERT_EQ(spline.SetKnotVector(knots).code(), absl::StatusCode::kOk);
    ASSERT_EQ(spline.SetControlPoints(polygon_small).code(),
              absl::StatusCode::kInvalidArgument);
    ASSERT_EQ(spline.SetControlPoints(polygon_large).code(),
              absl::StatusCode::kInvalidArgument);
    ASSERT_EQ(spline.SetControlPoints(polygon_size_error).code(),
              absl::StatusCode::kInvalidArgument);
    ASSERT_EQ(spline.SetControlPoints(polygon).code(), absl::StatusCode::kOk);
    for (double u = knots[0]; u <= knots[knots.size() - 1]; u += 0.01) {
      ASSERT_EQ(spline.EvalCurve(u, point).code(), absl::StatusCode::kOk);
      ASSERT_EQ(spline
                    .EvalCurveAndDerivatives(
                        u, absl::MakeSpan(points.data(), kDegree))
                    .code(),
                absl::StatusCode::kOk);
    }
  }

  // try changing number of points
  {
    BSplineNd spline;
    constexpr int kDegree = 2;
    constexpr int kDim = 1;
    std::vector<double> knots = {0, 0, 0, 1, 2, 3, 4, 5, 5, 5};
    std::vector<double> knots_small = {0, 0, 0, 1, 2, 3, 3, 3};
    ASSERT_EQ(spline.Init(kDegree, knots.size(), kDim).code(),
              absl::StatusCode::kOk);
    std::vector<BSplineNd::Point> polygon(spline.MaxNumPoints());
    std::vector<BSplineNd::Point> polygon_small(spline.MaxNumPoints() - 2);
    BSplineNd::Point point(kDim);
    std::vector<BSplineNd::Point> points(kDegree);
    for (auto &p : points) {
      p.resize(kDim);
    }

    for (auto &p : polygon) {
      p.resize(kDim);
    }
    for (auto &p : polygon_small) {
      p.resize(kDim);
    }

    // first, maximum polygon size
    ASSERT_EQ(spline.SetKnotVector(knots).code(), absl::StatusCode::kOk);
    ASSERT_EQ(spline.SetControlPoints(polygon).code(), absl::StatusCode::kOk);
    for (double u = knots[0]; u <= knots[knots.size() - 1]; u += 0.01) {
      ASSERT_EQ(spline.EvalCurve(u, point).code(), absl::StatusCode::kOk);
      ASSERT_EQ(spline
                    .EvalCurveAndDerivatives(
                        u, absl::MakeSpan(points.data(), kDegree))
                    .code(),
                absl::StatusCode::kOk);
    }
    // try using shorter spline
    ASSERT_EQ(spline.SetKnotVector(knots_small).code(), absl::StatusCode::kOk);
    ASSERT_EQ(spline.SetControlPoints(polygon_small).code(),
              absl::StatusCode::kOk);
    for (double u = knots_small[0]; u <= knots_small[knots_small.size() - 1];
         u += 0.01) {
      ASSERT_EQ(spline.EvalCurve(u, point).code(), absl::StatusCode::kOk);
      ASSERT_EQ(spline
                    .EvalCurveAndDerivatives(
                        u, absl::MakeSpan(points.data(), kDegree))
                    .code(),
                absl::StatusCode::kOk);
    }

    // now, larger polygon but without resetting knot vector
    ASSERT_EQ(spline.SetControlPoints(polygon).code(),
              absl::StatusCode::kInvalidArgument);
  }
}
TEST(BSplineNdTest, BasisFunctions) {
  BSplineNd spline;
  constexpr int kDegree = 2;
  constexpr int kPointDim = 1;
  std::vector<double> knots = {0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5};
  ASSERT_EQ(spline.Init(kDegree, knots.size(), kPointDim).code(),
            absl::StatusCode::kOk);
  ASSERT_EQ(spline.SetKnotVector(knots).code(), absl::StatusCode::kOk);

  BSplineNd::Point point(kPointDim);
  BSplineNd::Point one(kPointDim);
  BSplineNd::Point zero(kPointDim);
  one.setConstant(1.0);
  zero.setZero();
  std::vector<BSplineNd::Point> polygon(spline.NumPoints());
  for (auto &p : polygon) {
    p.resize(kPointDim);
    p = zero;
  }
  ArrayXd basis(spline.NumPoints());
  for (double u = knots[0]; u <= knots[knots.size() - 1]; u += 0.01) {
    basis = 0.0;
    for (size_t k = 0; k < polygon.size(); k++) {
      polygon[k] = one;
      ASSERT_EQ(spline.SetControlPoints(polygon).code(), absl::StatusCode::kOk);
      ASSERT_EQ(spline.EvalCurve(u, point).code(), absl::StatusCode::kOk);
      basis[k] = point[0];
      // basis functions should be >=0
      ASSERT_GE(basis[k], 0.0);
      if (absl::GetFlag(FLAGS_verbosity) >= 2) {
        printf("debug-basis-function-%zd,%d(u= %f ) %f\n", k, kDegree, u,
               point[0]);
      }
      polygon[k] = zero;
    }
    // sum of basis functions must be 1
    ASSERT_DOUBLE_EQ(1.0, basis.sum());
  }
}

TEST(BSplineTest, ParameterRange) {
  // umin,umax matching knot vector
  {
    const std::vector<BSpline1d::Point> points = {1, 2, 3, 4, 5, 6, 7, 8};
    static constexpr int kDegree = 3;
    BSpline1d spline;
    ASSERT_EQ(
        spline.Init(kDegree, BSplineBase::NumKnots(points.size(), kDegree))
            .code(),
        absl::StatusCode::kOk);
    ASSERT_GE(points.size(), BSplineBase::MinNumPoints(kDegree));
    ASSERT_EQ(
        spline
            .SetUniformKnotVector(BSplineBase::NumKnots(points.size(), kDegree))
            .code(),
        absl::StatusCode::kOk);
    ASSERT_EQ(spline.SetControlPoints(points).code(), absl::StatusCode::kOk);

    std::vector<BSpline1d::Point> values(2);
    BSpline1d::Point point;

    ASSERT_EQ(spline
                  .EvalCurveAndDerivatives(
                      0.5, absl::MakeSpan(values.data(), values.size()))
                  .code(),
              absl::StatusCode::kOk);
    ASSERT_EQ(spline.EvalCurve(0.5, point).code(), absl::StatusCode::kOk);

    ASSERT_EQ(spline
                  .EvalCurveAndDerivatives(
                      0.0, absl::MakeSpan(values.data(), values.size()))
                  .code(),
              absl::StatusCode::kOk);
    ASSERT_EQ(spline.EvalCurve(0.0, point).code(), absl::StatusCode::kOk);

    EXPECT_EQ(points.front(), point);
    EXPECT_EQ(points.front(), values[0]);

    ASSERT_EQ(spline
                  .EvalCurveAndDerivatives(
                      1.0, absl::MakeSpan(values.data(), values.size()))
                  .code(),
              absl::StatusCode::kOk);
    ASSERT_EQ(spline.EvalCurve(1.0, point).code(), absl::StatusCode::kOk);

    EXPECT_EQ(points.back(), point);
    EXPECT_EQ(points.back(), values[0]);

    ASSERT_EQ(spline
                  .EvalCurveAndDerivatives(
                      1.1, absl::MakeSpan(values.data(), values.size()))
                  .code(),
              absl::StatusCode::kOutOfRange);
    ASSERT_EQ(spline.EvalCurve(1.1, point).code(),
              absl::StatusCode::kOutOfRange);
    ASSERT_EQ(spline
                  .EvalCurveAndDerivatives(
                      -0.1, absl::MakeSpan(values.data(), values.size()))
                  .code(),
              absl::StatusCode::kOutOfRange);
    ASSERT_EQ(spline.EvalCurve(-0.1, point).code(),
              absl::StatusCode::kOutOfRange);
  }

  // umin,umax restricting parameter range
  {
    const std::vector<BSpline1d::Point> points = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<double> knots = {0,   0,   0, 0, 0.2, 0.4,
                                       0.6, 0.8, 1, 1, 1,   1};
    constexpr double kUmin = 0.1;
    constexpr double kUmax = 0.9;
    static constexpr int kDegree = 3;
    BSpline1d spline;
    ASSERT_EQ(
        spline.Init(kDegree, BSplineBase::NumKnots(points.size(), kDegree))
            .code(),
        absl::StatusCode::kOk);
    ASSERT_GE(points.size(), BSplineBase::MinNumPoints(kDegree));
    ASSERT_EQ(spline.SetKnotVector(knots, kUmin, kUmax).code(),
              absl::StatusCode::kOk);
    ASSERT_EQ(spline.SetControlPoints(points).code(), absl::StatusCode::kOk);

    std::vector<BSpline1d::Point> values(2);
    BSpline1d::Point point;

    ASSERT_EQ(spline
                  .EvalCurveAndDerivatives(
                      0.5, absl::MakeSpan(values.data(), values.size()))
                  .code(),
              absl::StatusCode::kOk);
    ASSERT_EQ(spline.EvalCurve(0.5, point).code(), absl::StatusCode::kOk);

    ASSERT_EQ(spline
                  .EvalCurveAndDerivatives(
                      0.0, absl::MakeSpan(values.data(), values.size()))
                  .code(),
              absl::StatusCode::kOutOfRange);
    ASSERT_EQ(spline.EvalCurve(0.0, point).code(),
              absl::StatusCode::kOutOfRange);

    ASSERT_EQ(spline
                  .EvalCurveAndDerivatives(
                      kUmin, absl::MakeSpan(values.data(), values.size()))
                  .code(),
              absl::StatusCode::kOk);
    ASSERT_EQ(spline.EvalCurve(kUmax, point).code(), absl::StatusCode::kOk);

    ASSERT_EQ(spline
                  .EvalCurveAndDerivatives(
                      1.0, absl::MakeSpan(values.data(), values.size()))
                  .code(),
              absl::StatusCode::kOutOfRange);
    ASSERT_EQ(spline.EvalCurve(1.0, point).code(),
              absl::StatusCode::kOutOfRange);

    ASSERT_EQ(spline
                  .EvalCurveAndDerivatives(
                      kUmax, absl::MakeSpan(values.data(), values.size()))
                  .code(),
              absl::StatusCode::kOk);
    ASSERT_EQ(spline.EvalCurve(kUmax, point).code(), absl::StatusCode::kOk);

    ASSERT_EQ(spline
                  .EvalCurveAndDerivatives(
                      1.1, absl::MakeSpan(values.data(), values.size()))
                  .code(),
              absl::StatusCode::kOutOfRange);
    ASSERT_EQ(spline.EvalCurve(1.1, point).code(),
              absl::StatusCode::kOutOfRange);
    ASSERT_EQ(spline
                  .EvalCurveAndDerivatives(
                      -0.1, absl::MakeSpan(values.data(), values.size()))
                  .code(),
              absl::StatusCode::kOutOfRange);
    ASSERT_EQ(spline.EvalCurve(-0.1, point).code(),
              absl::StatusCode::kOutOfRange);
  }
}

TEST(BSplineTest, AutoKnotVector) {
  {
    BSpline1d spline;
    constexpr int kNotsCount = 8;
    ASSERT_EQ(spline.Init(3, kNotsCount).code(), absl::StatusCode::kOk);
    ASSERT_EQ(spline.SetUniformKnotVector(kNotsCount).code(),
              absl::StatusCode::kOk);
    EXPECT_THAT(spline.GetKnotVector(),
                ElementsAre(0., 0., 0., 0., 1., 1., 1., 1.));
  }
  {
    BSpline1d spline;
    constexpr int kNotsCount = 9;
    ArrayXd knots = MakeArrayXd({});
    ASSERT_EQ(spline.Init(3, kNotsCount).code(), absl::StatusCode::kOk);
    ASSERT_EQ(spline.SetUniformKnotVector(kNotsCount).code(),
              absl::StatusCode::kOk);
    EXPECT_THAT(spline.GetKnotVector(),
                ElementsAre(0., 0., 0., 0., 0.5, 1., 1., 1., 1.));
  }
  {
    BSpline1d spline;
    constexpr int kNotsCount = 7;
    ASSERT_EQ(spline.Init(1, kNotsCount).code(), absl::StatusCode::kOk);
    ASSERT_EQ(spline.SetUniformKnotVector(kNotsCount).code(),
              absl::StatusCode::kOk);
    EXPECT_THAT(spline.GetKnotVector(),
                ElementsAre(0., 0., 0.25, 0.5, 0.75, 1., 1.));
  }
}

TEST(BSplineTest, CurveEvaluationWorksWithoutSettingControlPoints) {
  constexpr size_t kDegree = 3;
  constexpr std::array<double, 12> kKnots = {
      {0, 0, 0, 0, 1, 2, 3, 4, 5, 5, 5, 5}};
  BSpline2d spline;
  ASSERT_OK(spline.Init(kDegree, kKnots.size()));

  ASSERT_EQ(spline.SetKnotVector(kKnots).code(), absl::StatusCode::kOk);

  for (double u = kKnots[0]; u <= kKnots[kKnots.size() - 1]; u += 0.01) {
    Vector2d vec(1, 2);
    ASSERT_OK(spline.EvalCurve(u, vec));
    EXPECT_THAT(vec, IsApprox(Vector2d::Zero()));
  }
}

TEST(BSplineTest, InsertKnotFailsForInvalidInput) {
  BSpline2d spline;
  constexpr int kDegree = 3;
  constexpr int kMinPoints = BSplineBase::MinNumPoints(kDegree);
  constexpr int kMinCapacity = BSplineBase::NumKnots(kMinPoints, kDegree);
  ASSERT_OK(spline.Init(kDegree, kMinCapacity));

  EXPECT_EQ(spline.InsertKnotAndUpdateControlPoints(0.5, 10).code(),
            absl::StatusCode::kInvalidArgument);

  EXPECT_EQ(spline.InsertKnotAndUpdateControlPoints(0.5, 1).code(),
            absl::StatusCode::kFailedPrecondition);

  ASSERT_OK(spline.SetUniformKnotVector(kMinCapacity));

  EXPECT_EQ(spline.InsertKnotAndUpdateControlPoints(0.5, 1).code(),
            absl::StatusCode::kFailedPrecondition);

  // Resize and set a new knot vector.
  ASSERT_OK(spline.Init(kDegree, kMinCapacity + 1));
  ASSERT_OK(spline.SetUniformKnotVector(kMinCapacity));
  EXPECT_EQ(spline.InsertKnotAndUpdateControlPoints(-0.5, 1).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(spline.InsertKnotAndUpdateControlPoints(1.5, 1).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(spline.InsertKnotAndUpdateControlPoints(0.5, 10).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(BSplineTest, InsertKnotWorksForValidInput) {
  // Loops over a range of degrees, knot multiplicities and knot values and
  // verifies that the spline curve position values are unchanged after knot
  // insertion.
  // Incidentally, the case for degree=2, k=1 also covers the special case where
  // the knot to be inserted is equal to an already existing knot.
  BSpline2d spline_ref;
  const std::vector<Vector2d> points = {Vector2d(-1.5, -0.5),
                                        Vector2d(-1.1, -0.1),
                                        Vector2d(-0.5, -std::sqrt(3.0) * 0.5),
                                        Vector2d(0.5, -std::sqrt(3.0) * 0.5),
                                        Vector2d(1, 0),
                                        Vector2d(0.5, 0.5 * std::sqrt(3.0)),
                                        Vector2d(-0.5, 0.5 * std::sqrt(3.0)),
                                        Vector2d(-1.0, -0.0)};
  for (int degree = 1; degree <= 3; ++degree) {
    SCOPED_TRACE(absl::StrCat("degree= ", degree));
    for (int knot_multiplicity = 1; knot_multiplicity <= degree;
         ++knot_multiplicity) {
      SCOPED_TRACE(absl::StrCat("knot_multiplicity= ", knot_multiplicity));
      const int capacity =
          BSplineBase::NumKnots(points.size(), degree) + knot_multiplicity;
      ASSERT_OK(spline_ref.Init(degree, capacity));
      ASSERT_OK(spline_ref.SetUniformKnotVector(
          BSplineBase::NumKnots(points.size(), degree)));
      ASSERT_OK(spline_ref.SetControlPoints(points));

      // One knot in the first knot span, one somewhere in the middle, one in
      // the last knot span.
      const int knot_count = spline_ref.GetKnotVector().size();
      const std::vector<double> knots_to_insert = {
          0.5 * spline_ref.GetKnotVector()[0] +
              0.5 * spline_ref.GetKnotVector()[degree + 1],
          0.5 * spline_ref.GetKnotVector().front() +
              0.5 * spline_ref.GetKnotVector().back(),
          0.5 * spline_ref.GetKnotVector()[knot_count - degree - 2] +
              0.5 * spline_ref.GetKnotVector().back()};
      for (int k = 0; k < knots_to_insert.size(); ++k) {
        const double the_knot = knots_to_insert[k];
        SCOPED_TRACE(absl::StrCat("the_knot= ", the_knot));
        BSpline2d spline_with_extra_knots = spline_ref;
        BSpline2d spline_with_extra_knots_ref = spline_ref;
        const std::string log_prefix =
            absl::StrCat("D", degree, "M", knot_multiplicity, "K", k);
        // Log old and new control points, as well as spline curve samples for
        // debugging and plotting.
        for (const auto &point : spline_ref.GetControlPoints()) {
          LOG(INFO) << log_prefix
                    << "-OLD_CONTROL_POINTS: " << point.transpose() << "\n";
        }
        // Insert knot(s) by solving system of equations.
        ASSERT_OK(spline_with_extra_knots.InsertKnotAndUpdateControlPoints(
            the_knot, knot_multiplicity));

        // Insert knot(s) using reference implementation that exploits
        // structure.
        ASSERT_OK(
            spline_with_extra_knots_ref.InsertKnotAndUpdateControlPointsRef(
                the_knot, knot_multiplicity));

        for (const auto &point : spline_with_extra_knots.GetControlPoints()) {
          LOG(INFO) << log_prefix
                    << "-NEW_CONTROL_POINTS: " << point.transpose() << "\n";
        }
        for (const auto &point :
             spline_with_extra_knots_ref.GetControlPoints()) {
          LOG(INFO) << log_prefix
                    << "-NEW_CONTROL_POINTS_REF: " << point.transpose() << "\n";
        }

        for (double u = spline_ref.GetKnotVector().front();
             u <= spline_ref.GetKnotVector().back(); u += 0.001) {
          Vector2d curve_reference;
          ASSERT_OK(spline_ref.EvalCurve(u, curve_reference));
          Vector2d curve_value;
          ASSERT_OK(spline_with_extra_knots.EvalCurve(u, curve_value));
          Vector2d curve_ref_value;
          ASSERT_OK(spline_ref.EvalCurve(u, curve_ref_value));
          LOG(INFO) << log_prefix << "-CURVE_VALUE: " << u << " "
                    << curve_value.transpose() << "\n";
          LOG(INFO) << log_prefix << "-REFERENCE_VALUE: " << u << " "
                    << curve_reference.transpose() << "\n";
          EXPECT_THAT(curve_value, IsApprox(curve_reference))
              << "InsertKnotAndUpdateControlPoints modified curve values";
          EXPECT_THAT(curve_ref_value, IsApprox(curve_reference))
              << "InsertKnotAndUpdateControlPointsRef modified curve values";
        }
      }
    }
  }
}

TEST(BSplineTest, TruncateSplineAtIsNoopForLargeParameter) {
  BSpline2d spline_ref;
  constexpr int kDegree = 3;
  constexpr int kCapacity = 20;  // Large enoug.
  const std::vector<Vector2d> points = {Vector2d(0, 0), Vector2d(0, 1),
                                        Vector2d(1, 1), Vector2d(1, 0),
                                        Vector2d(2, 0), Vector2d(2, 1)};

  ASSERT_OK(spline_ref.Init(kDegree, kCapacity));
  ASSERT_OK(spline_ref.SetUniformKnotVector(
      BSplineBase::NumKnots(points.size(), kDegree)));
  ASSERT_OK(spline_ref.SetControlPoints(points));

  BSpline2d spline = spline_ref;

  ASSERT_OK(spline.TruncateSplineAt(spline.GetKnotVector().back() + 1));
  EXPECT_THAT(spline.GetKnotVector(),
              ElementsAreArray(spline_ref.GetKnotVector()));
  EXPECT_THAT(spline.GetControlPoints(), ElementsAreArray(points));

  ASSERT_OK(spline.TruncateSplineAt(spline.GetKnotVector().back()));
  EXPECT_THAT(spline.GetKnotVector(),
              ElementsAreArray(spline_ref.GetKnotVector()));
  EXPECT_THAT(spline.GetControlPoints(), ElementsAreArray(points));
}

TEST(BSplineTest, TruncateSplineAtClearsCurveForSmallParameter) {
  BSpline2d spline_ref;
  constexpr int kDegree = 3;
  constexpr int kCapacity = 20;  // Large enough.
  const std::vector<Vector2d> points = {Vector2d(1, 0), Vector2d(0, 1),
                                        Vector2d(1, 1), Vector2d(1, 0),
                                        Vector2d(2, 0), Vector2d(2, 1)};

  ASSERT_OK(spline_ref.Init(kDegree, kCapacity));
  ASSERT_OK(spline_ref.SetUniformKnotVector(
      BSplineBase::NumKnots(points.size(), kDegree)));
  ASSERT_OK(spline_ref.SetControlPoints(points));

  BSpline2d spline = spline_ref;
  Vector2d value;
  // Trunctating exactly at u_min should leave an empty curve.
  ASSERT_OK(spline.TruncateSplineAt(spline.GetKnotVector().front()));
  EXPECT_TRUE(spline.GetKnotVector().empty());
  EXPECT_TRUE(spline.GetControlPoints().empty());
  EXPECT_EQ(spline.EvalCurve(0.0, value).code(), absl::StatusCode::kOutOfRange);

  // Truncating before u_min should leave an empty curve.
  spline = spline_ref;
  ASSERT_OK(spline.TruncateSplineAt(spline.GetKnotVector().front() - 1.0));
  EXPECT_TRUE(spline.GetKnotVector().empty());
  EXPECT_TRUE(spline.GetControlPoints().empty());
  EXPECT_EQ(spline.EvalCurve(0.0, value).code(), absl::StatusCode::kOutOfRange);
}

TEST(BSplineTest, TruncateSplineAtWorksForParametersInValidRange) {
  BSpline2d spline_ref;
  constexpr int kDegree = 3;
  constexpr int kCapacity = 20;  // Large enough.
  const std::vector<Vector2d> points = {Vector2d(0, 0), Vector2d(0, 1),
                                        Vector2d(1, 1), Vector2d(1, 0),
                                        Vector2d(2, 0), Vector2d(2, 1)};

  ASSERT_OK(spline_ref.Init(kDegree, kCapacity));
  ASSERT_OK(spline_ref.SetUniformKnotVector(
      BSplineBase::NumKnots(points.size(), kDegree)));
  ASSERT_OK(spline_ref.SetControlPoints(points));

  // Log old and new control points, as well as spline curve samples for
  // debugging and plotting.
  for (const auto &point : spline_ref.GetControlPoints()) {
    LOG(INFO) << "OLD_CONTROL_POINTS: " << point.transpose() << "\n";
  }

  // Verify that truncation works for values in (umin, umax)
  for (double knot_end_value : {0.01, 0.1, 0.3333, 0.6, 0.9, 0.999}) {
    constexpr double kAcceptableError = 1e-7;
    SCOPED_TRACE(absl::StrCat("knot_end_value= ", knot_end_value));
    const std::string log_prefix = absl::StrCat("E", knot_end_value);
    BSpline2d spline = spline_ref;
    Vector2d expected_position_at_end;
    ASSERT_OK(spline.EvalCurve(knot_end_value, expected_position_at_end));
    ASSERT_OK(spline.TruncateSplineAt(knot_end_value));

    EXPECT_THAT(spline.GetControlPoints().back(),
                IsApprox(expected_position_at_end, kAcceptableError));
    EXPECT_DOUBLE_EQ(spline.GetKnotVector().back(), knot_end_value);

    for (const auto &point : spline.GetControlPoints()) {
      LOG(INFO) << "NEW_CONTROL_POINTS: " << point.transpose() << "\n";
    }

    for (double u = spline.GetKnotVector().front();
         u <= spline.GetKnotVector().back(); u += 0.001) {
      Vector2d curve_reference;
      ASSERT_OK(spline_ref.EvalCurve(u, curve_reference));
      Vector2d curve_value;
      ASSERT_OK(spline.EvalCurve(u, curve_value));
      EXPECT_THAT(curve_value, IsApprox(curve_reference, kAcceptableError))
          << " For u= " << u << " knot_end_value = " << knot_end_value
          << " diff= " << (curve_value - curve_reference).transpose();
      LOG(INFO) << log_prefix << "-CURVE_VALUE: " << u << " "
                << curve_value.transpose() << "\n";
    }
  }
}

TEST(BSplineTest, ExtendWithControlPointsFailsForInvalidInput) {
  BSpline2d spline;
  const std::vector<Vector2d> points = {Vector2d(0, 0), Vector2d(0, 1),
                                        Vector2d(1, 1), Vector2d(1, 0)};

  // Expect that unimplemented cases are detected.
  ASSERT_OK(spline.Init(3, 100));
  ASSERT_OK(
      spline.SetUniformKnotVector(BSplineBase::NumKnots(points.size(), 3)));
  ASSERT_OK(spline.SetControlPoints(points));

  EXPECT_EQ(spline.ExtendWithControlPoints(points).code(),
            absl::StatusCode::kUnimplemented);

  // Expect that preconditions are checked.
  ASSERT_OK(spline.Init(2, 10));
  ASSERT_OK(
      spline.SetUniformKnotVector(BSplineBase::NumKnots(points.size(), 2)));
  ASSERT_OK(spline.SetControlPoints(points));

  EXPECT_EQ(
      spline.ExtendWithControlPoints(absl::MakeConstSpan(points.data(), 1))
          .code(),
      absl::StatusCode::kUnimplemented);

  EXPECT_EQ(spline.ExtendWithControlPoints(points).code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST(BSplineTest, ExtendWithControlPointsWorks) {
  constexpr double kEpsilon = 1e-12;
  BSpline2d spline_ref;
  constexpr int kDegree = 2;
  constexpr int kCapacity = 50;  // Large enough.
  const std::vector<Vector2d> points = {Vector2d(0, 0), Vector2d(0, 1),
                                        Vector2d(1, 1), Vector2d(1, 0),
                                        Vector2d(2, 0), Vector2d(2, 1)};
  const std::vector<Vector2d> extra_points = {Vector2d(3, 1), Vector2d(3, 0),
                                              Vector2d(4, 0), Vector2d(4, 1),
                                              Vector2d(4, 2), Vector2d(4, 3)};

  ASSERT_OK(spline_ref.Init(kDegree, kCapacity));
  ASSERT_OK(spline_ref.SetUniformKnotVector(
      BSplineBase::NumKnots(points.size(), kDegree)));
  ASSERT_OK(spline_ref.SetControlPoints(points));

  for (double u = spline_ref.GetKnotVector().front();
       u <= spline_ref.GetKnotVector().back(); u += 0.01) {
    Vector2d curve_value;
    ASSERT_OK(spline_ref.EvalCurve(u, curve_value));
    LOG(INFO) << "REF_CURVE: " << u << " " << curve_value.transpose();
  }
  for (const auto &point : spline_ref.GetControlPoints()) {
    LOG(INFO) << "REF_CONTROL_POINTS: " << point.transpose();
  }

  // Extend the curve with a different number of extra control points
  // and expect the original section to be unchanged, while the new section
  // should connect smoothly.
  for (int extra_point_count = 2; extra_point_count < extra_points.size();
       ++extra_point_count) {
    SCOPED_TRACE(absl::StrCat("extra_point_count= ", extra_point_count));
    BSpline2d spline = spline_ref;

    ASSERT_OK(spline.ExtendWithControlPoints(
        absl::MakeConstSpan(extra_points.data(), extra_point_count)));

    const std::string log_prefix = absl::StrCat("P", extra_point_count);
    for (const auto &point : spline.GetControlPoints()) {
      LOG(INFO) << log_prefix
                << "-EXTENDED_CONTROL_POINTS: " << point.transpose();
    }

    std::vector<Vector2d> curve_value(2);
    std::vector<Vector2d> ref_curve_value(2);
    for (double u = spline.GetKnotVector().front();
         u <= spline.GetKnotVector().back(); u += 0.01) {
      ASSERT_OK(spline.EvalCurveAndDerivatives(
          u, absl::MakeSpan(curve_value.data(), curve_value.size())));

      if (u < spline_ref.GetKnotVector().back()) {
        ASSERT_OK(spline_ref.EvalCurveAndDerivatives(
            u, absl::MakeSpan(ref_curve_value.data(), ref_curve_value.size())));
        EXPECT_THAT(curve_value[0], IsApprox(ref_curve_value[0], kEpsilon));
        EXPECT_THAT(curve_value[1], IsApprox(ref_curve_value[1], kEpsilon));
      }
      LOG(INFO) << log_prefix << "-EXTENDED_CURVE: " << u << " pos "
                << curve_value[0].transpose() << " vel "
                << curve_value[1].transpose() << "\n";
    }
    // Last control point should be interpolated.
    ASSERT_OK(spline.EvalCurve(spline.GetKnotVector().back(), curve_value[0]));
    EXPECT_THAT(curve_value[0],
                IsApprox(extra_points[extra_point_count - 1], kEpsilon));
  }
}

// This isn't really a test, but it is useful for visualizing the geometry of
// rounded poly-lines for different polynomial degrees.
TEST(BSplineTest, DISABLED_CornerRounding) {
  const std::vector<Vector2d> waypoints = {
      Vector2d(0.0, 0.0), Vector2d(1.0, 0.0), Vector2d(1.0, 1.0),
      Vector2d(2.0, 1.0), Vector2d(2.0, 2.0)};

  for (int degree = 1; degree <= 3; ++degree) {
    std::vector<Vector2d> points;
    PolyLineToBspline3Waypoints(waypoints, 0.1, &points);
    BSpline2d spline;
    ASSERT_OK(
        spline.Init(degree, BSplineBase::NumKnots(points.size(), degree)));
    ASSERT_OK(spline.SetUniformKnotVector(
        BSplineBase::NumKnots(points.size(), degree)));
    ASSERT_OK(spline.SetControlPoints(points));

    const std::string log_prefix = absl::StrCat("D", degree);
    for (const auto &point : spline.GetControlPoints()) {
      LOG(INFO) << log_prefix << "-CONTROL_POINTS: " << point.transpose()
                << "\n";
    }

    for (double u = spline.GetKnotVector().front();
         u <= spline.GetKnotVector().back(); u += 0.001) {
      std::vector<Vector2d> curve_values(degree + 1);
      ASSERT_OK(spline.EvalCurveAndDerivatives(
          u, absl::MakeSpan(curve_values.data(), curve_values.size())));
      for (int i = 0; i < curve_values.size(); ++i) {
        LOG(INFO) << log_prefix << "-CURVE_VALUE-" << i << " " << u << " "
                  << curve_values[i].transpose() << " ";
      }
    }
  }
}

}  // namespace
}  // namespace trajectory_planning

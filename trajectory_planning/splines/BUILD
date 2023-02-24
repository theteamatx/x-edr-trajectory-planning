package(default_visibility = ["//visibility:public"])

cc_library(
    name = "splines",
    srcs = [
        "bspline_base.cc",
        "bsplineq.cc",
        "cubic_spline.cc",
        "spline_utils.cc",
    ],
    hdrs = [
        "bspline.h",
        "bspline_base.h",
        "bsplineq.h",
        "cubic_spline.h",
        "spline_utils.h",
    ],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@eigenmath//eigenmath",
    ],
)

cc_test(
    name = "spline_utils_test",
    srcs = [
        "spline_utils_test.cc",
    ],
    deps = [
        ":splines",
        "@com_google_googletest//:gtest_main",
        "@eigenmath//eigenmath:matchers",
    ],
)

cc_test(
    name = "bspline_test",
    srcs = [
        "bspline_test.cc",
    ],
    deps = [
        ":splines",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@eigenmath//eigenmath",
        "@eigenmath//eigenmath:matchers",
    ],
)

cc_test(
    name = "bsplineq_test",
    srcs = [
        "bsplineq_test.cc",
    ],
    deps = [
        ":splines",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@eigenmath//eigenmath",
        "@eigenmath//eigenmath:matchers",
    ],
)

cc_library(
    name = "finite_difference_test_utils",
    testonly = True,
    hdrs = [
        "finite_difference_test_utils.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@eigenmath//eigenmath",
    ],
)

cc_test(
    name = "cubic_spline_test",
    srcs = [
        "cubic_spline_test.cc",
    ],
    deps = [
        ":finite_difference_test_utils",
        ":splines",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

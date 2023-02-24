licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "time_optimal_path_timing",
    srcs = ["time_optimal_path_timing.cc"],
    hdrs = ["time_optimal_path_timing.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@eigen",
    ],
)

cc_test(
    name = "time_optimal_path_timing_test",
    srcs = ["time_optimal_path_timing_test.cc"],
    deps = [
        ":time_optimal_path_timing",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "timeable_path",
    hdrs = [
        "timeable_path.h",
    ],
    deps = [
        ":time_optimal_path_timing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@eigenmath//eigenmath",
    ],
)

cc_library(
    name = "trajectory_planner",
    hdrs = [
        "trajectory_planner.h",
    ],
    deps = [
        ":timeable_path",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@eigenmath//eigenmath",
    ],
)

cc_library(
    name = "path_timing_trajectory",
    srcs = [
        "path_timing_trajectory.cc",
    ],
    hdrs = [
        "path_timing_trajectory.h",
    ],
    linkopts = ["-lm"],
    deps = [
        "time_optimal_path_timing",
        ":time",
        ":timeable_path",
        ":trajectory_planner",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@eigenmath//eigenmath",
    ],
)

cc_test(
    name = "path_timing_trajectory_test",
    srcs = [
        "path_timing_trajectory_test.cc",
    ],
    deps = [
        ":path_timing_trajectory",
        ":time",
        ":timeable_path_cartesian_spline",
        ":timeable_path_joint_spline",
        ":trajectory_buffer",
        ":trajectory_test_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@eigenmath//eigenmath",
        "@eigenmath//eigenmath:matchers",
    ],
)

cc_library(
    name = "path_tools",
    srcs = ["path_tools.cc"],
    hdrs = ["path_tools.h"],
    deps = [
        "//trajectory_planning/splines",
        "@com_google_absl//absl/status",
        "@eigenmath//eigenmath",
    ],
)

cc_test(
    name = "path_tools_test",
    srcs = ["path_tools_test.cc"],
    deps = [
        ":path_tools",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@eigenmath//eigenmath",
        "@eigenmath//eigenmath:matchers",
    ],
)

cc_library(
    name = "timeable_path_cartesian_spline",
    srcs = [
        "timeable_path_cartesian_spline.cc",
    ],
    hdrs = [
        "timeable_path_cartesian_spline.h",
    ],
    linkopts = ["-lm"],
    deps = [
        "time_optimal_path_timing",
        ":path_tools",
        ":timeable_path",
        "//trajectory_planning/splines",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@eigenmath//eigenmath",
    ],
)

cc_library(
    name = "timeable_path_joint_spline",
    srcs = [
        "timeable_path_joint_spline.cc",
    ],
    hdrs = [
        "timeable_path_joint_spline.h",
    ],
    linkopts = ["-lm"],
    deps = [
        "time_optimal_path_timing",
        ":path_tools",
        ":timeable_path",
        "//trajectory_planning/splines",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@eigenmath//eigenmath",
    ],
)

cc_library(
    name = "sampled_trajectory",
    srcs = ["sampled_trajectory.cc"],
    hdrs = ["sampled_trajectory.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@eigenmath//eigenmath",
    ],
)

cc_test(
    name = "sampled_trajectory_test",
    srcs = ["sampled_trajectory_test.cc"],
    deps = [
        ":sampled_trajectory",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "trajectory_buffer",
    srcs = ["trajectory_buffer.cc"],
    hdrs = ["trajectory_buffer.h"],
    deps = [
        ":rescale_to_stop",
        ":sampled_trajectory",
        ":time",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@eigenmath//eigenmath",
    ],
)

cc_test(
    name = "trajectory_buffer_test",
    srcs = ["trajectory_buffer_test.cc"],
    deps = [
        ":sampled_trajectory",
        ":time",
        ":trajectory_buffer",
        ":trajectory_test_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@eigenmath//eigenmath",
        "@eigenmath//eigenmath:matchers",
    ],
)

cc_library(
    name = "rescale_to_stop",
    srcs = ["rescale_to_stop.cc"],
    hdrs = ["rescale_to_stop.h"],
    deps = [
        ":sampled_trajectory",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@eigenmath//eigenmath",
    ],
)

cc_test(
    name = "rescale_to_stop_test",
    srcs = ["rescale_to_stop_test.cc"],
    deps = [
        ":rescale_to_stop",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@eigenmath//eigenmath:matchers",
    ],
)

cc_library(
    name = "time",
    hdrs = ["time.h"],
    deps = [
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "trajectory_test_utils",
    hdrs = ["trajectory_test_utils.h"],
    deps = [
        "@com_google_absl//absl/types:span",
        "@eigenmath//eigenmath",
    ],
)

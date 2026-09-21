// road_centerline_node.cpp
// ROS2 node: subscribes to a camera image, runs the road centerline CNN,
// publishes CenterlineResult + a sensor_msgs/Image with visual overlays.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include <torch/script.h>
#include <torch/cuda.h>
#include <ATen/Context.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <limits>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <geometry_msgs/msg/twist.hpp>
#if __has_include(<cv_bridge/cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge/cv_bridge.hpp>
#elif __has_include(<cv_bridge/cv_bridge.h>)
#include <cv_bridge/cv_bridge.h>
#else
#include <cv_bridge/cv_bridge/cv_bridge.h>
#endif

#include "road_centerline/msg/centerline_result.hpp"
#include "control_interfaces/msg/control_msg.hpp"

// Expand a leading "~" to $HOME so path parameters (model_path, config_path,
// flat_world_yaml, intrinsics_yaml) work across machines/users rather than
// being tied to one hardcoded home directory. Leaves the path unchanged if it
// doesn't start with "~/" (or is exactly "~") or if $HOME isn't set.
static std::string expandUserPath(const std::string & path)
{
    if (path.empty() || path[0] != '~') return path;
    if (path.size() > 1 && path[1] != '/') return path;   // e.g. "~otheruser"
    const char * home = std::getenv("HOME");
    if (!home) return path;
    return std::string(home) + path.substr(1);
}

// ── Colours (BGR for OpenCV) ──────────────────────────────────────────────────
static const cv::Scalar _PRED  (255, 160,  50);   // blue
static const cv::Scalar _GUIDE ( 80, 210,   0);   // green
static const cv::Scalar _RED   ( 30,  30, 220);

// ── Model geometry ─────────────────────────────────────────────────────────────
struct ModelConfig {
    int    image_width  = 160;
    int    image_height = 120;
    int    n_buckets    = 40;
    int    n_rows       = 3;
    std::vector<double> row_fractions;
    std::vector<double> norm_mean;
    std::vector<double> norm_std;
    double crop_top     = 0.0;
    double crop_bottom  = 0.0;
    std::vector<std::vector<double>> bucket_edges; // [n_rows][n_buckets+1]
};

static ModelConfig loadConfig(const std::string & path)
{
    YAML::Node y = YAML::LoadFile(path);
    ModelConfig c;
    c.image_width  = y["image_width"].as<int>();
    c.image_height = y["image_height"].as<int>();
    c.n_buckets    = y["n_buckets"].as<int>();
    c.n_rows       = y["n_rows"].as<int>();
    for (auto v : y["row_fractions"]) c.row_fractions.push_back(v.as<double>());
    for (auto v : y["norm_mean"])     c.norm_mean.push_back(v.as<double>());
    for (auto v : y["norm_std"])      c.norm_std.push_back(v.as<double>());
    c.crop_top    = y["crop_top"].as<double>();
    c.crop_bottom = y["crop_bottom"].as<double>();
    for (auto row : y["bucket_edges"]) {
        std::vector<double> edges;
        for (auto e : row) edges.push_back(e.as<double>());
        c.bucket_edges.push_back(edges);
    }
    return c;
}

// ── Peak detection ─────────────────────────────────────────────────────────────
struct Peak {
    double cx_frac;
    float  conf;
    bool   is_primary = false;  // true for the temporal-bias winner
};

// Returns all significant cluster peaks sorted left-to-right.
//
// Temporal bias: clusters near prev_cx_frac get a confidence boost so the
//   tracker stays with the same road lane when two clusters have similar conf.
//   prev_cx_frac < 0 means no prior frame (bias disabled).
//
// Hint peaks: secondary peak threshold for this row is lowered from
//   fork_suppress_thresh to fork_hint_thresh when a hint peak (from the
//   farther row) lies within fork_hint_proximity of the candidate secondary.
//   hint_peaks = nullptr disables hinting.
static std::vector<Peak> findAllPeaks(
    const std::vector<float>  & probs,
    const std::vector<double> & edges,
    float  cluster_thresh_frac,
    float  fork_suppress_thresh    = 0.9f,
    double prev_cx_frac            = -1.0,
    float  temporal_bias_weight    = 0.1f,
    float  temporal_proximity_range = 0.1f,
    const std::vector<Peak> * hint_peaks = nullptr,
    float  fork_hint_thresh        = 0.55f,
    float  fork_hint_proximity     = 0.20f)
{
    int   n         = static_cast<int>(probs.size());
    float peak_conf = *std::max_element(probs.begin(), probs.end());
    float thresh    = std::max(0.05f, peak_conf * cluster_thresh_frac);

    struct Cluster { int lo, hi; float peak_conf; double cx_frac; float eff_conf; };
    std::vector<Cluster> clusters;
    int start = -1;
    for (int b = 0; b < n; ++b) {
        if (probs[b] >= thresh) {
            if (start < 0) start = b;
        } else if (start >= 0) {
            float pc = *std::max_element(probs.begin()+start, probs.begin()+b);
            double tw = 0, wx = 0;
            for (int i = start; i < b; ++i) {
                double cx = (edges[i] + edges[i+1]) * 0.5;
                tw += probs[i]; wx += cx * probs[i];
            }
            clusters.push_back({start, b-1, pc,
                tw > 0 ? wx/tw : (edges[start]+edges[b])*0.5, pc});
            start = -1;
        }
    }
    if (start >= 0) {
        float pc = *std::max_element(probs.begin()+start, probs.end());
        double tw = 0, wx = 0;
        for (int i = start; i < n; ++i) {
            double cx = (edges[i] + edges[i+1]) * 0.5;
            tw += probs[i]; wx += cx * probs[i];
        }
        clusters.push_back({start, n-1, pc,
            tw > 0 ? wx/tw : (edges[start]+edges[n])*0.5, pc});
    }

    if (clusters.empty()) {
        int pb = static_cast<int>(std::max_element(probs.begin(), probs.end()) - probs.begin());
        return {{(edges[pb]+edges[pb+1])*0.5, peak_conf, true}};
    }

    // Apply temporal proximity bias to effective confidence
    if (prev_cx_frac >= 0.0) {
        for (auto & cl : clusters) {
            double dist = std::abs(cl.cx_frac - prev_cx_frac);
            float proximity = static_cast<float>(
                std::max(0.0, 1.0 - dist / temporal_proximity_range));
            cl.eff_conf = cl.peak_conf * (1.0f + temporal_bias_weight * proximity);
        }
    }

    // Sort by effective confidence descending to find temporal-bias winner
    std::sort(clusters.begin(), clusters.end(),
        [](const Cluster & a, const Cluster & b){ return a.eff_conf > b.eff_conf; });

    float primary_conf = clusters[0].peak_conf;   // raw conf for threshold math
    std::vector<Peak> result;
    result.push_back({clusters[0].cx_frac, clusters[0].peak_conf, true});   // primary

    // Accept secondary if it meets the fork suppression threshold.
    // Lower the threshold when a hint peak (from the far row) is nearby.
    if (clusters.size() > 1) {
        float sec_thresh = primary_conf * fork_suppress_thresh;
        if (hint_peaks && !hint_peaks->empty()) {
            double sec_cx = clusters[1].cx_frac;
            for (const auto & hp : *hint_peaks) {
                if (std::abs(hp.cx_frac - sec_cx) < fork_hint_proximity) {
                    sec_thresh = std::min(sec_thresh,
                                         primary_conf * fork_hint_thresh);
                    break;
                }
            }
        }
        if (clusters[1].peak_conf >= sec_thresh)
            result.push_back({clusters[1].cx_frac, clusters[1].peak_conf, false});
    }

    // Sort left-to-right for fork drawing (primary may no longer be [0])
    std::sort(result.begin(), result.end(),
        [](const Peak & a, const Peak & b){ return a.cx_frac < b.cx_frac; });
    return result;
}

// Returns the highest-confidence (primary) peak from a row's peak list.
// After sorting by cx_frac the primary may be at any index, so we search by flag.
static const Peak & primaryOf(const std::vector<Peak> & peaks)
{
    for (const auto & p : peaks) if (p.is_primary) return p;
    return peaks[0];   // fallback: should never be reached
}

// ── Drawing helpers ────────────────────────────────────────────────────────────

static void dashedHLine(cv::Mat & img, int y, cv::Scalar color, int dash, int thickness)
{
    int w = img.cols;
    if (y < 0 || y >= img.rows) return;
    for (int x = 0; x < w; x += dash * 2)
        cv::line(img, {x, y}, {std::min(x+dash-1, w-1), y}, color, thickness, cv::LINE_AA);
}

static void drawBranch(cv::Mat & canvas,
                       cv::Point p_near, cv::Point p_mid, cv::Point * p_far,
                       int x_lo, int x_hi, int lw)
{
    cv::line(canvas, p_near, p_mid, _PRED, lw, cv::LINE_AA);
    if (!p_far) return;

    cv::Point2d _p2(p_near), _p1(p_mid), _p0(*p_far);
    cv::Point2d seg   = _p1 - _p2;
    double seg_len    = cv::norm(seg);
    cv::Point2d tdir_s = seg_len > 0 ? seg / seg_len : cv::Point2d(0, -1);
    cv::Point2d chord  = _p0 - _p1;
    double chord_len   = cv::norm(chord);
    cv::Point2d chord_hat = chord_len > 0 ? chord / chord_len : tdir_s;
    double dot    = tdir_s.x*chord_hat.x + tdir_s.y*chord_hat.y;
    cv::Point2d tdir_e = 2.0*dot*chord_hat - tdir_s;
    double alpha  = chord_len / 3.0;
    cv::Point2d cp1 = _p1 + alpha * tdir_s;
    cv::Point2d cp2 = _p0 - alpha * tdir_e;
    // Clamp control-point y so neither goes above the far scan line.
    // Bezier convex-hull property: if all four control points have y >= _p0.y
    // then every point on the curve does too — no flat segments, still smooth.
    cp1.y = std::max(cp1.y, _p0.y);
    cp2.y = std::max(cp2.y, _p0.y);

    // Swing budget: constrain lateral excursion to 0.5 × vertical span
    double swing = 0.5 * (std::abs(_p1.y - _p2.y) + std::abs(_p0.y - _p1.y));
    double pts_x_min = std::min({_p2.x, _p1.x, _p0.x});
    double pts_x_max = std::max({_p2.x, _p1.x, _p0.x});
    x_lo = static_cast<int>(std::max((double)x_lo, pts_x_min - swing));
    x_hi = static_cast<int>(std::min((double)x_hi, pts_x_max + swing));

    int n_pts = std::max(20, static_cast<int>(chord_len));
    std::vector<cv::Point> pts;
    for (int i = 0; i <= n_pts; ++i) {
        double t = static_cast<double>(i) / n_pts;
        double mt = 1.0 - t;
        cv::Point2d p = mt*mt*mt*_p1
                      + 3*mt*mt*t*cp1
                      + 3*mt*t*t*cp2
                      + t*t*t*_p0;
        int x = std::clamp(static_cast<int>(std::round(p.x)), x_lo, x_hi);
        int y = std::clamp(static_cast<int>(std::round(p.y)), 0, canvas.rows-1);
        pts.push_back({x, y});
    }
    std::vector<std::vector<cv::Point>> contours{pts};
    cv::polylines(canvas, contours, false, _PRED, lw, cv::LINE_AA);
}

// ── Flat-world ground projection ──────────────────────────────────────────────

struct FlatWorldExtrinsic {
    double height_m   = 1.2;
    double pitch_deg  = 0.0;
    double roll_deg   = 0.0;
    double x_offset_m = 0.0;
    double y_offset_m = 0.0;
};

static FlatWorldExtrinsic loadExtrinsic(const std::string & path)
{
    YAML::Node y = YAML::LoadFile(path);
    FlatWorldExtrinsic e;
    e.height_m   = y["camera_height_m"].as<double>();
    e.pitch_deg  = y["camera_pitch_deg"].as<double>();
    e.roll_deg   = y["camera_roll_deg"].as<double>();
    e.x_offset_m = y["camera_x_offset_m"].as<double>();
    e.y_offset_m = y["camera_y_offset_m"].as<double>();
    return e;
}

// Reads a camera K matrix from a yaml file.
// Supports ROS camera_info format (camera_matrix.data) or simple K flat array.
// Scales K to match the model's pixel resolution.
static cv::Matx33d loadAndScaleK(const std::string & path, int model_w, int model_h)
{
    YAML::Node y = YAML::LoadFile(path);

    std::vector<double> kv;
    if (y["camera_matrix"] && y["camera_matrix"]["data"]) {
        for (auto v : y["camera_matrix"]["data"]) kv.push_back(v.as<double>());
    } else {
        for (auto v : y["K"]) kv.push_back(v.as<double>());
    }

    cv::Matx33d K(kv[0], kv[1], kv[2],
                  kv[3], kv[4], kv[5],
                  kv[6], kv[7], kv[8]);

    // Scale K from native image resolution to model resolution
    int native_w = model_w, native_h = model_h;
    if (y["image_width"])  native_w = y["image_width"].as<int>();
    if (y["image_height"]) native_h = y["image_height"].as<int>();

    double sx = static_cast<double>(model_w) / native_w;
    double sy = static_cast<double>(model_h) / native_h;
    K(0, 0) *= sx;  // fx
    K(0, 2) *= sx;  // cx
    K(1, 1) *= sy;  // fy
    K(1, 2) *= sy;  // cy
    return K;
}

// Rotation matrix: vehicle frame (X=fwd, Y=left, Z=up) → camera frame (X=right, Y=down, Z=fwd)
// R = Rz(roll) @ Rx(pitch) @ R_base
static cv::Matx33d buildRotation(double pitch_deg, double roll_deg)
{
    // Base: camera pointing straight ahead, perfectly level
    cv::Matx33d R_base(0, -1,  0,
                       0,  0, -1,
                       1,  0,  0);

    double pitch = pitch_deg * M_PI / 180.0;
    double roll  = roll_deg  * M_PI / 180.0;
    double cp = std::cos(pitch), sp = std::sin(pitch);
    double cr = std::cos(roll),  sr = std::sin(roll);

    cv::Matx33d Rx(1,  0,   0,
                   0,  cp, -sp,
                   0,  sp,  cp);

    cv::Matx33d Rz(cr, -sr,  0,
                   sr,  cr,  0,
                   0,   0,   1);

    return Rz * Rx * R_base;
}

// Unproject a model-space pixel to the flat ground plane (Z=0 in vehicle frame).
// Returns false if the ray points away from the ground.
static bool unprojectToGround(double u, double v,
                               const cv::Matx33d & K,
                               const cv::Matx33d & R,
                               const FlatWorldExtrinsic & e,
                               double & X_fwd, double & Y_left)
{
    // Ray direction in camera frame (un-normalised)
    double fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);
    cv::Vec3d d_cam((u - cx) / fx, (v - cy) / fy, 1.0);

    // Ray direction in vehicle frame
    cv::Vec3d d_veh = R.t() * d_cam;

    // Ground plane Z=0: height + t * d_veh[2] = 0
    if (d_veh[2] >= 0.0) return false;   // ray pointing upward, won't hit ground
    double t = -e.height_m / d_veh[2];

    X_fwd  = e.x_offset_m + t * d_veh[0];
    Y_left = e.y_offset_m + t * d_veh[1];
    return true;
}

// Back-project a vehicle-frame ground point to model-space pixel coordinates.
static cv::Point2d projectGroundToModel(double X_fwd, double Y_left,
                                         const cv::Matx33d & K,
                                         const cv::Matx33d & R,
                                         const FlatWorldExtrinsic & e)
{
    cv::Vec3d p_rel(X_fwd - e.x_offset_m, Y_left - e.y_offset_m, -e.height_m);
    cv::Vec3d p_cam = R * p_rel;
    double u = K(0, 0) * p_cam[0] / p_cam[2] + K(0, 2);
    double v = K(1, 1) * p_cam[1] / p_cam[2] + K(1, 2);
    return {u, v};
}

// Sample the road centerline path in model-space pixel coordinates.
// Produces a linear segment (p_near → p_mid) followed by a cubic Bézier
// (p_mid → p_far) using the same tangent logic as drawBranch.
// p_far may be nullptr — in that case only the linear segment is returned.
static std::vector<cv::Point2d> samplePathModel(cv::Point2d p_near,
                                                 cv::Point2d p_mid,
                                                 const cv::Point2d * p_far,
                                                 int n_lin = 20,
                                                 int n_bez = 50)
{
    std::vector<cv::Point2d> pts;
    pts.reserve(n_lin + n_bez);

    // Linear segment: near → mid
    for (int i = 0; i <= n_lin; ++i) {
        double t = static_cast<double>(i) / n_lin;
        pts.push_back(p_near + t * (p_mid - p_near));
    }

    if (!p_far) return pts;

    // Cubic Bézier: mid → far (same tangent construction as drawBranch)
    cv::Point2d seg      = p_mid - p_near;
    double      seg_len  = cv::norm(seg);
    cv::Point2d tdir_s   = seg_len > 0 ? seg / seg_len : cv::Point2d(0, -1);
    cv::Point2d chord    = *p_far - p_mid;
    double      clen     = cv::norm(chord);
    cv::Point2d chord_hat = clen > 0 ? chord / clen : tdir_s;
    double dot    = tdir_s.x * chord_hat.x + tdir_s.y * chord_hat.y;
    cv::Point2d tdir_e = 2.0 * dot * chord_hat - tdir_s;
    double alpha  = clen / 3.0;
    cv::Point2d cp1 = p_mid + alpha * tdir_s;
    cv::Point2d cp2 = *p_far - alpha * tdir_e;
    cp1.y = std::max(cp1.y, p_far->y);
    cp2.y = std::max(cp2.y, p_far->y);

    for (int i = 1; i <= n_bez; ++i) {
        double t = static_cast<double>(i) / n_bez;
        double mt = 1.0 - t;
        cv::Point2d p = mt*mt*mt * p_mid
                      + 3*mt*mt*t * cp1
                      + 3*mt*t*t  * cp2
                      + t*t*t     * (*p_far);
        pts.push_back(p);
    }
    return pts;
}

// Walk the sampled path on the ground plane, find the first point at or beyond
// lookahead_m from the vehicle origin, and return the pure-pursuit curvature.
// Also fills lookahead_img_pt (model pixels) for the visual overlay.
// Returns NaN when the path cannot be unprojected or is too short.
static float computeSteeringCurvature(const std::vector<cv::Point2d> & path_model,
                                       const cv::Matx33d & K,
                                       const cv::Matx33d & R,
                                       const FlatWorldExtrinsic & e,
                                       double lookahead_m,
                                       cv::Point2d & lookahead_img_pt)
{
    const float NaN = std::numeric_limits<float>::quiet_NaN();

    // Unproject every sample to vehicle ground frame; keep valid ones
    struct GroundPt { double X, Y; cv::Point2d img; };
    std::vector<GroundPt> gpts;
    gpts.reserve(path_model.size());
    for (const auto & p : path_model) {
        double X, Y;
        if (unprojectToGround(p.x, p.y, K, R, e, X, Y))
            gpts.push_back({X, Y, p});
    }
    if (gpts.empty()) return NaN;

    // Find first point at or beyond lookahead_m from vehicle origin (0,0)
    for (const auto & g : gpts) {
        double dist = std::hypot(g.X, g.Y);
        if (dist >= lookahead_m) {
            lookahead_img_pt = g.img;
            // κ = 2·Y_L / L²   (pure pursuit, Y_L = lateral, L = straight-line dist)
            return static_cast<float>(2.0 * g.Y / (dist * dist));
        }
    }

    // Lookahead extends beyond the visible path: use farthest point
    const auto & last = gpts.back();
    double dist = std::hypot(last.X, last.Y);
    if (dist < 1e-3) return NaN;
    lookahead_img_pt = last.img;
    return static_cast<float>(2.0 * last.Y / (dist * dist));
}

// ── ROS2 node ──────────────────────────────────────────────────────────────────
class RoadCenterlineNode : public rclcpp::Node
{
public:
    explicit RoadCenterlineNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
        : Node("road_centerline_node", options)
    {
        declare_parameter("model_path",          "road_model.pt");
        declare_parameter("config_path",         "road_model_config.yaml");
        declare_parameter("image_topic",         "/camera/image_raw");
        declare_parameter("output_topic",        "road_centerline");
        declare_parameter("visual_topic",        "road_centerline/visual");
        declare_parameter("control_topic",       "control_cmd");
        declare_parameter("road_threshold",      0.5);
        declare_parameter("min_peak_conf",       0.2);
        declare_parameter("cluster_thresh",      0.45);
        declare_parameter("debug_mode",           false);
        declare_parameter("min_display_height",   480);
        declare_parameter("far_row_curve_thresh", 0.0);
        declare_parameter("fork_confirm_frames",  4);
        declare_parameter("flat_world_yaml",           std::string(""));
        declare_parameter("intrinsics_yaml",           std::string(""));
        declare_parameter("lookahead_distance_m",      3.0);
        declare_parameter("forward_speed_mps",         0.0);
        declare_parameter("vehicle_wheelbase_m",       0.3);
        declare_parameter("temporal_bias_weight",      0.1);
        declare_parameter("temporal_proximity_range",  0.1);
        declare_parameter("fork_hint_thresh",          0.55);
        declare_parameter("fork_hint_proximity",       0.20);
        declare_parameter("road_momentum_decay",       0.6);
        declare_parameter("road_momentum_thresh",      0.25);
        declare_parameter("very_high_confidence_value", 0.4);
        declare_parameter("maximum_speed",              2.0);
        declare_parameter("maximum_bias_curvature",     0.1);
        declare_parameter("maximum_curvature_speed_reduction", 0.5);

        road_threshold_      = get_parameter("road_threshold").as_double();
        min_peak_conf_       = get_parameter("min_peak_conf").as_double();
        cluster_thresh_      = get_parameter("cluster_thresh").as_double();
        debug_mode_          = get_parameter("debug_mode").as_bool();
        min_display_height_  = get_parameter("min_display_height").as_int();
        far_row_curve_thresh_= get_parameter("far_row_curve_thresh").as_double();
        fork_confirm_frames_ = get_parameter("fork_confirm_frames").as_int();
        lookahead_distance_m_      = get_parameter("lookahead_distance_m").as_double();
        forward_speed_mps_         = get_parameter("forward_speed_mps").as_double();
        vehicle_wheelbase_m_       = get_parameter("vehicle_wheelbase_m").as_double();
        temporal_bias_weight_      = static_cast<float>(get_parameter("temporal_bias_weight").as_double());
        temporal_proximity_range_  = static_cast<float>(get_parameter("temporal_proximity_range").as_double());
        fork_hint_thresh_          = static_cast<float>(get_parameter("fork_hint_thresh").as_double());
        fork_hint_proximity_       = static_cast<float>(get_parameter("fork_hint_proximity").as_double());
        road_momentum_decay_       = static_cast<float>(get_parameter("road_momentum_decay").as_double());
        road_momentum_thresh_      = static_cast<float>(get_parameter("road_momentum_thresh").as_double());
        very_high_confidence_value_ = static_cast<float>(get_parameter("very_high_confidence_value").as_double());
        maximum_speed_              = static_cast<float>(get_parameter("maximum_speed").as_double());
        maximum_bias_curvature_     = static_cast<float>(get_parameter("maximum_bias_curvature").as_double());
        maximum_curvature_speed_reduction_ =
            static_cast<float>(get_parameter("maximum_curvature_speed_reduction").as_double());

        // Flat-world steering (optional)
        {
            auto fw   = expandUserPath(get_parameter("flat_world_yaml").as_string());
            auto intr = expandUserPath(get_parameter("intrinsics_yaml").as_string());
            bool fw_missing   = !fw.empty()   && !std::filesystem::exists(fw);
            bool intr_missing = !intr.empty() && !std::filesystem::exists(intr);

            if (!fw.empty() && !intr.empty() && !fw_missing && !intr_missing)
            {
                // Config not loaded yet; parse image dims first
                YAML::Node mc = YAML::LoadFile(expandUserPath(get_parameter("config_path").as_string()));
                int mw = mc["image_width"].as<int>(), mh = mc["image_height"].as<int>();
                extr_    = loadExtrinsic(fw);
                K_model_ = loadAndScaleK(intr, mw, mh);
                R_cam_   = buildRotation(extr_.pitch_deg, extr_.roll_deg);
                flat_world_loaded_ = true;
                RCLCPP_INFO(get_logger(),
                    "Flat-world steering enabled  h=%.2f m  pitch=%.2f°  roll=%.2f°  lookahead=%.1f m",
                    extr_.height_m, extr_.pitch_deg, extr_.roll_deg, lookahead_distance_m_);
            } else if (fw_missing || intr_missing) {
                RCLCPP_ERROR(get_logger(),
                    "\n"
                    "********************************************************************\n"
                    "***  FLAT-WORLD CALIBRATION FILE(S) NOT FOUND — STEERING DISABLED  ***\n"
                    "***  flat_world_yaml : %-40s %s\n"
                    "***  intrinsics_yaml : %-40s %s\n"
                    "***  steering_curvature will be NaN; desired_speed forced to 0.0.  ***\n"
                    "********************************************************************",
                    fw.c_str(),   fw_missing   ? "(MISSING)" : "(ok)",
                    intr.c_str(), intr_missing ? "(MISSING)" : "(ok)");
            } else {
                RCLCPP_WARN(get_logger(),
                    "Flat-world yaml not configured — steering_curvature will be NaN "
                    "and desired_speed will be forced to 0.0");
            }
        }

        cfg_ = loadConfig(expandUserPath(get_parameter("config_path").as_string()));
        RCLCPP_INFO(get_logger(), "Model config: %dx%d  buckets=%d  rows=%d",
            cfg_.image_width, cfg_.image_height, cfg_.n_buckets, cfg_.n_rows);

        device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        RCLCPP_INFO(get_logger(), "Running inference on %s",
            device_.is_cuda() ? "GPU (CUDA)" : "CPU");

        at::globalContext().setFlushDenormal(true);   // subnormal weights → 0; prevents 100x CPU slowdown
        model_ = torch::jit::load(expandUserPath(get_parameter("model_path").as_string()), device_);
        model_.eval();

        norm_mean_ = torch::tensor(cfg_.norm_mean).to(torch::kFloat32).reshape({1,3,1,1});
        norm_std_  = torch::tensor(cfg_.norm_std).to(torch::kFloat32).reshape({1,3,1,1});

        auto image_topic   = get_parameter("image_topic").as_string();
        auto output_topic  = get_parameter("output_topic").as_string();
        auto visual_topic  = get_parameter("visual_topic").as_string();
        auto control_topic = get_parameter("control_topic").as_string();

        image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            image_topic, 10,
            std::bind(&RoadCenterlineNode::imageCallback, this, std::placeholders::_1));
        result_pub_   = create_publisher<road_centerline::msg::CenterlineResult>(output_topic, 10);
        visual_pub_   = create_publisher<sensor_msgs::msg::Image>(visual_topic, 10);
        steering_pub_ = create_publisher<geometry_msgs::msg::Twist>("steering_cmd", 10);
        control_pub_  = create_publisher<control_interfaces::msg::ControlMsg>(control_topic, 10);

        RCLCPP_INFO(get_logger(), "Listening on '%s'", image_topic.c_str());
        RCLCPP_INFO(get_logger(), "Publishing results on '%s'", output_topic.c_str());
        RCLCPP_INFO(get_logger(), "Publishing visuals on '%s'", visual_topic.c_str());
        RCLCPP_INFO(get_logger(), "Publishing control commands on '%s'", control_topic.c_str());
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Cycle rate: exponential moving average of instantaneous Hz between
        // successive calls, logged at most once/sec so it doesn't spam.
        auto now = std::chrono::steady_clock::now();
        if (have_last_cycle_time_) {
            double dt = std::chrono::duration<double>(now - last_cycle_time_).count();
            if (dt > 0.0) {
                double inst_hz = 1.0 / dt;
                cycle_hz_ = (cycle_hz_ == 0.0) ? inst_hz : 0.9 * cycle_hz_ + 0.1 * inst_hz;
            }
        }
        last_cycle_time_ = now;
        have_last_cycle_time_ = true;
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                              "Cycle rate: %.1f Hz", cycle_hz_);

        // Convert to BGR OpenCV mat
        cv::Mat bgr;
        try {
            bgr = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (const cv_bridge::Exception & e) {
            RCLCPP_WARN(get_logger(), "cv_bridge: %s", e.what());
            return;
        }

        // Resize to model input
        cv::Mat resized;
        cv::resize(bgr, resized,
                   cv::Size(cfg_.image_width, cfg_.image_height), 0, 0, cv::INTER_LINEAR);

        // BGR uint8 → RGB float [0,1] → tensor [1,3,H,W]
        cv::Mat rgb_f;
        cv::cvtColor(resized, rgb_f, cv::COLOR_BGR2RGB);
        rgb_f.convertTo(rgb_f, CV_32F, 1.0/255.0);

        std::vector<cv::Mat> ch(3);
        cv::split(rgb_f, ch);
        torch::Tensor tensor = torch::zeros({1,3,cfg_.image_height,cfg_.image_width}, torch::kFloat32);
        for (int c = 0; c < 3; ++c)
            tensor[0][c] = torch::from_blob(ch[c].data, {cfg_.image_height,cfg_.image_width}, torch::kFloat32).clone();
        tensor = (tensor - norm_mean_) / norm_std_;
        tensor = tensor.to(device_);

        // Forward pass
        torch::jit::IValue output;
        {
            torch::NoGradGuard ng;
            output = model_.forward({tensor});
        }
        auto elems      = output.toTuple()->elements();
        auto cls_logits = elems[0].toTensor().squeeze(0);
        std::vector<torch::Tensor> row_logits(3);
        for (int r = 0; r < 3; ++r)
            row_logits[r] = elems[r+1].toTensor().squeeze(0);

        float road_prob   = torch::softmax(cls_logits, 0)[1].item<float>();
        bool  road_present = road_prob > road_threshold_;

        // Per-row peak detection with temporal bias and hint peaks.
        // Process r0 (far) first so its peaks can hint r1 (mid) secondary threshold.
        std::vector<std::vector<Peak>> all_peaks(3);
        std::vector<std::vector<float>> row_probs(3);
        for (int r = 0; r < 3; ++r) {
            // .data_ptr() reads directly from wherever the tensor lives, so it
            // must be brought back to host memory first when running on CUDA.
            auto sig = torch::sigmoid(row_logits[r]).to(torch::kCPU);
            row_probs[r].assign(sig.data_ptr<float>(),
                                 sig.data_ptr<float>() + cfg_.n_buckets);
        }
        // r0 (far): no hints
        all_peaks[0] = findAllPeaks(row_probs[0], cfg_.bucket_edges[0],
            static_cast<float>(cluster_thresh_), 0.9f,
            prev_cx_frac_[0], temporal_bias_weight_, temporal_proximity_range_);
        // r1 (mid): r0 peaks as hints
        all_peaks[1] = findAllPeaks(row_probs[1], cfg_.bucket_edges[1],
            static_cast<float>(cluster_thresh_), 0.9f,
            prev_cx_frac_[1], temporal_bias_weight_, temporal_proximity_range_,
            &all_peaks[0], fork_hint_thresh_, fork_hint_proximity_);
        // r2 (near): no hints
        all_peaks[2] = findAllPeaks(row_probs[2], cfg_.bucket_edges[2],
            static_cast<float>(cluster_thresh_), 0.9f,
            prev_cx_frac_[2], temporal_bias_weight_, temporal_proximity_range_);

        float conf_sum = 0.0f;
        for (int r = 0; r < 3; ++r)
            conf_sum += primaryOf(all_peaks[r]).conf;
        float mean_conf = conf_sum / 3.0f;
        if (mean_conf < min_peak_conf_) road_present = false;

        // Road momentum: decay when road drops; restore from last valid peaks if
        // momentum is still above threshold (roads don't just disappear).
        if (road_present) {
            road_momentum_ = 1.0f;
            prev_peaks_    = all_peaks;
            for (int r = 0; r < 3; ++r)
                prev_cx_frac_[r] = primaryOf(all_peaks[r]).cx_frac;
        } else {
            road_momentum_ *= road_momentum_decay_;
            fork_frame_count_ = 0;   // reset fork tracking whenever raw signal drops
            if (road_momentum_ >= road_momentum_thresh_ && !prev_peaks_.empty()) {
                all_peaks    = prev_peaks_;
                road_present = true;
            }
        }
        // No road (momentum exhausted too): confidence reads as exactly zero,
        // both in the published CenterlineResult and the visual overlay, since
        // downstream speed determination will key off this value.
        if (!road_present) mean_conf = 0.0f;

        // mean_conf normalized against the "very high confidence" tier, clipped
        // to 1.0 so callers get a clean [0,1] fraction for speed determination.
        float adjusted_mean_conf = (very_high_confidence_value_ > 0.0f)
            ? std::min(1.0f, mean_conf / very_high_confidence_value_)
            : 0.0f;

        // Fork confirmation: both r1 (mid) and r0 (far) must show 2+ peaks for
        // fork_confirm_frames_ consecutive road-present frames before drawing a fork.
        bool fork_raw = road_present && (all_peaks[1].size() >= 2 && all_peaks[0].size() >= 2);
        if (fork_raw) {
            fork_frame_count_++;
        } else {
            fork_frame_count_ = 0;
        }
        bool draw_fork = road_present && (fork_frame_count_ >= fork_confirm_frames_);

        // ── Flat-world steering ───────────────────────────────────────────────
        float steering_curvature = std::numeric_limits<float>::quiet_NaN();
        cv::Point2d lookahead_img_pt(-1, -1);
        if (road_present && flat_world_loaded_) {
            const Peak & r2 = primaryOf(all_peaks[2]);
            const Peak & r1 = primaryOf(all_peaks[1]);
            const Peak & r0 = primaryOf(all_peaks[0]);

            cv::Point2d p_near(r2.cx_frac * cfg_.image_width,
                               cfg_.row_fractions[2] * cfg_.image_height);
            cv::Point2d p_mid (r1.cx_frac * cfg_.image_width,
                               cfg_.row_fractions[1] * cfg_.image_height);
            cv::Point2d p_far_val(r0.cx_frac * cfg_.image_width,
                                  cfg_.row_fractions[0] * cfg_.image_height);
            const cv::Point2d * p_far_ptr =
                (r0.conf >= far_row_curve_thresh_) ? &p_far_val : nullptr;

            auto path = samplePathModel(p_near, p_mid, p_far_ptr);
            steering_curvature = computeSteeringCurvature(
                path, K_model_, R_cam_, extr_, lookahead_distance_m_, lookahead_img_pt);
        }

        // Publish Twist steering command
        if (!std::isnan(steering_curvature)) {
            geometry_msgs::msg::Twist twist;
            twist.linear.x  = forward_speed_mps_;
            twist.angular.z = forward_speed_mps_ * static_cast<double>(steering_curvature);
            steering_pub_->publish(twist);
        }

        // Publish ControlMsg every cycle. listen_to_* tells the downstream
        // control node whether the corresponding field is trustworthy this
        // cycle (valid curvature / road actually detected).
        //
        // road_present already reflects the road-momentum grace period above,
        // so it only goes false once the road has been missing for several
        // consecutive frames. At that point: stop (speed 0), hold the last
        // valid steering command for reference, and tell the listener not to
        // act on steering. The same applies whenever flat-world calibration
        // isn't loaded — without it, steering_curvature can never be valid,
        // so it's not safe to command any forward speed either.
        float final_desired_speed = 0.0f;
        {
            bool have_curvature = !std::isnan(steering_curvature);
            if (have_curvature) {
                last_valid_curvature_      = steering_curvature;
                have_last_valid_curvature_ = true;
            }

            control_interfaces::msg::ControlMsg control_msg;
            if (road_present && flat_world_loaded_) {
                control_msg.desired_curvature  = have_curvature ? steering_curvature : 0.0f;

                // Speed = maximum_speed, scaled down by confidence and by how
                // sharp the turn is. Curvature bias ramps linearly from 1.0
                // (straight) to (1 - maximum_curvature_speed_reduction) at
                // maximum_bias_curvature; beyond that the attenuation caps
                // there rather than continuing to shrink.
                float curvature_for_speed = have_curvature ? steering_curvature : 0.0f;
                float abs_curvature = std::fabs(curvature_for_speed);
                float max_reduction = std::clamp(maximum_curvature_speed_reduction_, 0.0f, 1.0f);
                float curvature_bias_factor = (maximum_bias_curvature_ > 0.0f)
                    ? 1.0f - max_reduction * std::min(1.0f, abs_curvature / maximum_bias_curvature_)
                    : (abs_curvature > 0.0f ? (1.0f - max_reduction) : 1.0f);
                control_msg.desired_speed = maximum_speed_ * adjusted_mean_conf * curvature_bias_factor;

                control_msg.listen_to_steering = have_curvature;
                control_msg.listen_to_speed    = true;
            } else {
                control_msg.desired_curvature  = have_last_valid_curvature_ ?
                    last_valid_curvature_ : 0.0f;
                control_msg.desired_speed      = 0.0f;
                control_msg.listen_to_steering = false;
                control_msg.listen_to_speed    = true;
            }
            final_desired_speed = control_msg.desired_speed;
            control_pub_->publish(control_msg);
        }

        // ── Publish CenterlineResult ──────────────────────────────────────────
        road_centerline::msg::CenterlineResult result;
        result.header          = msg->header;
        result.road_present    = road_present;
        result.road_confidence = road_prob;
        result.peak_confidence = mean_conf;
        for (int r = 0; r < 3; ++r) result.row_confidences[r] = primaryOf(all_peaks[r]).conf;
        if (road_present) {
            for (int r = 2; r >= 0; --r) {
                const Peak & pk = primaryOf(all_peaks[r]);
                geometry_msgs::msg::Point32 pt;
                pt.x = static_cast<float>(pk.cx_frac);
                pt.y = static_cast<float>(cfg_.row_fractions[r]);
                pt.z = pk.conf;
                result.points.push_back(pt);
            }
        }
        result.steering_curvature = steering_curvature;
        result_pub_->publish(result);

        // ── Build and publish visual ──────────────────────────────────────────
        if (visual_pub_->get_subscription_count() > 0) {
            auto vis = buildVisual(resized, row_logits, row_probs, all_peaks,
                                   road_present, road_prob, mean_conf,
                                   std::vector<float>{primaryOf(all_peaks[0]).conf,
                                                       primaryOf(all_peaks[1]).conf,
                                                       primaryOf(all_peaks[2]).conf},
                                   draw_fork,
                                   steering_curvature, lookahead_img_pt,
                                   final_desired_speed, adjusted_mean_conf);
            std_msgs::msg::Header hdr = msg->header;
            auto vis_msg = cv_bridge::CvImage(hdr, "bgr8", vis).toImageMsg();
            visual_pub_->publish(*vis_msg);
        }
    }

    // ── Visual overlay builder ────────────────────────────────────────────────
    cv::Mat buildVisual(const cv::Mat                        & src_bgr,
                        const std::vector<torch::Tensor>     & row_logits,
                        const std::vector<std::vector<float>>& row_probs,
                        const std::vector<std::vector<Peak>> & all_peaks,
                        bool   road_present,
                        float  road_prob,
                        float  mean_conf,
                        const std::vector<float>             & row_confs,
                        bool   draw_fork,
                        float  steering_curvature  = std::numeric_limits<float>::quiet_NaN(),
                        cv::Point2d lookahead_model = {-1, -1},
                        float  desired_speed        = 0.0f,
                        float  adjusted_mean_conf   = 0.0f)
    {
        // Upscale small images to min_display_height
        cv::Mat canvas = src_bgr.clone();
        if (canvas.rows < min_display_height_) {
            double up = static_cast<double>(min_display_height_) / canvas.rows;
            cv::resize(canvas, canvas,
                       cv::Size(static_cast<int>(canvas.cols*up), min_display_height_),
                       0, 0, cv::INTER_LINEAR);
        }
        int h = canvas.rows, w = canvas.cols;

        // Scale factors (reference = 480px tall)
        double s    = h / 480.0;
        int lw      = std::max(1, static_cast<int>(std::round(2*s)));
        int cr      = std::max(2, static_cast<int>(std::round(6*s)));
        int dot_r   = std::max(1, static_cast<int>(std::round(2*s)));
        double fs   = std::max(0.4, 0.7*s);
        double fs_l = std::max(0.4, 0.8*s);
        int lbl_h   = std::max(18, static_cast<int>(std::round(36*s)));
        int lbl_y   = std::max(12, static_cast<int>(std::round(26*s)));
        int border  = std::max(3,  static_cast<int>(std::round(8*s)));
        int dash    = std::max(4,  static_cast<int>(std::round(8*s)));

        // Bucket edge → pixel x
        auto bucketCx = [&](int r, int b) -> int {
            return static_cast<int>(
                (cfg_.bucket_edges[r][b] + cfg_.bucket_edges[r][b+1]) * 0.5 * w);
        };
        auto edgePx = [&](int r, int b) -> int {
            return static_cast<int>(cfg_.bucket_edges[r][b] * w);
        };

        // Guide dashed lines
        for (int r = 0; r < cfg_.n_rows; ++r) {
            int row_y = static_cast<int>(cfg_.row_fractions[r] * h);
            dashedHLine(canvas, row_y, _GUIDE, dash, lw);
        }

        // Debug sigmoid bars
        if (debug_mode_) {
            cv::Mat dbg = canvas.clone();
            int bar_max_h = std::max(2, h/8);
            for (int r = 0; r < cfg_.n_rows; ++r) {
                int row_y = static_cast<int>(cfg_.row_fractions[r] * h);
                for (int b = 0; b < cfg_.n_buckets; ++b) {
                    float p = row_probs[r][b];
                    if (p < 0.01f) continue;
                    int bar_h = std::max(1, static_cast<int>(p * bar_max_h));
                    int x0 = edgePx(r, b);
                    int x1 = edgePx(r, b+1);
                    int y0 = std::max(0, row_y - bar_h);
                    cv::rectangle(dbg, {x0, y0}, {x1, row_y}, cv::Scalar(80,220,0), cv::FILLED);
                }
            }
            cv::addWeighted(dbg, 0.45, canvas, 0.55, 0, canvas);
        }

        if (road_present) {
            // Translucent peak rectangles (all peaks when fork is confirmed)
            cv::Mat overlay = canvas.clone();
            int rect_hh = std::max(2, h/30);
            for (int r = 0; r < cfg_.n_rows; ++r) {
                int row_y = static_cast<int>(cfg_.row_fractions[r] * h);
                int n_pks = (draw_fork && r < 2) ? static_cast<int>(all_peaks[r].size()) : 1;
                for (int pi = 0; pi < n_pks; ++pi) {
                    // In fork mode use sorted (left→right) order; otherwise use primary
                    const Peak & pk = (n_pks > 1) ? all_peaks[r][pi] : primaryOf(all_peaks[r]);
                    int cx = static_cast<int>(pk.cx_frac * w);
                    int best_b = 0; double best_d = 1e9;
                    for (int b = 0; b < cfg_.n_buckets; ++b) {
                        double d = std::abs(bucketCx(r,b) - cx);
                        if (d < best_d) { best_d = d; best_b = b; }
                    }
                    int bx0 = edgePx(r, best_b);
                    int bx1 = edgePx(r, best_b+1);
                    cv::rectangle(overlay, {bx0, row_y-rect_hh}, {bx1, row_y+rect_hh},
                                  _PRED, cv::FILLED);
                }
            }
            cv::addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas);

            // Near point (r2): always use the primary peak
            cv::Point p_near(static_cast<int>(primaryOf(all_peaks[2]).cx_frac * w),
                              static_cast<int>(cfg_.row_fractions[2] * h));

            if (draw_fork && all_peaks[1].size() >= 2) {
                // Match r0 peaks to r1 branches (optimal 1-to-1 assignment)
                std::vector<const Peak*> r0_for_branch(2, nullptr);
                if (all_peaks[0].size() == 1) {
                    double d0 = std::abs(all_peaks[0][0].cx_frac - all_peaks[1][0].cx_frac);
                    double d1 = std::abs(all_peaks[0][0].cx_frac - all_peaks[1][1].cx_frac);
                    r0_for_branch[d0 <= d1 ? 0 : 1] = &all_peaks[0][0];
                } else if (all_peaks[0].size() >= 2) {
                    double a0 = all_peaks[0][0].cx_frac, a1 = all_peaks[0][1].cx_frac;
                    double b0 = all_peaks[1][0].cx_frac, b1 = all_peaks[1][1].cx_frac;
                    if (std::abs(b0-a1)+std::abs(b1-a0) < std::abs(b0-a0)+std::abs(b1-a1)) {
                        r0_for_branch[0] = &all_peaks[0][1];
                        r0_for_branch[1] = &all_peaks[0][0];
                    } else {
                        r0_for_branch[0] = &all_peaks[0][0];
                        r0_for_branch[1] = &all_peaks[0][1];
                    }
                }
                int sep_x = (static_cast<int>(all_peaks[1][0].cx_frac * w) +
                             static_cast<int>(all_peaks[1][1].cx_frac * w)) / 2;
                for (int bi = 0; bi < 2; ++bi) {
                    cv::Point p_mid(static_cast<int>(all_peaks[1][bi].cx_frac * w),
                                    static_cast<int>(cfg_.row_fractions[1] * h));
                    cv::Point p_far_val;
                    cv::Point * p_far_ptr = nullptr;
                    if (r0_for_branch[bi] && r0_for_branch[bi]->conf >= far_row_curve_thresh_) {
                        p_far_val = cv::Point(
                            static_cast<int>(r0_for_branch[bi]->cx_frac * w),
                            static_cast<int>(cfg_.row_fractions[0] * h));
                        p_far_ptr = &p_far_val;
                    }
                    int x_lo = (bi == 0) ? 0     : sep_x;
                    int x_hi = (bi == 0) ? sep_x : w - 1;
                    drawBranch(canvas, p_near, p_mid, p_far_ptr, x_lo, x_hi, lw);
                }
            } else {
                // Single branch: use primary peaks, not leftmost
                const Peak & r1_pk = primaryOf(all_peaks[1]);
                const Peak & r0_pk = primaryOf(all_peaks[0]);
                cv::Point p_mid(static_cast<int>(r1_pk.cx_frac * w),
                                 static_cast<int>(cfg_.row_fractions[1] * h));
                cv::Point p_far(static_cast<int>(r0_pk.cx_frac * w),
                                 static_cast<int>(cfg_.row_fractions[0] * h));
                cv::Point * p_far_ptr = (r0_pk.conf >= far_row_curve_thresh_) ? &p_far : nullptr;
                drawBranch(canvas, p_near, p_mid, p_far_ptr, 0, w-1, lw);
            }

            // Peak circles + confidence labels (all peaks when fork is confirmed)
            for (int r = 0; r < cfg_.n_rows; ++r) {
                int row_y = static_cast<int>(cfg_.row_fractions[r] * h);
                int n_pks = (draw_fork && r < 2) ? static_cast<int>(all_peaks[r].size()) : 1;
                for (int pi = 0; pi < n_pks; ++pi) {
                    const Peak & pk = (n_pks > 1) ? all_peaks[r][pi] : primaryOf(all_peaks[r]);
                    int cx     = static_cast<int>(pk.cx_frac * w);
                    float conf = pk.conf;
                    cv::circle(canvas, {cx, row_y}, cr,    _PRED,                  cv::FILLED, cv::LINE_AA);
                    cv::circle(canvas, {cx, row_y}, cr,    cv::Scalar(80,40,10), 1,            cv::LINE_AA);
                    cv::circle(canvas, {cx, row_y}, dot_r, cv::Scalar(255,255,255), cv::FILLED, cv::LINE_AA);
                    std::string txt = std::to_string(static_cast<int>(conf * 100)) + "%";
                    int tx = std::min(cx + std::max(2, static_cast<int>(4*s)), w - 40);
                    int ty = row_y - std::max(2, static_cast<int>(4*s));
                    cv::putText(canvas, txt, {tx,ty}, cv::FONT_HERSHEY_PLAIN, fs,
                                cv::Scalar(0,0,0), 2, cv::LINE_AA);
                    cv::putText(canvas, txt, {tx,ty}, cv::FONT_HERSHEY_PLAIN, fs,
                                _PRED,             1, cv::LINE_AA);
                }
            }
        }

        // Lookahead drive-to point (cyan circle)
        if (flat_world_loaded_ && road_present && lookahead_model.x >= 0) {
            double scale_x = static_cast<double>(w) / cfg_.image_width;
            double scale_y = static_cast<double>(h) / cfg_.image_height;
            cv::Point lpt(static_cast<int>(lookahead_model.x * scale_x),
                          static_cast<int>(lookahead_model.y * scale_y));
            if (lpt.x >= 0 && lpt.x < w && lpt.y >= 0 && lpt.y < h) {
                cv::circle(canvas, lpt, cr + 2, cv::Scalar(230, 200, 0),   2, cv::LINE_AA);  // cyan ring
                cv::circle(canvas, lpt, dot_r,  cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);
            }
        }

        // No-road red border
        if (!road_present) {
            canvas.rowRange(0, border)     = _RED;
            canvas.rowRange(h-border, h)   = _RED;
            canvas.colRange(0, border)     = _RED;
            canvas.colRange(w-border, w)   = _RED;
        }

        // Speed / confidence readout (bottom-right corner)
        {
            int    margin    = border + std::max(4, static_cast<int>(std::round(6*s)));
            int    sq_side   = std::max(8, static_cast<int>(std::round(14*s)));
            int    gap       = std::max(2, static_cast<int>(std::round(4*s)));
            double fs_hud    = std::max(0.5, 0.9*s);
            int    hud_thick = std::max(1, static_cast<int>(std::round(1.5*s)));

            char speed_buf[48], conf_buf[32];
            if (std::isnan(steering_curvature))
                std::snprintf(speed_buf, sizeof(speed_buf), "curv=n/a  speed=%.1f", desired_speed);
            else
                std::snprintf(speed_buf, sizeof(speed_buf), "curv=%.1f  speed=%.1f",
                              static_cast<double>(steering_curvature), desired_speed);
            std::snprintf(conf_buf,  sizeof(conf_buf),  "conf=%.1f",  adjusted_mean_conf);

            int baseline = 0;
            cv::Size speed_sz = cv::getTextSize(speed_buf, cv::FONT_HERSHEY_PLAIN, fs_hud, hud_thick, &baseline);
            cv::Size conf_sz  = cv::getTextSize(conf_buf,  cv::FONT_HERSHEY_PLAIN, fs_hud, hud_thick, &baseline);

            int conf_y  = h - margin;
            int speed_y = conf_y - std::max(sq_side, conf_sz.height) - gap;

            auto drawHudText = [&](const std::string & txt, const cv::Size & sz, int y) {
                cv::Point org(w - margin - sz.width, y);
                cv::putText(canvas, txt, org, cv::FONT_HERSHEY_PLAIN, fs_hud,
                            cv::Scalar(0,0,0), hud_thick+1, cv::LINE_AA);
                cv::putText(canvas, txt, org, cv::FONT_HERSHEY_PLAIN, fs_hud,
                            cv::Scalar(255,255,255), hud_thick, cv::LINE_AA);
                return org;
            };

            drawHudText(speed_buf, speed_sz, speed_y);
            cv::Point conf_org = drawHudText(conf_buf, conf_sz, conf_y);

            // Confidence tier: green > 0.7, yellow in [0.4, 0.7], red < 0.4
            cv::Scalar conf_color = (adjusted_mean_conf > 0.7f)  ? cv::Scalar(0,180,0)
                                   : (adjusted_mean_conf >= 0.4f) ? cv::Scalar(0,210,210)
                                                                   : cv::Scalar(0,0,210);

            int text_center_y = conf_org.y - conf_sz.height/2;
            int sq_right = conf_org.x - gap;
            int sq_left  = sq_right - sq_side;
            int sq_top   = text_center_y - sq_side/2;
            int sq_bottom= sq_top + sq_side;
            cv::rectangle(canvas, {sq_left, sq_top}, {sq_right, sq_bottom}, conf_color, cv::FILLED, cv::LINE_AA);
            cv::rectangle(canvas, {sq_left, sq_top}, {sq_right, sq_bottom}, cv::Scalar(0,0,0), 1, cv::LINE_AA);
        }

        // Status bar
        cv::Scalar bar_col = road_present ? cv::Scalar(30,140,30) : cv::Scalar(40,40,160);
        cv::Mat label_bar(lbl_h, w, CV_8UC3, bar_col);
        std::string label;
        if (road_present) {
            label = "ROAD " + std::to_string(static_cast<int>(road_prob*100)) + "%"
                  + "  conf:" + std::to_string(static_cast<int>(mean_conf*100)) + "%"
                  + "  [" + std::to_string(static_cast<int>(row_confs[0]*100))
                  + "/" + std::to_string(static_cast<int>(row_confs[1]*100))
                  + "/" + std::to_string(static_cast<int>(row_confs[2]*100)) + "%]";
            if (!std::isnan(steering_curvature)) {
                char buf[48];
                double kappa = static_cast<double>(steering_curvature);
                double alpha = std::atan(vehicle_wheelbase_m_ * kappa) * 180.0 / M_PI;
                if (std::abs(kappa) < 1e-3)
                    std::snprintf(buf, sizeof(buf), "  str: straight");
                else
                    std::snprintf(buf, sizeof(buf), "  str: R=%.1fm  α=%.1f°",
                                  1.0 / kappa, alpha);
                label += buf;
            }
        } else {
            label = "NO ROAD " + std::to_string(static_cast<int>((1-road_prob)*100)) + "%"
                  + "  conf:" + std::to_string(static_cast<int>(mean_conf*100)) + "%";
        }

        cv::putText(label_bar, label, {std::max(2,static_cast<int>(4*s)), lbl_y},
                    cv::FONT_HERSHEY_PLAIN, fs_l, cv::Scalar(255,255,255),
                    std::max(1,static_cast<int>(std::round(1.5*s))), cv::LINE_AA);

        cv::Mat out;
        cv::vconcat(label_bar, canvas, out);
        return out;
    }

    // ── Members ───────────────────────────────────────────────────────────────
    ModelConfig cfg_;
    torch::Device device_ = torch::kCPU;
    torch::jit::script::Module model_;
    torch::Tensor norm_mean_, norm_std_;

    double road_threshold_;
    double min_peak_conf_;
    double cluster_thresh_;
    double far_row_curve_thresh_;
    bool   debug_mode_;
    int    min_display_height_;
    int    fork_confirm_frames_;
    int    fork_frame_count_ = 0;

    // Cycle-rate tracking (smoothed Hz, logged at most once per second)
    std::chrono::steady_clock::time_point last_cycle_time_;
    bool   have_last_cycle_time_ = false;
    double cycle_hz_             = 0.0;

    // Temporal tracking
    std::array<double, 3>      prev_cx_frac_ = {-1.0, -1.0, -1.0};
    std::vector<std::vector<Peak>> prev_peaks_;
    float  road_momentum_ = 0.0f;
    float  last_valid_curvature_      = 0.0f;
    bool   have_last_valid_curvature_ = false;

    // Peak detection parameters
    float temporal_bias_weight_     = 0.1f;
    float temporal_proximity_range_ = 0.1f;
    float fork_hint_thresh_         = 0.55f;
    float fork_hint_proximity_      = 0.20f;
    float road_momentum_decay_      = 0.6f;
    float road_momentum_thresh_     = 0.25f;

    // Speed determination
    float very_high_confidence_value_        = 0.4f;
    float maximum_speed_                     = 2.0f;
    float maximum_bias_curvature_            = 0.1f;
    float maximum_curvature_speed_reduction_ = 0.5f;

    // Flat-world steering
    bool             flat_world_loaded_ = false;
    FlatWorldExtrinsic extr_;
    cv::Matx33d      K_model_;
    cv::Matx33d      R_cam_;
    double           lookahead_distance_m_ = 3.0;
    double           forward_speed_mps_    = 0.0;
    double           vehicle_wheelbase_m_  = 0.3;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<road_centerline::msg::CenterlineResult>::SharedPtr result_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr visual_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr steering_pub_;
    rclcpp::Publisher<control_interfaces::msg::ControlMsg>::SharedPtr control_pub_;
};

static const std::string DEFAULT_PARAMS_FILE =
    expandUserPath("~/ros2_ws/src/runtime_road_centerline_node/config/road_centerline_params.yaml");

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);

    // If the user didn't supply --params-file on the command line, load the
    // default config so the node always has sensible values out of the box.
    bool has_params_file = false;
    for (int i = 0; i < argc; ++i) {
        if (std::string(argv[i]) == "--params-file") {
            has_params_file = true;
            break;
        }
    }

    rclcpp::NodeOptions options;
    if (!has_params_file && std::filesystem::exists(DEFAULT_PARAMS_FILE)) {
        options.arguments({"--ros-args", "--params-file", DEFAULT_PARAMS_FILE});
    }

    rclcpp::spin(std::make_shared<RoadCenterlineNode>(options));
    rclcpp::shutdown();
    return 0;
}

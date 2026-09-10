# Published by road_centerline_node after each inference frame.
# All coordinates are normalised fractions of the model input image [0, 1].

std_msgs/Header header

# True when the road-present classifier fired and peak confidence passed threshold
bool road_present

# Classifier softmax probability for "road present"
float32 road_confidence

# Mean peak sigmoid confidence across all scan rows
float32 peak_confidence

# Per-row peak confidences  [r0=farthest, r1=middle, r2=nearest]
float32[3] row_confidences

# Centerline sample points, one per detected scan row.
# point.x  — normalised column  [0,1]  (0 = left edge, 1 = right edge)
# point.y  — normalised row     [0,1]  (0 = top,       1 = bottom)
# point.z  — peak confidence for this row
geometry_msgs/Point32[] points

# Desired steering curvature (m⁻¹) from flat-world pure-pursuit.
# Positive = turn left (positive lateral offset), negative = turn right.
# NaN when flat-world calibration is not loaded or road is not present.
float32 steering_curvature

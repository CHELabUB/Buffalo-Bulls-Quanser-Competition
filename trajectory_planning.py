# code written by Haosong, for trajectory planning

# native import
import os
import sys

# from docker
import cv2 
import numpy as np
import matplotlib.pyplot as plt


# every callback, there will be two images, rgb and the depth
# in order to know where the line is, and plan a control, we have to know the
# relative position between the way points w.r.t car.

#### load the images collected from manual drive mode, it requires both rgb and the depth image
rgb_path = "captured_images/color_20260225_170550_653720.png"
depth_path = "captured_images/depth_raw_20260225_170550_663659.png"


# K_rgb: inrinsic matrix of rgb
# K_depth: intrinsic matrix of depth
# T_rgb2depth: transformation from rgb to depth

K_rgb = np.array([
    [455.2, 0.0, 308.53],
    [0.0, 459.43, 213.56],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

K_depth = np.array([
    [385.6, 0.0, 321.9],
    [0.0, 385.6, 237.3],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

T_rgb_to_depth = np.array([
    [1.0, 0.004008, 0.0001655, -0.01474],
    [-0.004007, 1.0, -0.003435, -0.0004152],
    [-0.0001792, 0.003434, 1.0, -0.0002451],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float32)

# quick overlay check
img_bgr = cv2.imread(rgb_path)
if img_bgr is None:
    raise FileNotFoundError(f"Could not read image: {rgb_path}")

g_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
hsv_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# find the road first
lower = np.array([0, 0, 0])
upper = np.array([180, 255, 100])  # V < 100
mask = cv2.inRange(hsv_rgb, lower, upper)

h, w = mask.shape

# filter out bottom 30% pixels (ego-car area)
cut_row = int(h)
mask[cut_row:h, :] = 0

boundary_img = img_bgr.copy()
boundary_mask = np.zeros_like(mask)
boundary_thickness = 2

# for every row, pick the outmost road pixel from mask (rightmost non-zero column)
boundary_points = []

for y in range(h):
    cols = np.where(mask[y] > 0)[0]
    if cols.size > 0:
        x = int(cols[-1])
        boundary_points.append((x, y))

if len(boundary_points) > 1:
    pts = np.array(boundary_points, dtype=np.int32)
    cv2.polylines(boundary_img, [pts], isClosed=False, color=(0, 255, 0), thickness=boundary_thickness)
    cv2.polylines(boundary_mask, [pts], isClosed=False, color=255, thickness=boundary_thickness)
else:
    pass
    

# convert the boundary points from rgb to depth to estimate the track location w.r.t the car. 
# this to be verified
if len(boundary_points) > 1:
    print(pts.shape)
else:
    print("No boundary points found")

img_depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
if img_depth_raw is None:
    raise FileNotFoundError(f"Could not read depth image: {depth_path}")

# use 8-bit depth only for Canny/visualization
depth_8u = cv2.normalize(img_depth_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

depth_blur = cv2.GaussianBlur(depth_8u, (5,5), 0)
depth_edges = cv2.Canny(depth_blur, 0, 40)

# apply the same ego-car exclusion on depth edges
depth_h, depth_w = depth_edges.shape
depth_cut_row = int(depth_h * 0.7)
depth_edges[depth_cut_row:depth_h, :] = 0

depth_contours, _ = cv2.findContours(depth_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

depth_bgr = cv2.cvtColor(depth_8u, cv2.COLOR_GRAY2BGR)
contour_layer = np.zeros_like(depth_bgr)
cv2.drawContours(contour_layer, depth_contours, -1, (0, 255, 0), 2)
vis = cv2.addWeighted(depth_bgr, 1.0, contour_layer, 1.0, 0)

# pick bottom-most green contour pixel and mark it in red
green_pixels = np.where(contour_layer[:, :, 1] > 0)
if green_pixels[0].size > 0:
    y_bottom = int(np.max(green_pixels[0]))
    x_candidates = green_pixels[1][green_pixels[0] == y_bottom]
    x_bottom = int(np.median(x_candidates))
    cv2.circle(depth_bgr, (x_bottom, y_bottom), 5, (0, 0, 255), -1)
    cv2.circle(vis, (x_bottom, y_bottom), 5, (0, 0, 255), -1)
    print(f"Bottom green pixel: (x={x_bottom}, y={y_bottom})")

    depth_stop_bottom = float(img_depth_raw[y_bottom, x_bottom])
    print(f"raw depth at bottom green pixel: {depth_stop_bottom}")

    f_depth = K_depth[1, 1]  # focal length in pixels in y (f_y)
    z_stop_bottom = depth_stop_bottom / 10000
    h_d435 = (z_stop_bottom * (y_bottom - K_depth[1, 2])) / f_depth
    print(f"Estimated camera height: {h_d435:.2f} meters")
else:
    print("No green contour pixels found")

# Note: I don't think the depth unit if correct, need to confirm?
# I assume the height is around 0.25 m, roughly 25cm ? this is higher than the ones from quanser document

# pack coordinate frame conversion as a functoin 
# this is based on the assumption that the lines are on the ground. 
def p2c(u, v, K, camera_height):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    z = (camera_height * fy) / (v - cy)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z

def c2p_ground(x, z, K, camera_height):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    x = np.asarray(x)
    z = np.asarray(z)

    z_safe = np.where(np.abs(z) < 1e-6, np.nan, z)
    u = fx * x / z_safe + cx
    v = fy * camera_height / z_safe + cy
    return u, v

line_detected = []

tune = True
height_adjust  = 0.15
if tune:
    h_d435 = height_adjust
else:
    h_d435 = h_d435

for (x, y) in boundary_points:
    x_c, y_c, z_c = p2c(x, y, K_rgb, h_d435)
    line_detected.append((x_c, y_c, z_c))
    print(f'the pos right now is x:{x_c} y:{z_c}')

cv2.imshow("Road Mask", mask)
cv2.imshow("Boundary Line Only", boundary_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save detected line points to file (workspace-safe path with fallback)
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
output_path = os.path.join(output_dir, "line_detected.txt")

try:
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for (x_c, y_c, z_c) in line_detected:
            f.write(f"{x_c:.4f}, {y_c:.4f}, {z_c:.4f}\n")
    print(f"Saved line points to: {output_path}")
except PermissionError:
    fallback_path = "/tmp/line_detected.txt"
    with open(fallback_path, "w") as f:
        for (x_c, y_c, z_c) in line_detected:
            f.write(f"{x_c:.4f}, {y_c:.4f}, {z_c:.4f}\n")
    print(f"Permission denied in workspace path. Saved to: {fallback_path}")


# please note that when shifting to the vehicle frame, it should be x left/right,  y forward/backward
# there are also lines missiing from the horizontal line, need a line fitting

# method 1: using slope threshold, not goodd for filtering
def detect_line_or_curve(x, z, angle_threshold_deg=5, poly_degree=2):

    x = np.asarray(x)
    z = np.asarray(z)

    dx = np.diff(x)
    dz = np.diff(z)

    theta = np.arctan2(dz, dx)
    theta = np.unwrap(theta)

    mean_theta = np.mean(theta)
    std_theta = np.std(theta)

    threshold = np.deg2rad(angle_threshold_deg)

    if std_theta < threshold:

        points = np.vstack((x, z)).T
        mean = points.mean(axis=0)

        U, S, Vt = np.linalg.svd(points - mean)
        direction = Vt[0]  # dominant direction (unit vector)
        projected = []
        for p in points:
            length = np.dot(p - mean, direction)
            p_proj = mean + length * direction
            projected.append(p_proj)

        projected = np.array(projected)

        return "line", projected[:, 0], projected[:, 1]

    else:

        coeffs = np.polyfit(x, z, poly_degree)
        z_fitted = np.polyval(coeffs, x)

        return "curve", x, z_fitted

# method 2: sensitive to the outliers, and the sequence is reversed. 
def line_alignment(x, z, angle_threshold_deg=10):

    x = np.asarray(x).copy()
    z = np.asarray(z).copy()

    # ---- Sort by z ----
    indices = np.argsort(z)
    x_sorted = x[indices].copy()
    z_sorted = z[indices].copy()

    n = len(x_sorted)
    if n < 3:
        return x_sorted, z_sorted

    threshold = np.deg2rad(angle_threshold_deg)

    # ---- First direction ----
    dx0 = x_sorted[1] - x_sorted[0]
    dz0 = z_sorted[1] - z_sorted[0]
    prev_theta = np.arctan2(dz0, dx0)

    for i in range(2, n):

        dx = x_sorted[i] - x_sorted[i-1]
        dz = z_sorted[i] - z_sorted[i-1]

        theta = np.arctan2(dz, dx)

        dtheta = np.abs(np.unwrap([prev_theta, theta])[1] - prev_theta)

        if dtheta > threshold:

            # project current point along previous direction
            step_length = np.sqrt(dx**2 + dz**2)

            x_sorted[i] = x_sorted[i-1] + step_length * np.cos(prev_theta)
            z_sorted[i] = z_sorted[i-1] + step_length * np.sin(prev_theta)

        else:
            prev_theta = theta

    return x_sorted, z_sorted

# method 3: so far the best fit.
def robust_boundary_adjustment(x, z,
                               keep_ratio=0.8,
                               curve_degree=2,
                               vertical_ratio_threshold=3.0):

    x = np.asarray(x)
    z = np.asarray(z)

    # variance in both lateral and longitudinal directions
    var_x = np.var(x)
    var_z = np.var(z)

    # checking which direction thhe outline is moving
    # this is for determine the line or the turn!
    vertical_dominant = var_z > vertical_ratio_threshold * var_x


    n = len(x)
    n_keep = int(n * keep_ratio)
    if vertical_dominant:
        # Work in x-space (since x nearly constant)

        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]

        best_width = np.inf
        best_start = 0

        for i in range(n - n_keep):
            width = x_sorted[i + n_keep] - x_sorted[i]
            if width < best_width:
                best_width = width
                best_start = i

        x_min = x_sorted[best_start]
        x_max = x_sorted[best_start + n_keep]

        mask = (x >= x_min) & (x <= x_max)

    else:
        # Work in z-space
        sorted_idx = np.argsort(z)
        z_sorted = z[sorted_idx]

        best_width = np.inf
        best_start = 0

        for i in range(n - n_keep):
            width = z_sorted[i + n_keep] - z_sorted[i]
            if width < best_width:
                best_width = width
                best_start = i

        z_min = z_sorted[best_start]
        z_max = z_sorted[best_start + n_keep]

        mask = (z >= z_min) & (z <= z_max)

    x_inliers = x[mask]
    z_inliers = z[mask]

    ##### fit the line and the curve
    if vertical_dominant:
        # Fit x = f(z)
        coeffs = np.polyfit(z_inliers, x_inliers, 1)
        x_fit = np.polyval(coeffs, z)
        return "vertical_line", x_fit, z

    else:
        dx = np.diff(x_inliers)
        dz = np.diff(z_inliers)
        theta = np.unwrap(np.arctan2(dz, dx))
        std_theta = np.std(theta)

        if std_theta < np.deg2rad(5):
            # line
            coeffs = np.polyfit(x_inliers, z_inliers, 1)
            z_fit = np.polyval(coeffs, x)
            return "line", x, z_fit
        else:
            # curve
            coeffs = np.polyfit(x_inliers, z_inliers, curve_degree)
            z_fit = np.polyval(coeffs, x)
            return "curve", x, z_fit

if len(line_detected) > 3:
    xy = np.array(line_detected, dtype=np.float32)
    x_cam = xy[:, 0]
    z_cam = xy[:, 2]
    # print(x_cam, z_cam)
    # 
    _,x_filtered, z_filtered = robust_boundary_adjustment(x_cam, z_cam,
                                                       keep_ratio=0.8,
                                                       curve_degree=2,
                                                       vertical_ratio_threshold=3.0)
    print(f"Filtered points:\nX: {x_filtered}\nZ: {z_filtered}")

    fig = plt.figure(figsize=(7, 6))
    plt.scatter(x_cam, z_cam, c='tab:gray', s=14, alpha=0.8, label='Original (x, z)')
    plt.plot(x_filtered, z_filtered, c='tab:blue', linewidth=2.2)
    plt.scatter(0, 0, c='r', s=40, label='Camera')
    plt.title('Original vs Filtered')
    plt.xlabel('X (m) [right +]')
    plt.ylabel('Z (m) [forward +]')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()

    comparison_plot_path = os.path.join(output_dir, "fitting_comparison.png")
    fig.savefig(comparison_plot_path, dpi=180)
    print(f"Saved comparison plot to: {comparison_plot_path}")

    # map filtered (x, z) back to RGB image pixels
    u_filtered, v_filtered = c2p_ground(x_filtered, z_filtered, K_rgb, h_d435)

    valid = (
        np.isfinite(u_filtered)
        & np.isfinite(v_filtered)
        & (u_filtered >= 0)
        & (u_filtered < img_bgr.shape[1])
        & (v_filtered >= 0)
        & (v_filtered < img_bgr.shape[0])
    )

    uv_points = np.column_stack((u_filtered[valid], v_filtered[valid])).astype(np.int32)

    reproj_img = img_bgr.copy()
    for (u_i, v_i) in uv_points:
        cv2.circle(reproj_img, (u_i, v_i), 2, (255, 0, 0), -1)

    if len(uv_points) > 1:
        cv2.polylines(reproj_img, [uv_points], isClosed=False, color=(255, 0, 0), thickness=2)

    reproj_path = os.path.join(output_dir, "filtered_reprojected_pixels.png")
    cv2.imwrite(reproj_path, reproj_img)
    print(f"Saved reprojected filtered pixels to: {reproj_path}")

    plt.figure(figsize=(8, 5))
    plt.imshow(cv2.cvtColor(reproj_img, cv2.COLOR_BGR2RGB))
    plt.title('Filtered Curve Reprojected to RGB Pixels')
    plt.axis('off')
    plt.show()
else:
    print("Not enough valid points to compare filtered/unfiltered curves.")


# cv2.imshow("Depth Original (8-bit)", depth_bgr)
# cv2.imshow("Depth Edges", depth_edges)
# cv2.imshow("Depth + Green Contours", vis)



# cv2.imshow("Original", img_bgr)
# cv2.imshow("Road Mask", mask)
# cv2.imshow("Boundary Line Only", boundary_mask)

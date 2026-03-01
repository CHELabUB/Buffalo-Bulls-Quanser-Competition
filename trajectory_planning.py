# native import
import os
import sys
from time import time

# from docker
import cv2 
import numpy as np
import matplotlib.pyplot as plt


class TrajectoryPlanner:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.saved_depth_count = 0
        self.plan_headway = 0.5 # m
        self.tune = True
        self.height_adjust  = 0.15
        self.d435_height = 0.21
        self.half_road_width = 0.5/4
        self.K_rgb = np.array([
                        [455.2, 0.0, 308.53],
                        [0.0, 459.43, 213.56],
                        [0.0, 0.0, 1.0],], dtype=np.float32)
        self.vertical_ratio_threshold = 3.0
        self.keep_ratio = 0.8
        self.curve_degree = 2
        
        
    def line_detect(self, image):
        # detecting both line is possible, but added up the complexity of tunning
        # here I try to detect right most boundary of the road then minus the half of the road width

        g_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # road color is darker, this aims to filter out the road area, then based on road area, detect the bonudary line 
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 100])  # V < 100
        mask = cv2.inRange(hsv_rgb, lower, upper)
        h, w = mask.shape

        # this step aims to detect the outer line the road. 
        boundary_points = []
        for y in range(h):
            cols = np.where(mask[y] > 0)[0]
            if cols.size > 0:
                x = int(cols[-1])
                boundary_points.append((x, y))

        pts = np.array(boundary_points, dtype=np.float32)

        line_detected = []
        if self.tune:
            h_d435 = self.height_adjust
        else:
            h_d435 = self.d435_height

        for (x, y) in boundary_points:
            x_c, y_c, z_c = self.p2c(x, y, h_d435)
            line_detected.append((x_c, y_c, z_c))

        # in camera frame, x is lateral, and z is longitudinal
        xz = np.array(line_detected, dtype=np.float32)
        x_cam = xz[:, 0]
        z_cam = xz[:, 2]
        var_x = np.var(x_cam)
        var_z = np.var(z_cam)
        vertical_dominant = var_z > self.vertical_ratio_threshold * var_x
        n = len(x_cam)
        n_keep = int(n * self.keep_ratio)
        if vertical_dominant:
            # Work in x-space (since x nearly constant)
            sorted_idx = np.argsort(x_cam)
            x_sorted = x_cam[sorted_idx]

            best_width = np.inf
            best_start = 0

            for i in range(n - n_keep):
                width = x_sorted[i + n_keep] - x_sorted[i]
                if width < best_width:
                    best_width = width
                    best_start = i

            x_min = x_sorted[best_start]
            x_max = x_sorted[best_start + n_keep]

            outline_mask = (x_cam >= x_min) & (x_cam <= x_max)

        else:
            # Work in z-space
            sorted_idx = np.argsort(z_cam)
            z_sorted = z_cam[sorted_idx]

            best_width = np.inf
            best_start = 0

            for i in range(n - n_keep):
                width = z_sorted[i + n_keep] - z_sorted[i]
                if width < best_width:
                    best_width = width
                    best_start = i

            z_min = z_sorted[best_start]
            z_max = z_sorted[best_start + n_keep]

            outline_mask = (z_cam >= z_min) & (z_cam <= z_max)

        x_inliers = x_cam[outline_mask]
        z_inliers = z_cam[outline_mask]

        ##### fit the line and the curve
        if vertical_dominant:
            # Fit x = f(z)
            coeffs = np.polyfit(z_inliers, x_inliers, 1)
            x_fit = np.polyval(coeffs, z_inliers)
            
            x_out_adjusted = x_fit 
            z_out_adjusted = z_inliers

        else:
            dx = np.diff(x_inliers)
            dz = np.diff(z_inliers)
            theta = np.unwrap(np.arctan2(dz, dx))
            std_theta = np.std(theta)

            if std_theta < np.deg2rad(5):
                # line
                coeffs = np.polyfit(x_inliers, z_inliers, 1)
                z_fit = np.polyval(coeffs, x_inliers)
                x_out_adjusted = x_inliers
                z_out_adjusted = z_fit
            else:
                # curve
                coeffs = np.polyfit(x_inliers, z_inliers, 2)
                z_fit = np.polyval(coeffs, x_inliers)
                x_out_adjusted = x_inliers
                z_out_adjusted = z_fit

        x_target = x_out_adjusted - self.half_road_width
        z_target = z_out_adjusted + self.plan_headway
        return x_target, z_target

    def p2c(self, u, v, camera_height):
        # from pixels to camera coordinate system, reverse pinhole camera theory

        fx = self.K_rgb[0, 0]
        fy = self.K_rgb[1, 1]
        cx = self.K_rgb[0, 2]
        cy = self.K_rgb[1, 2]
        z = (camera_height * fy) / (v - cy)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return x, y, z

    def c2p_ground(self, x, z, camera_height):
        # map the 3D cam coord back to 2D pixel coords
        fx = self.K_rgb[0, 0]
        fy = self.K_rgb[1, 1]
        cx = self.K_rgb[0, 2]
        cy = self.K_rgb[1, 2]

        x = np.asarray(x)
        z = np.asarray(z)

        z_safe = np.where(np.abs(z) < 1e-6, np.nan, z)
        u = fx * x / z_safe + cx
        v = fy * camera_height / z_safe + cy
        return u, v



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
   
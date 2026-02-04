//! Plane representation and fitting.
//!
//! Provides plane detection using RANSAC and least-squares fitting.

use nalgebra::{Matrix3, Point3, Vector3};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// A plane in 3D space represented by ax + by + cz + d = 0.
/// The normal vector (a, b, c) is normalized.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plane {
    /// Normal vector (normalized)
    pub normal: Vector3<f64>,
    /// Distance from origin (d in ax + by + cz + d = 0)
    pub d: f64,
    /// Indices of inlier points (if available)
    #[serde(skip)]
    pub inliers: Vec<usize>,
    /// Root mean square error of fit
    pub rmse: f64,
}

impl Plane {
    /// Create a new plane from normal and distance.
    pub fn new(normal: Vector3<f64>, d: f64) -> Self {
        let normal = normal.normalize();
        Self {
            normal,
            d,
            inliers: Vec::new(),
            rmse: 0.0,
        }
    }

    /// Create a plane from a point and normal.
    pub fn from_point_and_normal(point: &Point3<f64>, normal: Vector3<f64>) -> Self {
        let normal = normal.normalize();
        let d = -normal.dot(&point.coords);
        Self {
            normal,
            d,
            inliers: Vec::new(),
            rmse: 0.0,
        }
    }

    /// Create a plane from three points.
    pub fn from_three_points(p1: &Point3<f64>, p2: &Point3<f64>, p3: &Point3<f64>) -> Option<Self> {
        let v1 = p2 - p1;
        let v2 = p3 - p1;
        let normal = v1.cross(&v2);

        if normal.norm() < 1e-10 {
            // Points are collinear
            return None;
        }

        Some(Self::from_point_and_normal(p1, normal))
    }

    /// Signed distance from a point to the plane.
    pub fn signed_distance(&self, point: &Point3<f64>) -> f64 {
        self.normal.dot(&point.coords) + self.d
    }

    /// Absolute distance from a point to the plane.
    pub fn distance(&self, point: &Point3<f64>) -> f64 {
        self.signed_distance(point).abs()
    }

    /// Project a point onto the plane.
    pub fn project(&self, point: &Point3<f64>) -> Point3<f64> {
        let dist = self.signed_distance(point);
        Point3::from(point.coords - dist * self.normal)
    }

    /// Get a point on the plane (closest point to origin).
    pub fn point_on_plane(&self) -> Point3<f64> {
        Point3::from(-self.d * self.normal)
    }

    /// Compute slope angle in degrees (angle from horizontal).
    pub fn slope_degrees(&self) -> f64 {
        let vertical = Vector3::new(0.0, 0.0, 1.0);
        let cos_angle = self.normal.dot(&vertical).abs();
        (90.0 - cos_angle.acos().to_degrees()).abs()
    }

    /// Compute azimuth angle in degrees (direction of steepest descent).
    pub fn azimuth_degrees(&self) -> f64 {
        let horizontal = Vector3::new(self.normal.x, self.normal.y, 0.0);
        if horizontal.norm() < 1e-10 {
            return 0.0; // Horizontal plane
        }
        let horizontal = horizontal.normalize();

        let mut azimuth = horizontal.y.atan2(horizontal.x).to_degrees();
        if azimuth < 0.0 {
            azimuth += 360.0;
        }
        azimuth
    }

    /// Fit a plane to points using least squares (SVD).
    pub fn fit_least_squares(points: &[Point3<f64>]) -> Option<Self> {
        if points.len() < 3 {
            return None;
        }

        // Compute centroid
        let n = points.len() as f64;
        let centroid: Vector3<f64> = points.iter().fold(Vector3::zeros(), |acc, p| acc + p.coords) / n;

        // Build covariance matrix
        let mut cov = Matrix3::zeros();
        for p in points {
            let d = p.coords - centroid;
            cov += d * d.transpose();
        }

        // SVD to find normal (smallest singular value eigenvector)
        let svd = cov.svd(true, true);
        let v = svd.v_t?;

        // Normal is the last row of V^T (corresponding to smallest singular value)
        let normal = Vector3::new(v[(2, 0)], v[(2, 1)], v[(2, 2)]);
        let centroid_point = Point3::from(centroid);

        let mut plane = Self::from_point_and_normal(&centroid_point, normal);

        // Compute RMSE
        let sum_sq: f64 = points.iter().map(|p| plane.distance(p).powi(2)).sum();
        plane.rmse = (sum_sq / n).sqrt();

        Some(plane)
    }
}

/// Configuration for RANSAC plane detection.
#[derive(Debug, Clone)]
pub struct RansacConfig {
    /// Maximum distance from plane to be considered an inlier
    pub epsilon: f64,
    /// Number of neighbors to consider for local plane fitting
    pub k: usize,
    /// Minimum number of points to form a valid plane
    pub min_points: usize,
    /// Maximum number of RANSAC iterations
    pub max_iterations: usize,
    /// Threshold for classifying a plane as vertical (wall) based on |n·z|
    pub metrics_is_wall_threshold: f64,
    /// Threshold for classifying a plane as horizontal based on |n·z|
    pub metrics_is_horizontal_threshold: f64,
    /// Angle threshold (degrees) for merging similar planes
    pub merge_angle_degrees: f64,
    /// Distance threshold for merging similar planes
    pub merge_distance: f64,
    /// Distance threshold for merging horizontal planes
    pub merge_distance_horizontal: f64,
    /// Minimum inlier overlap ratio for merging planes
    pub merge_inlier_overlap: f64,
    /// Minimum inlier ratio for accepting a plane (relative to remaining points)
    pub min_inlier_ratio: f64,
    /// Max inliers to treat a plane as "small" for merging
    pub small_plane_max_inliers: usize,
    /// Neighbor distance threshold for merging small planes
    pub merge_neighbor_distance: f64,
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.3,
            k: 15,
            min_points: 15,
            max_iterations: 1000,
            metrics_is_wall_threshold: 0.3,
            metrics_is_horizontal_threshold: 0.995,
            merge_angle_degrees: 7.5,
            merge_distance: 0.5,
            merge_distance_horizontal: 1.5,
            merge_inlier_overlap: 0.6,
            min_inlier_ratio: 0.04,
            small_plane_max_inliers: 60,
            merge_neighbor_distance: 2.0,
        }
    }
}

/// Detect planes in a point cloud using RANSAC.
pub struct PlaneDetector {
    config: RansacConfig,
}

impl PlaneDetector {
    pub fn new(config: RansacConfig) -> Self {
        Self { config }
    }

    /// Detect a single best plane using RANSAC.
    pub fn detect_single(&self, points: &[Point3<f64>]) -> Option<Plane> {
        if points.len() < self.config.min_points {
            return None;
        }

        let mut rng = rand::thread_rng();
        let mut best_plane: Option<Plane> = None;
        let mut best_inlier_count = 0;

        for _ in 0..self.config.max_iterations {
            // Sample 3 random points
            let indices: Vec<usize> = (0..points.len()).collect();
            let sample: Vec<usize> = indices.choose_multiple(&mut rng, 3).cloned().collect();

            if sample.len() < 3 {
                continue;
            }

            // Fit plane to sample
            let candidate = match Plane::from_three_points(
                &points[sample[0]],
                &points[sample[1]],
                &points[sample[2]],
            ) {
                Some(p) => p,
                None => continue,
            };

            // Count inliers
            let inliers: Vec<usize> = points
                .iter()
                .enumerate()
                .filter(|(_, p)| candidate.distance(p) < self.config.epsilon)
                .map(|(i, _)| i)
                .collect();

            if inliers.len() > best_inlier_count && inliers.len() >= self.config.min_points {
                best_inlier_count = inliers.len();

                // Refit plane using all inliers
                let inlier_points: Vec<Point3<f64>> = inliers.iter().map(|&i| points[i]).collect();
                if let Some(mut refined) = Plane::fit_least_squares(&inlier_points) {
                    refined.inliers = inliers;
                    best_plane = Some(refined);
                }
            }
        }

        best_plane
    }

    /// Detect multiple planes using iterative RANSAC.
    pub fn detect_multiple(&self, points: &[Point3<f64>], max_planes: usize) -> Vec<Plane> {
        let mut planes = Vec::new();
        let mut remaining_mask = vec![true; points.len()];
        let mut remaining_count = points.len();

        while planes.len() < max_planes && remaining_count >= self.config.min_points {
            // Extract remaining points
            let remaining_indices: Vec<usize> = remaining_mask
                .iter()
                .enumerate()
                .filter(|(_, &active)| active)
                .map(|(i, _)| i)
                .collect();

            let remaining_points: Vec<Point3<f64>> =
                remaining_indices.iter().map(|&i| points[i]).collect();

            // Detect single plane
            if let Some(mut plane) = self.detect_single(&remaining_points) {
                // Filter out near-vertical planes (walls)
                let horizontality = plane.normal.dot(&Vector3::new(0.0, 0.0, 1.0)).abs();
                if horizontality < self.config.metrics_is_wall_threshold {
                    for &idx in &plane.inliers {
                        let original_idx = remaining_indices[idx];
                        remaining_mask[original_idx] = false;
                        remaining_count -= 1;
                    }
                    continue;
                }

                // Map inlier indices back to original
                plane.inliers = plane
                    .inliers
                    .iter()
                    .map(|&i| remaining_indices[i])
                    .collect();

                // Remove inliers from remaining points
                for &idx in &plane.inliers {
                    remaining_mask[idx] = false;
                    remaining_count -= 1;
                }

                // Keep only planes with enough inliers and ratio
                let inlier_ratio = plane.inliers.len() as f64 / remaining_points.len() as f64;
                if plane.inliers.len() >= self.config.min_points
                    && inlier_ratio >= self.config.min_inlier_ratio
                {
                    planes.push(plane);
                }
            } else {
                break;
            }
        }

        merge_similar_planes(&mut planes, points, &self.config)
    }
}

fn merge_similar_planes(planes: &mut Vec<Plane>, points: &[Point3<f64>], config: &RansacConfig) -> Vec<Plane> {
    let mut merged: Vec<Plane> = Vec::new();
    let angle_thresh = config.merge_angle_degrees.to_radians().cos();
    for plane in planes.drain(..) {
        let mut merged_into = false;
        for existing in merged.iter_mut() {
            let dot = existing.normal.dot(&plane.normal).abs();
            let dist = (existing.d - plane.d).abs();
            let horiz_existing = existing.normal.dot(&Vector3::new(0.0, 0.0, 1.0)).abs();
            let horiz_plane = plane.normal.dot(&Vector3::new(0.0, 0.0, 1.0)).abs();
            let is_horizontal = horiz_existing >= config.metrics_is_horizontal_threshold
                && horiz_plane >= config.metrics_is_horizontal_threshold;
            let dist_thresh = if is_horizontal {
                config.merge_distance_horizontal
            } else {
                config.merge_distance
            };
            let centroid_existing = plane_inlier_centroid(existing, points);
            let centroid_plane = plane_inlier_centroid(&plane, points);
            let allow_small_merge = (existing.inliers.len() <= config.small_plane_max_inliers
                || plane.inliers.len() <= config.small_plane_max_inliers)
                && (centroid_existing - centroid_plane).norm() <= config.merge_neighbor_distance;
            if !allow_small_merge && !inlier_overlap_ok(existing, &plane, config.merge_inlier_overlap) {
                continue;
            }
            if dot >= angle_thresh && dist <= dist_thresh {
                // merge inliers and refit
                let mut inliers = existing.inliers.clone();
                inliers.extend_from_slice(&plane.inliers);
                inliers.sort();
                inliers.dedup();
                let inlier_points: Vec<Point3<f64>> = inliers.iter().map(|&i| points[i]).collect();
                if let Some(mut refit) = Plane::fit_least_squares(&inlier_points) {
                    refit.inliers = inliers;
                    *existing = refit;
                }
                merged_into = true;
                break;
            }
        }
        if !merged_into {
            merged.push(plane);
        }
    }
    merged
}

fn plane_inlier_centroid(plane: &Plane, points: &[Point3<f64>]) -> Point3<f64> {
    if plane.inliers.is_empty() {
        return plane.point_on_plane();
    }
    let mut sum = Vector3::zeros();
    for &idx in &plane.inliers {
        sum += points[idx].coords;
    }
    let n = plane.inliers.len() as f64;
    Point3::from(sum / n)
}

fn inlier_overlap_ok(a: &Plane, b: &Plane, min_overlap: f64) -> bool {
    if a.inliers.is_empty() || b.inliers.is_empty() {
        return true;
    }
    let mut i = 0usize;
    let mut j = 0usize;
    let mut overlap = 0usize;
    let mut a_sorted = a.inliers.clone();
    let mut b_sorted = b.inliers.clone();
    a_sorted.sort();
    b_sorted.sort();
    while i < a_sorted.len() && j < b_sorted.len() {
        if a_sorted[i] == b_sorted[j] {
            overlap += 1;
            i += 1;
            j += 1;
        } else if a_sorted[i] < b_sorted[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    let denom = a_sorted.len().min(b_sorted.len()) as f64;
    if denom == 0.0 {
        true
    } else {
        (overlap as f64 / denom) >= min_overlap
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_plane_from_three_points() {
        let p1 = Point3::new(0.0, 0.0, 0.0);
        let p2 = Point3::new(1.0, 0.0, 0.0);
        let p3 = Point3::new(0.0, 1.0, 0.0);

        let plane = Plane::from_three_points(&p1, &p2, &p3).unwrap();

        // Should be a horizontal plane (normal pointing up or down)
        assert_relative_eq!(plane.normal.z.abs(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(plane.d, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_plane_distance() {
        let plane = Plane::new(Vector3::new(0.0, 0.0, 1.0), 0.0); // z = 0 plane

        let point = Point3::new(0.0, 0.0, 5.0);
        assert_relative_eq!(plane.distance(&point), 5.0, epsilon = 1e-10);

        let point2 = Point3::new(1.0, 2.0, 0.0);
        assert_relative_eq!(plane.distance(&point2), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_plane_fit_least_squares() {
        // Create points on z = 1 plane with some noise
        let points: Vec<Point3<f64>> = vec![
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(0.5, 0.5, 1.0),
        ];

        let plane = Plane::fit_least_squares(&points).unwrap();

        // Normal should be approximately (0, 0, 1)
        assert_relative_eq!(plane.normal.z.abs(), 1.0, epsilon = 1e-10);
        // d should be approximately -1 or 1 depending on normal direction
        assert_relative_eq!(plane.d.abs(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_plane_slope_and_azimuth() {
        // Horizontal plane
        let horizontal = Plane::new(Vector3::new(0.0, 0.0, 1.0), 0.0);
        assert_relative_eq!(horizontal.slope_degrees(), 0.0, epsilon = 1e-10);

        // 45-degree slope facing north
        let sloped = Plane::new(Vector3::new(0.0, 1.0, 1.0).normalize(), 0.0);
        assert_relative_eq!(sloped.slope_degrees(), 45.0, epsilon = 1e-6);
    }
}

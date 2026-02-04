//! Point cloud segmentation.
//!
//! Provides algorithms for segmenting point clouds into regions
//! based on geometric properties.

use crate::plane::Plane;
use crate::point_cloud::PointCloud;
use kiddo::KdTree;
use std::collections::VecDeque;

/// Point segmenter using region growing.
pub struct PointSegmenter {
    /// Maximum distance between points to be considered neighbors
    pub neighbor_distance: f64,
    /// Maximum distance from plane to be considered part of segment
    pub plane_tolerance: f64,
    /// Minimum points in a segment
    pub min_segment_size: usize,
}

impl Default for PointSegmenter {
    fn default() -> Self {
        Self {
            neighbor_distance: 0.5,
            plane_tolerance: 0.3,
            min_segment_size: 10,
        }
    }
}

impl PointSegmenter {
    pub fn new(neighbor_distance: f64, plane_tolerance: f64, min_segment_size: usize) -> Self {
        Self {
            neighbor_distance,
            plane_tolerance,
            min_segment_size,
        }
    }

    /// Segment points based on detected planes.
    pub fn segment_by_planes(
        &self,
        point_cloud: &PointCloud,
        planes: &[Plane],
    ) -> Vec<Vec<usize>> {
        if point_cloud.is_empty() || planes.is_empty() {
            return vec![];
        }

        let mut segments: Vec<Vec<usize>> = vec![Vec::new(); planes.len()];
        let mut unsegmented = Vec::new();

        // Assign each point to the closest plane (if within tolerance)
        for (i, point) in point_cloud.positions.iter().enumerate() {
            let mut best_plane = None;
            let mut best_distance = f64::MAX;

            for (j, plane) in planes.iter().enumerate() {
                let dist = plane.distance(point);
                if dist < self.plane_tolerance && dist < best_distance {
                    best_distance = dist;
                    best_plane = Some(j);
                }
            }

            if let Some(plane_idx) = best_plane {
                segments[plane_idx].push(i);
            } else {
                unsegmented.push(i);
            }
        }

        // Filter out small segments
        segments
            .into_iter()
            .filter(|s| s.len() >= self.min_segment_size)
            .collect()
    }

    /// Region growing segmentation based on local planarity.
    pub fn region_growing(&self, point_cloud: &PointCloud) -> Vec<Vec<usize>> {
        if point_cloud.is_empty() {
            return vec![];
        }

        // Build KD-tree for neighbor queries
        let mut kdtree: KdTree<f64, 3> = KdTree::new();
        for (i, p) in point_cloud.positions.iter().enumerate() {
            kdtree.add(&[p.x, p.y, p.z], i as u64);
        }

        let mut labels = vec![-1i32; point_cloud.len()];
        let mut current_label = 0i32;
        let mut segments = Vec::new();

        for seed_idx in 0..point_cloud.len() {
            if labels[seed_idx] >= 0 {
                continue;
            }

            // Start new region
            let mut region = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(seed_idx);
            labels[seed_idx] = current_label;

            while let Some(idx) = queue.pop_front() {
                region.push(idx);

                let point = &point_cloud.positions[idx];
                let neighbors = kdtree.within::<kiddo::SquaredEuclidean>(
                    &[point.x, point.y, point.z],
                    self.neighbor_distance * self.neighbor_distance,
                );

                for neighbor in neighbors {
                    let neighbor_idx = neighbor.item as usize;
                    if labels[neighbor_idx] >= 0 {
                        continue;
                    }

                    // Check if neighbor is compatible (simple distance check)
                    labels[neighbor_idx] = current_label;
                    queue.push_back(neighbor_idx);
                }
            }

            if region.len() >= self.min_segment_size {
                segments.push(region);
                current_label += 1;
            }
        }

        segments
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_by_planes() {
        let segmenter = PointSegmenter::default();

        // Create a point cloud with two distinct planes
        let mut positions = Vec::new();

        // Points on z = 10 plane
        for x in 0..5 {
            for y in 0..5 {
                positions.push(Point3::new(x as f64, y as f64, 10.0));
            }
        }

        // Points on z = 15 plane
        for x in 0..5 {
            for y in 0..5 {
                positions.push(Point3::new(x as f64, y as f64, 15.0));
            }
        }

        let point_cloud = PointCloud::from_positions(positions);

        // Create two planes
        let planes = vec![
            Plane::new(nalgebra::Vector3::new(0.0, 0.0, 1.0), -10.0),
            Plane::new(nalgebra::Vector3::new(0.0, 0.0, 1.0), -15.0),
        ];

        let segments = segmenter.segment_by_planes(&point_cloud, &planes);

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].len(), 25);
        assert_eq!(segments[1].len(), 25);
    }

    #[test]
    fn test_region_growing() {
        let segmenter = PointSegmenter {
            neighbor_distance: 1.5,
            min_segment_size: 5,
            ..Default::default()
        };

        // Create a point cloud with two clusters
        let mut positions = Vec::new();

        // Cluster 1 centered at (0, 0, 0)
        for x in 0..3 {
            for y in 0..3 {
                positions.push(Point3::new(x as f64, y as f64, 0.0));
            }
        }

        // Cluster 2 centered at (10, 10, 0) - far away
        for x in 10..13 {
            for y in 10..13 {
                positions.push(Point3::new(x as f64, y as f64, 0.0));
            }
        }

        let point_cloud = PointCloud::from_positions(positions);
        let segments = segmenter.region_growing(&point_cloud);

        assert_eq!(segments.len(), 2);
    }
}

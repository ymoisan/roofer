//! Point cloud data structures.
//!
//! Provides efficient storage for 3D point clouds with optional attributes
//! like intensity and classification.

use nalgebra::Point3;
use serde::{Deserialize, Serialize};

/// A 3D point cloud with optional per-point attributes.
#[derive(Debug, Clone, Default)]
pub struct PointCloud {
    /// 3D positions
    pub positions: Vec<Point3<f64>>,
    /// Optional intensity values (0-65535 typically)
    pub intensity: Option<Vec<u16>>,
    /// Optional classification (LAS standard classes)
    pub classification: Option<Vec<u8>>,
    /// Optional return number
    pub return_number: Option<Vec<u8>>,
    /// Optional number of returns
    pub number_of_returns: Option<Vec<u8>>,
}

impl PointCloud {
    /// Create a new empty point cloud.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a point cloud with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            positions: Vec::with_capacity(capacity),
            intensity: None,
            classification: None,
            return_number: None,
            number_of_returns: None,
        }
    }

    /// Create a point cloud from positions only.
    pub fn from_positions(positions: Vec<Point3<f64>>) -> Self {
        Self {
            positions,
            ..Default::default()
        }
    }

    /// Number of points in the cloud.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Add a point with just position.
    pub fn push(&mut self, point: Point3<f64>) {
        self.positions.push(point);
        // Extend optional arrays if they exist
        if let Some(ref mut intensity) = self.intensity {
            intensity.push(0);
        }
        if let Some(ref mut classification) = self.classification {
            classification.push(0);
        }
        if let Some(ref mut return_number) = self.return_number {
            return_number.push(0);
        }
        if let Some(ref mut number_of_returns) = self.number_of_returns {
            number_of_returns.push(0);
        }
    }

    /// Add a point with all attributes.
    pub fn push_full(
        &mut self,
        point: Point3<f64>,
        intensity: u16,
        classification: u8,
        return_number: u8,
        number_of_returns: u8,
    ) {
        self.positions.push(point);

        // Initialize or extend intensity
        if self.intensity.is_none() && self.positions.len() > 1 {
            self.intensity = Some(vec![0; self.positions.len() - 1]);
        }
        if let Some(ref mut arr) = self.intensity {
            arr.push(intensity);
        } else {
            self.intensity = Some(vec![intensity]);
        }

        // Initialize or extend classification
        if self.classification.is_none() && self.positions.len() > 1 {
            self.classification = Some(vec![0; self.positions.len() - 1]);
        }
        if let Some(ref mut arr) = self.classification {
            arr.push(classification);
        } else {
            self.classification = Some(vec![classification]);
        }

        // Initialize or extend return_number
        if self.return_number.is_none() && self.positions.len() > 1 {
            self.return_number = Some(vec![0; self.positions.len() - 1]);
        }
        if let Some(ref mut arr) = self.return_number {
            arr.push(return_number);
        } else {
            self.return_number = Some(vec![return_number]);
        }

        // Initialize or extend number_of_returns
        if self.number_of_returns.is_none() && self.positions.len() > 1 {
            self.number_of_returns = Some(vec![0; self.positions.len() - 1]);
        }
        if let Some(ref mut arr) = self.number_of_returns {
            arr.push(number_of_returns);
        } else {
            self.number_of_returns = Some(vec![number_of_returns]);
        }
    }

    /// Get the bounding box as (min, max) points.
    pub fn bounding_box(&self) -> Option<(Point3<f64>, Point3<f64>)> {
        if self.positions.is_empty() {
            return None;
        }

        let mut min = self.positions[0];
        let mut max = self.positions[0];

        for p in &self.positions[1..] {
            min.x = min.x.min(p.x);
            min.y = min.y.min(p.y);
            min.z = min.z.min(p.z);
            max.x = max.x.max(p.x);
            max.y = max.y.max(p.y);
            max.z = max.z.max(p.z);
        }

        Some((min, max))
    }

    /// Get centroid of the point cloud.
    pub fn centroid(&self) -> Option<Point3<f64>> {
        if self.positions.is_empty() {
            return None;
        }

        let sum: Point3<f64> = self
            .positions
            .iter()
            .fold(Point3::origin(), |acc, p| Point3::new(acc.x + p.x, acc.y + p.y, acc.z + p.z));

        let n = self.positions.len() as f64;
        Some(Point3::new(sum.x / n, sum.y / n, sum.z / n))
    }

    /// Filter points by classification.
    pub fn filter_by_classification(&self, classes: &[u8]) -> PointCloud {
        let classification = match &self.classification {
            Some(c) => c,
            None => return self.clone(),
        };

        let indices: Vec<usize> = classification
            .iter()
            .enumerate()
            .filter(|(_, &c)| classes.contains(&c))
            .map(|(i, _)| i)
            .collect();

        self.select_indices(&indices)
    }

    /// Select points by indices.
    pub fn select_indices(&self, indices: &[usize]) -> PointCloud {
        let positions: Vec<Point3<f64>> = indices.iter().map(|&i| self.positions[i]).collect();

        let intensity = self
            .intensity
            .as_ref()
            .map(|arr| indices.iter().map(|&i| arr[i]).collect());

        let classification = self
            .classification
            .as_ref()
            .map(|arr| indices.iter().map(|&i| arr[i]).collect());

        let return_number = self
            .return_number
            .as_ref()
            .map(|arr| indices.iter().map(|&i| arr[i]).collect());

        let number_of_returns = self
            .number_of_returns
            .as_ref()
            .map(|arr| indices.iter().map(|&i| arr[i]).collect());

        PointCloud {
            positions,
            intensity,
            classification,
            return_number,
            number_of_returns,
        }
    }

    /// Merge another point cloud into this one.
    pub fn extend(&mut self, other: &PointCloud) {
        self.positions.extend(other.positions.iter().cloned());

        // Handle intensity
        match (&mut self.intensity, &other.intensity) {
            (Some(ref mut arr), Some(other_arr)) => arr.extend(other_arr.iter().cloned()),
            (None, Some(other_arr)) => {
                let mut new_arr = vec![0u16; self.positions.len() - other.positions.len()];
                new_arr.extend(other_arr.iter().cloned());
                self.intensity = Some(new_arr);
            }
            (Some(ref mut arr), None) => arr.extend(vec![0u16; other.positions.len()]),
            (None, None) => {}
        }

        // Handle classification
        match (&mut self.classification, &other.classification) {
            (Some(ref mut arr), Some(other_arr)) => arr.extend(other_arr.iter().cloned()),
            (None, Some(other_arr)) => {
                let mut new_arr = vec![0u8; self.positions.len() - other.positions.len()];
                new_arr.extend(other_arr.iter().cloned());
                self.classification = Some(new_arr);
            }
            (Some(ref mut arr), None) => arr.extend(vec![0u8; other.positions.len()]),
            (None, None) => {}
        }

        // Handle return_number
        match (&mut self.return_number, &other.return_number) {
            (Some(ref mut arr), Some(other_arr)) => arr.extend(other_arr.iter().cloned()),
            (None, Some(other_arr)) => {
                let mut new_arr = vec![0u8; self.positions.len() - other.positions.len()];
                new_arr.extend(other_arr.iter().cloned());
                self.return_number = Some(new_arr);
            }
            (Some(ref mut arr), None) => arr.extend(vec![0u8; other.positions.len()]),
            (None, None) => {}
        }

        // Handle number_of_returns
        match (&mut self.number_of_returns, &other.number_of_returns) {
            (Some(ref mut arr), Some(other_arr)) => arr.extend(other_arr.iter().cloned()),
            (None, Some(other_arr)) => {
                let mut new_arr = vec![0u8; self.positions.len() - other.positions.len()];
                new_arr.extend(other_arr.iter().cloned());
                self.number_of_returns = Some(new_arr);
            }
            (Some(ref mut arr), None) => arr.extend(vec![0u8; other.positions.len()]),
            (None, None) => {}
        }
    }

    /// Compute statistics about the point cloud.
    pub fn compute_statistics(&self) -> PointCloudStatistics {
        let (min, max) = self.bounding_box().unwrap_or((Point3::origin(), Point3::origin()));
        let centroid = self.centroid().unwrap_or(Point3::origin());

        // Compute elevation percentiles
        let mut z_values: Vec<f64> = self.positions.iter().map(|p| p.z).collect();
        z_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let z_min = z_values.first().copied().unwrap_or(0.0);
        let z_max = z_values.last().copied().unwrap_or(0.0);
        let z_50p = percentile(&z_values, 50.0);
        let z_70p = percentile(&z_values, 70.0);
        let z_98p = percentile(&z_values, 98.0);

        PointCloudStatistics {
            count: self.len(),
            bounds_min: min,
            bounds_max: max,
            centroid,
            z_min,
            z_max,
            z_50p,
            z_70p,
            z_98p,
        }
    }
}

/// Statistics computed from a point cloud.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointCloudStatistics {
    pub count: usize,
    pub bounds_min: Point3<f64>,
    pub bounds_max: Point3<f64>,
    pub centroid: Point3<f64>,
    pub z_min: f64,
    pub z_max: f64,
    pub z_50p: f64,
    pub z_70p: f64,
    pub z_98p: f64,
}

/// Compute percentile from a sorted array.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_cloud_basic() {
        let mut pc = PointCloud::new();
        pc.push(Point3::new(0.0, 0.0, 0.0));
        pc.push(Point3::new(1.0, 1.0, 1.0));
        pc.push(Point3::new(2.0, 2.0, 2.0));

        assert_eq!(pc.len(), 3);

        let (min, max) = pc.bounding_box().unwrap();
        assert_eq!(min, Point3::new(0.0, 0.0, 0.0));
        assert_eq!(max, Point3::new(2.0, 2.0, 2.0));

        let centroid = pc.centroid().unwrap();
        assert_eq!(centroid, Point3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn test_point_cloud_statistics() {
        let positions: Vec<Point3<f64>> = (0..100)
            .map(|i| Point3::new(i as f64, i as f64, i as f64))
            .collect();
        let pc = PointCloud::from_positions(positions);

        let stats = pc.compute_statistics();
        assert_eq!(stats.count, 100);
        assert_eq!(stats.z_min, 0.0);
        assert_eq!(stats.z_max, 99.0);
    }
}

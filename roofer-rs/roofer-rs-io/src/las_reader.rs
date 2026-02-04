//! LAS/LAZ point cloud reader.
//!
//! Reads LAS and LAZ files using the `las` crate.
//! Supports reading CRS from LAS/LAZ headers (GeoTIFF or WKT VLRs).

use las::crs::GeoTiffData;
use las::Reader;
use nalgebra::Point3;
use roofer_rs_geometry::PointCloud;
use std::path::Path;
use thiserror::Error;

/// Parse horizontal EPSG code from WKT (e.g. AUTHORITY["EPSG","25832"]).
fn parse_epsg_from_wkt(wkt: &str) -> Option<String> {
    // Look for AUTHORITY["EPSG","XXXX"] - prefer first occurrence (often horizontal CRS)
    let marker = "AUTHORITY[\"EPSG\",\"";
    let start = wkt.find(marker)?;
    let value_start = start + marker.len();
    let value_end = wkt[value_start..].find('"')? + value_start;
    let code = wkt[value_start..value_end].trim();
    if code.chars().all(|c| c.is_ascii_digit()) && !code.is_empty() {
        Some(code.to_string())
    } else {
        None
    }
}

#[derive(Debug, Error)]
pub enum LasError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("LAS error: {0}")]
    LasError(#[from] las::Error),
    #[error("File not found: {0}")]
    FileNotFound(String),
}

/// Reader for LAS/LAZ point cloud files.
pub struct LasReader {
    /// Classification codes to filter (empty = all classes)
    pub class_filter: Vec<u8>,
}

impl Default for LasReader {
    fn default() -> Self {
        Self::new()
    }
}

impl LasReader {
    pub fn new() -> Self {
        Self {
            class_filter: Vec::new(),
        }
    }

    /// Set classification filter.
    pub fn with_class_filter(mut self, classes: Vec<u8>) -> Self {
        self.class_filter = classes;
        self
    }

    /// Read a LAS/LAZ file into a PointCloud.
    pub fn read_file(&self, path: &Path) -> Result<PointCloud, LasError> {
        if !path.exists() {
            return Err(LasError::FileNotFound(path.display().to_string()));
        }

        let mut reader = Reader::from_path(path)?;
        let header = reader.header();

        tracing::debug!(
            "Reading LAS file: {} ({} points)",
            path.display(),
            header.number_of_points()
        );

        let mut point_cloud = PointCloud::with_capacity(header.number_of_points() as usize);

        for wrapped_point in reader.points() {
            let point = wrapped_point?;

            // Apply class filter if set
            if !self.class_filter.is_empty()
                && !self.class_filter.contains(&point.classification.into())
            {
                continue;
            }

            point_cloud.push_full(
                Point3::new(point.x, point.y, point.z),
                point.intensity,
                point.classification.into(),
                point.return_number,
                point.number_of_returns,
            );
        }

        Ok(point_cloud)
    }

    /// Read multiple LAS/LAZ files into a single PointCloud.
    pub fn read_files(&self, paths: &[impl AsRef<Path>]) -> Result<PointCloud, LasError> {
        let mut combined = PointCloud::new();

        for path in paths {
            let pc = self.read_file(path.as_ref())?;
            combined.extend(&pc);
        }

        Ok(combined)
    }

    /// Read the coordinate reference system from a LAS/LAZ file header.
    /// Returns the EPSG code as a string (e.g. "25832") if present in GeoTIFF or WKT VLRs.
    pub fn read_crs(&self, path: &Path) -> Result<Option<String>, LasError> {
        if !path.exists() {
            return Err(LasError::FileNotFound(path.display().to_string()));
        }

        let reader = Reader::from_path(path)?;
        let header = reader.header();

        // Try GeoTIFF CRS first (key 2048 = ProjectedCSTypeGeoKey, 3072 = GeographicTypeGeoKey)
        if let Ok(Some(geotiff)) = header.get_geotiff_crs() {
            for entry in &geotiff.entries {
                if entry.id == 2048 || entry.id == 3072 {
                    if let GeoTiffData::U16(epsg) = &entry.data {
                        tracing::info!("Found CRS in LAS header (GeoTIFF): EPSG:{}", epsg);
                        return Ok(Some(epsg.to_string()));
                    }
                }
            }
        }

        // Fallback: parse WKT CRS for AUTHORITY["EPSG","XXXX"]
        if let Some(wkt_bytes) = header.get_wkt_crs_bytes() {
            let wkt = String::from_utf8_lossy(wkt_bytes);
            if let Some(epsg) = parse_epsg_from_wkt(&wkt) {
                tracing::info!("Found CRS in LAS header (WKT): EPSG:{}", epsg);
                return Ok(Some(epsg.to_string()));
            }
        }

        Ok(None)
    }

    /// Read CRS from the first of several LAS/LAZ files (e.g. point cloud list).
    pub fn read_crs_from_first(&self, paths: &[impl AsRef<Path>]) -> Result<Option<String>, LasError> {
        for path in paths {
            let path = path.as_ref();
            if path.exists() {
                if let Ok(Some(crs)) = self.read_crs(path) {
                    return Ok(Some(crs));
                }
            }
        }
        Ok(None)
    }

    /// Get metadata from a LAS file without reading all points.
    pub fn read_metadata(&self, path: &Path) -> Result<LasMetadata, LasError> {
        if !path.exists() {
            return Err(LasError::FileNotFound(path.display().to_string()));
        }

        let reader = Reader::from_path(path)?;
        let header = reader.header();
        let bounds = header.bounds();

        Ok(LasMetadata {
            point_count: header.number_of_points(),
            bounds_min: Point3::new(bounds.min.x, bounds.min.y, bounds.min.z),
            bounds_max: Point3::new(bounds.max.x, bounds.max.y, bounds.max.z),
            point_format: header.point_format().to_u8().unwrap_or(0),
            version: format!("{}.{}", header.version().major, header.version().minor),
        })
    }
}

/// Metadata from a LAS file header.
#[derive(Debug, Clone)]
pub struct LasMetadata {
    pub point_count: u64,
    pub bounds_min: Point3<f64>,
    pub bounds_max: Point3<f64>,
    pub point_format: u8,
    pub version: String,
}

impl LasMetadata {
    /// Check if a 2D point (x, y) is within the file bounds.
    pub fn contains_2d(&self, x: f64, y: f64) -> bool {
        x >= self.bounds_min.x
            && x <= self.bounds_max.x
            && y >= self.bounds_min.y
            && y <= self.bounds_max.y
    }

    /// Check if a 2D box overlaps with the file bounds.
    pub fn overlaps_2d(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> bool {
        !(max_x < self.bounds_min.x
            || min_x > self.bounds_max.x
            || max_y < self.bounds_min.y
            || min_y > self.bounds_max.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_las_metadata_bounds() {
        let meta = LasMetadata {
            point_count: 1000,
            bounds_min: Point3::new(0.0, 0.0, 0.0),
            bounds_max: Point3::new(100.0, 100.0, 50.0),
            point_format: 1,
            version: "1.4".to_string(),
        };

        assert!(meta.contains_2d(50.0, 50.0));
        assert!(!meta.contains_2d(150.0, 50.0));

        assert!(meta.overlaps_2d(50.0, 50.0, 150.0, 150.0));
        assert!(!meta.overlaps_2d(150.0, 150.0, 200.0, 200.0));
    }
}

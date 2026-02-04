//! Configuration file parsing.
//!
//! Parses TOML configuration files compatible with the C++ roofer.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Point cloud source configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PointCloudConfig {
    /// Name of the point cloud
    pub name: String,
    /// Source paths (files or directories)
    pub source: Vec<PathBuf>,
}

/// Output attribute name mapping.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(default)]
pub struct OutputAttributes {
    pub status: String,
    pub reconstruction_time: String,
    pub val3dity_lod12: String,
    pub val3dity_lod13: String,
    pub val3dity_lod22: String,
    pub is_glass_roof: String,
    pub nodata_frac: String,
    pub nodata_r: String,
    pub pt_density: String,
    pub is_mutated: String,
    pub pc_select: String,
    pub pc_source: String,
    pub pc_year: String,
    pub force_lod11: String,
    pub roof_type: String,
    pub h_roof_50p: String,
    pub h_roof_70p: String,
    pub h_roof_min: String,
    pub h_roof_max: String,
    pub roof_n_planes: String,
    pub rmse_lod12: String,
    pub rmse_lod13: String,
    pub rmse_lod22: String,
    pub h_ground: String,
    pub slope: String,
    pub azimuth: String,
}

impl OutputAttributes {
    pub fn with_defaults() -> Self {
        Self {
            status: "rf_status".to_string(),
            reconstruction_time: "rf_reconstruction_time".to_string(),
            val3dity_lod12: "rf_val3dity_lod12".to_string(),
            val3dity_lod13: "rf_val3dity_lod13".to_string(),
            val3dity_lod22: "rf_val3dity_lod22".to_string(),
            is_glass_roof: "rf_is_glass_roof".to_string(),
            nodata_frac: "rf_nodata_frac".to_string(),
            nodata_r: "rf_nodata_r".to_string(),
            pt_density: "rf_pt_density".to_string(),
            is_mutated: "rf_is_mutated".to_string(),
            pc_select: "rf_pc_select".to_string(),
            pc_source: "rf_pc_source".to_string(),
            pc_year: "rf_pc_year".to_string(),
            force_lod11: "rf_force_lod11".to_string(),
            roof_type: "rf_roof_type".to_string(),
            h_roof_50p: "rf_roof_elevation_50p".to_string(),
            h_roof_70p: "rf_roof_elevation_70p".to_string(),
            h_roof_min: "rf_roof_elevation_min".to_string(),
            h_roof_max: "rf_roof_elevation_max".to_string(),
            roof_n_planes: "rf_roof_n_planes".to_string(),
            rmse_lod12: "rf_rmse_lod12".to_string(),
            rmse_lod13: "rf_rmse_lod13".to_string(),
            rmse_lod22: "rf_rmse_lod22".to_string(),
            h_ground: "rf_h_ground".to_string(),
            slope: "rf_slope".to_string(),
            azimuth: "rf_azimuth".to_string(),
        }
    }
}

/// Main configuration structure.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct Config {
    /// Vector source with polygon footprints
    pub polygon_source: PathBuf,

    /// Layer name in polygon source
    #[serde(default)]
    pub polygon_source_layer: Option<String>,

    /// Building ID attribute
    #[serde(default)]
    pub id_attribute: Option<String>,

    /// Boolean attribute for forcing LoD 1.1
    #[serde(default)]
    pub force_lod11_attribute: Option<String>,

    /// OGR SQL WHERE clause filter
    #[serde(default)]
    pub filter: Option<String>,

    /// Override SRS for inputs and outputs
    #[serde(default)]
    pub srs: Option<String>,

    /// Output CRS for CityJSON (e.g., "EPSG:4326"). When set, output is reprojected.
    #[serde(default, rename = "output-srs")]
    pub output_srs: Option<String>,

    /// Region of interest [x_min, y_min, x_max, y_max]
    #[serde(rename = "box", default)]
    pub roi_box: Option<[f64; 4]>,

    /// Point density ceiling
    #[serde(default = "default_ceil_point_density")]
    pub ceil_point_density: f64,

    /// Tile size [size_x, size_y]
    #[serde(default = "default_tilesize")]
    pub tilesize: [f64; 2],

    /// Cellsize for quick point cloud analysis
    #[serde(default = "default_cellsize")]
    pub cellsize: f64,

    // Reconstruction options
    /// Plane detect epsilon
    #[serde(default = "default_plane_detect_epsilon")]
    pub plane_detect_epsilon: f64,

    /// Plane detect k (neighbors)
    #[serde(default = "default_plane_detect_k")]
    pub plane_detect_k: usize,

    /// Plane detect minimum points
    #[serde(default = "default_plane_detect_min_points")]
    pub plane_detect_min_points: usize,

    /// Threshold for classifying a plane as vertical (wall) based on |n·z|
    #[serde(default = "default_plane_metrics_is_wall_threshold")]
    pub plane_metrics_is_wall_threshold: f64,

    /// Threshold for classifying a plane as horizontal based on |n·z|
    #[serde(default = "default_plane_metrics_is_horizontal_threshold")]
    pub plane_metrics_is_horizontal_threshold: f64,

    /// Angle threshold (degrees) for merging similar planes
    #[serde(default = "default_plane_merge_angle_degrees")]
    pub plane_merge_angle_degrees: f64,

    /// Distance threshold for merging similar planes
    #[serde(default = "default_plane_merge_distance")]
    pub plane_merge_distance: f64,

    /// Distance threshold for merging horizontal planes
    #[serde(default = "default_plane_merge_distance_horizontal")]
    pub plane_merge_distance_horizontal: f64,

    /// Minimum inlier overlap ratio for merging planes
    #[serde(default = "default_plane_merge_inlier_overlap")]
    pub plane_merge_inlier_overlap: f64,

    /// Minimum inlier ratio for accepting a plane (relative to remaining points)
    #[serde(default = "default_plane_min_inlier_ratio")]
    pub plane_min_inlier_ratio: f64,

    /// Max inliers to treat a plane as "small" for merging
    #[serde(default = "default_plane_small_plane_max_inliers")]
    pub plane_small_plane_max_inliers: usize,

    /// Neighbor distance threshold for merging small planes
    #[serde(default = "default_plane_merge_neighbor_distance")]
    pub plane_merge_neighbor_distance: f64,

    /// Step height for LoD 1.3
    #[serde(default = "default_lod13_step_height")]
    pub lod13_step_height: f64,

    /// Complexity factor
    #[serde(default = "default_complexity_factor")]
    pub complexity_factor: f64,

    /// Which LoDs to generate (12, 13, 22)
    #[serde(default)]
    pub lod: Option<u8>,

    // Output options
    /// Split CityJSON Sequence per building
    #[serde(default)]
    pub split_cjseq: bool,

    /// Omit metadata from output
    #[serde(default)]
    pub omit_metadata: bool,

    /// CityJSON transform translation
    #[serde(default)]
    pub cj_translate: Option<[f64; 3]>,

    /// CityJSON transform scale
    #[serde(default = "default_cj_scale")]
    pub cj_scale: [f64; 3],

    /// Output directory
    pub output_directory: PathBuf,

    /// Point cloud sources
    #[serde(default)]
    pub pointclouds: Vec<PointCloudConfig>,

    /// Output attribute names
    #[serde(default = "OutputAttributes::with_defaults")]
    pub output_attributes: OutputAttributes,

    /// Include process/debug attributes in output
    #[serde(default = "default_include_process_attributes")]
    pub include_process_attributes: bool,

    // Wall generation options
    /// Minimum height difference to create internal wall (meters).
    /// If None or 0.0, create walls for all edges regardless of height difference.
    #[serde(default, rename = "wall-min-height-diff")]
    pub wall_min_height_diff: Option<f64>,

    /// Snap tolerance exponent (10^-exp) for filtering short edges.
    #[serde(default = "default_wall_snap_tolerance_exp", rename = "wall-snap-tolerance-exp")]
    pub wall_snap_tolerance_exp: i32,

    /// Build walls for all edges vs only height-different edges.
    #[serde(default = "default_wall_build_all_edges", rename = "wall-build-all-edges")]
    pub wall_build_all_edges: bool,

    /// Minimum edge length to create wall (meters).
    #[serde(default = "default_wall_min_edge_length", rename = "wall-min-edge-length")]
    pub wall_min_edge_length: f64,
}

fn default_ceil_point_density() -> f64 {
    20.0
}

fn default_tilesize() -> [f64; 2] {
    [1000.0, 1000.0]
}

fn default_cellsize() -> f64 {
    0.5
}

fn default_plane_detect_epsilon() -> f64 {
    0.3
}

fn default_plane_detect_k() -> usize {
    15
}

fn default_plane_detect_min_points() -> usize {
    15
}

fn default_plane_metrics_is_wall_threshold() -> f64 {
    0.3
}

fn default_plane_metrics_is_horizontal_threshold() -> f64 {
    0.995
}

fn default_plane_merge_angle_degrees() -> f64 {
    7.5
}

fn default_plane_merge_distance() -> f64 {
    0.5
}

fn default_plane_merge_distance_horizontal() -> f64 {
    1.5
}

fn default_plane_merge_inlier_overlap() -> f64 {
    0.6
}

fn default_plane_min_inlier_ratio() -> f64 {
    0.04
}

fn default_plane_small_plane_max_inliers() -> usize {
    60
}

fn default_plane_merge_neighbor_distance() -> f64 {
    2.0
}

fn default_lod13_step_height() -> f64 {
    3.0
}

fn default_complexity_factor() -> f64 {
    0.7
}

fn default_cj_scale() -> [f64; 3] {
    [0.001, 0.001, 0.001]
}

fn default_include_process_attributes() -> bool {
    true
}

fn default_wall_snap_tolerance_exp() -> i32 {
    4
}

fn default_wall_build_all_edges() -> bool {
    true
}

fn default_wall_min_edge_length() -> f64 {
    0.0
}

impl Config {
    /// Load configuration from a TOML file.
    pub fn from_file(path: &std::path::Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path).map_err(|e| ConfigError::IoError(e.to_string()))?;
        Self::from_str(&content)
    }

    /// Parse configuration from a TOML string.
    pub fn from_str(content: &str) -> Result<Self, ConfigError> {
        toml::from_str(content).map_err(|e| ConfigError::ParseError(e.to_string()))
    }

    /// Get all point cloud source paths.
    pub fn point_cloud_paths(&self) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        for pc in &self.pointclouds {
            for source in &pc.source {
                if source.is_dir() {
                    // Recursively find LAS/LAZ files
                    if let Ok(entries) = walkdir(source) {
                        paths.extend(entries);
                    }
                } else {
                    paths.push(source.clone());
                }
            }
        }
        paths
    }
}

/// Walk directory for LAS/LAZ files.
fn walkdir(dir: &std::path::Path) -> std::io::Result<Vec<PathBuf>> {
    let mut results = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            results.extend(walkdir(&path)?);
        } else if let Some(ext) = path.extension() {
            let ext = ext.to_string_lossy().to_lowercase();
            if ext == "las" || ext == "laz" {
                results.push(path);
            }
        }
    }
    Ok(results)
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    IoError(String),
    #[error("Parse error: {0}")]
    ParseError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_config() {
        let toml = r#"
            polygon-source = "data/footprints.gpkg"
            output-directory = "output"
            
            [[pointclouds]]
            name = "AHN3"
            source = ["data/pointcloud.laz"]
        "#;

        let config = Config::from_str(toml).unwrap();
        assert_eq!(config.polygon_source, PathBuf::from("data/footprints.gpkg"));
        assert_eq!(config.pointclouds.len(), 1);
        assert_eq!(config.pointclouds[0].name, "AHN3");
    }

    #[test]
    fn test_default_values() {
        let toml = r#"
            polygon-source = "footprints.gpkg"
            output-directory = "output"
        "#;

        let config = Config::from_str(toml).unwrap();
        assert_eq!(config.plane_detect_epsilon, 0.3);
        assert_eq!(config.plane_detect_k, 15);
        assert_eq!(config.ceil_point_density, 20.0);
    }
}

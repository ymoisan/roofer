//! Building reconstruction algorithms.
//!
//! This module contains the core algorithms for reconstructing 3D building
//! geometry from point clouds and footprints.

pub mod lod12;
pub mod lod22;
pub mod segmentation;

pub use lod12::Lod12Reconstructor;
pub use lod22::Lod22Reconstructor;
pub use segmentation::PointSegmenter;

use crate::mesh::{BuildingGeometry, SurfaceType};
use crate::plane::RansacConfig;
use crate::point_cloud::PointCloud;
use crate::polygon::Footprint;

pub use lod22::WallConfig;

/// Configuration for building reconstruction.
#[derive(Debug, Clone)]
pub struct ReconstructionConfig {
    /// RANSAC parameters for plane detection
    pub plane_config: RansacConfig,
    /// Step height for LoD 1.3 (meters)
    pub lod13_step_height: f64,
    /// Complexity factor for optimization (0.0 - 1.0)
    pub complexity_factor: f64,
    /// Which LoDs to generate
    pub lods: Vec<Lod>,
    /// Wall generation configuration
    pub wall_config: WallConfig,
}

impl Default for ReconstructionConfig {
    fn default() -> Self {
        Self {
            plane_config: RansacConfig::default(),
            lod13_step_height: 3.0,
            complexity_factor: 0.7,
            lods: vec![Lod::Lod12, Lod::Lod13, Lod::Lod22],
            wall_config: WallConfig::default(),
        }
    }
}

/// Level of Detail options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lod {
    Lod12,
    Lod13,
    Lod22,
}

/// Result of building reconstruction.
#[derive(Debug)]
pub struct ReconstructionResult {
    pub geometry: BuildingGeometry,
    pub success: bool,
    pub error_message: Option<String>,
    pub reconstruction_time_ms: u64,
}

/// Main reconstructor that orchestrates the reconstruction pipeline.
pub struct BuildingReconstructor {
    config: ReconstructionConfig,
    lod12: Lod12Reconstructor,
    lod22: Lod22Reconstructor,
}

impl BuildingReconstructor {
    pub fn new(config: ReconstructionConfig) -> Self {
        Self {
            lod12: Lod12Reconstructor::new(),
            lod22: Lod22Reconstructor::new_with_wall_config(
                config.plane_config.clone(),
                config.wall_config.clone(),
            ),
            config,
        }
    }

    /// Reconstruct a building from its footprint and point cloud.
    pub fn reconstruct(&self, footprint: &Footprint, point_cloud: &PointCloud) -> ReconstructionResult {
        let start = std::time::Instant::now();
        let mut geometry = BuildingGeometry::new(&footprint.id);

        // Copy attributes from footprint
        geometry.attributes = footprint.attributes.clone();

        // Check if we have enough points
        if point_cloud.len() < self.config.plane_config.min_points {
            return ReconstructionResult {
                geometry,
                success: false,
                error_message: Some(format!(
                    "Not enough points: {} < {}",
                    point_cloud.len(),
                    self.config.plane_config.min_points
                )),
                reconstruction_time_ms: start.elapsed().as_millis() as u64,
            };
        }

        // Compute ground height (use minimum Z or footprint Z if available)
        let stats = point_cloud.compute_statistics();
        geometry.h_ground = stats.z_min;

        // Generate requested LoDs
        let mut success = false;
        let mut error_message = None;

        if self.config.lods.contains(&Lod::Lod12) {
            match self.lod12.reconstruct(footprint, point_cloud, geometry.h_ground) {
                Ok(mesh) => {
                    geometry.lod12 = Some(mesh);
                    success = true;
                }
                Err(e) => {
                    tracing::warn!("LoD 1.2 reconstruction failed: {}", e);
                    error_message = Some(e.to_string());
                }
            }
        }

        if self.config.lods.contains(&Lod::Lod22) {
            match self.lod22.reconstruct(footprint, point_cloud, geometry.h_ground) {
                Ok(mesh) => {
                    geometry.lod22 = Some(mesh);
                    success = true;
                }
                Err(e) => {
                    tracing::warn!("LoD 2.2 reconstruction failed: {}", e);
                    if error_message.is_none() {
                        error_message = Some(e.to_string());
                    }
                }
            }
        }

        // Add reconstruction attributes
        geometry.attributes.insert("rf_success", success);
        geometry.attributes.insert("rf_h_ground", geometry.h_ground);
        geometry.attributes.insert("rf_h_roof_50p", stats.z_50p);
        geometry.attributes.insert("rf_h_roof_70p", stats.z_70p);
        geometry.attributes.insert("rf_h_roof_max", stats.z_max);
        geometry.attributes.insert("rf_h_roof_min", stats.z_min);
        geometry.attributes.insert("rf_pt_density", point_cloud.len() as f64);

        // Roof analytics from LoD 2.2 mesh, when available
        if let Some(ref mesh) = geometry.lod22 {
            let roof_planes = mesh
                .semantics
                .iter()
                .filter(|s| s.surface_type == SurfaceType::RoofSurface)
                .count();
            geometry
                .attributes
                .insert("rf_roof_planes", roof_planes as i64);

            let (horiz_cnt, slant_cnt) = count_roof_plane_types(mesh);
            let roof_type = classify_roof_type(roof_planes, horiz_cnt, slant_cnt);
            geometry.attributes.insert("rf_roof_type", roof_type);

            // Heuristic ridgeline count when we don't intersect planes explicitly
            let ridgelines = if slant_cnt > 1 {
                slant_cnt - 1
            } else {
                0
            };
            geometry
                .attributes
                .insert("rf_ridgelines", ridgelines as i64);

            let volume = mesh.compute_volume();
            geometry.attributes.insert("rf_volume_lod22", volume);
        }

        ReconstructionResult {
            geometry,
            success,
            error_message,
            reconstruction_time_ms: start.elapsed().as_millis() as u64,
        }
    }
}

fn count_roof_plane_types(mesh: &crate::mesh::Mesh) -> (usize, usize) {
    let mut horizontal = 0;
    let mut slanted = 0;
    for s in &mesh.semantics {
        if s.surface_type != SurfaceType::RoofSurface {
            continue;
        }
        if let Some(slope) = s.slope {
            if slope <= 5.0 {
                horizontal += 1;
            } else {
                slanted += 1;
            }
        }
    }
    (horizontal, slanted)
}

fn classify_roof_type(roof_planes: usize, horizontal: usize, slanted: usize) -> String {
    if roof_planes == 0 {
        "no points".to_string()
    } else if horizontal == 1 && slanted == 0 {
        "horizontal".to_string()
    } else if horizontal > 1 && slanted == 0 {
        "multiple horizontal".to_string()
    } else if slanted > 0 {
        "slanted".to_string()
    } else {
        "no planes".to_string()
    }
}

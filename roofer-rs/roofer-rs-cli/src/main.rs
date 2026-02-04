//! Roofer CLI - Building reconstruction from LiDAR point clouds.
//!
//! This is the Rust implementation of roofer for automatic LoD2 building
//! reconstruction.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use gdal::spatial_ref::{CoordTransform, SpatialRef};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use rstar::RTree;
use serde_json;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::{info, warn};

use roofer_rs_geometry::plane::RansacConfig;
use roofer_rs_geometry::point_cloud::PointCloud;
use roofer_rs_geometry::polygon::Footprint;
use roofer_rs_geometry::reconstruction::{BuildingReconstructor, Lod, ReconstructionConfig};
use roofer_rs_io::cityjson::{CityJsonTransform, CityJsonWriter};
use roofer_rs_io::config::Config;
use roofer_rs_io::las_reader::LasReader;

use roofer_rs_io::VectorReader;

#[derive(Debug, Clone, Default)]
struct PlaneOverrides {
    plane_detect_epsilon: Option<f64>,
    plane_detect_k: Option<usize>,
    plane_detect_min_points: Option<usize>,
    plane_metrics_is_wall_threshold: Option<f64>,
    plane_metrics_is_horizontal_threshold: Option<f64>,
    plane_merge_angle_degrees: Option<f64>,
    plane_merge_distance: Option<f64>,
    plane_merge_distance_horizontal: Option<f64>,
    plane_merge_inlier_overlap: Option<f64>,
    plane_min_inlier_ratio: Option<f64>,
    plane_small_plane_max_inliers: Option<usize>,
    plane_merge_neighbor_distance: Option<f64>,
}

#[derive(Parser)]
#[command(name = "roofer-rs")]
#[command(about = "Automatic 3D building reconstruction from LiDAR", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    /// Quiet mode (suppress output)
    #[arg(short, long, global = true)]
    quiet: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the full reconstruction pipeline
    Run {
        /// Path to configuration file (TOML)
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Path to building footprints (GeoPackage, Shapefile, etc.)
        #[arg(short = 'f', long)]
        footprints: Option<PathBuf>,

        /// Path to point cloud file(s) (LAS/LAZ)
        #[arg(short = 'p', long)]
        pointcloud: Option<Vec<PathBuf>>,

        /// Output directory
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Output CRS for CityJSON (e.g. "EPSG:4326")
        #[arg(long)]
        output_srs: Option<String>,

        /// Exclude process/debug attributes from output
        #[arg(long)]
        exclude_process_attributes: bool,

        /// Split CityJSON Sequence per building into features/
        #[arg(long)]
        split_cjseq: bool,

        /// Plane detect epsilon (RANSAC inlier threshold)
        #[arg(long)]
        plane_detect_epsilon: Option<f64>,

        /// Plane detect k (neighbors)
        #[arg(long)]
        plane_detect_k: Option<usize>,

        /// Plane detect minimum points
        #[arg(long)]
        plane_detect_min_points: Option<usize>,

        /// Threshold for classifying a plane as vertical (wall) based on |n·z|
        #[arg(long)]
        plane_metrics_is_wall_threshold: Option<f64>,

        /// Threshold for classifying a plane as horizontal based on |n·z|
        #[arg(long)]
        plane_metrics_is_horizontal_threshold: Option<f64>,

        /// Angle threshold (degrees) for merging similar planes
        #[arg(long)]
        plane_merge_angle_degrees: Option<f64>,

        /// Distance threshold for merging similar planes
        #[arg(long)]
        plane_merge_distance: Option<f64>,

        /// Distance threshold for merging horizontal planes
        #[arg(long)]
        plane_merge_distance_horizontal: Option<f64>,

        /// Minimum inlier overlap ratio for merging planes
        #[arg(long)]
        plane_merge_inlier_overlap: Option<f64>,

        /// Minimum inlier ratio for accepting a plane (relative to remaining points)
        #[arg(long)]
        plane_min_inlier_ratio: Option<f64>,

        /// Max inliers to treat a plane as "small" for merging
        #[arg(long)]
        plane_small_plane_max_inliers: Option<usize>,

        /// Neighbor distance threshold for merging small planes
        #[arg(long)]
        plane_merge_neighbor_distance: Option<f64>,

        /// Minimum height difference to create internal wall (meters).
        /// If 0.0 or not set, create walls for all edges regardless of height difference.
        #[arg(long)]
        wall_min_height_diff: Option<f64>,

        /// Snap tolerance exponent (10^-exp) for filtering short edges.
        #[arg(long)]
        wall_snap_tolerance_exp: Option<i32>,

        /// Build walls for all edges, not just height-different edges.
        #[arg(long)]
        wall_build_all_edges: Option<bool>,

        /// Minimum edge length to create wall (meters).
        #[arg(long)]
        wall_min_edge_length: Option<f64>,

        /// Number of parallel reconstruction threads
        #[arg(short = 'j', long, default_value = "0")]
        jobs: usize,
    },

    /// Crop point clouds to building footprints
    Crop {
        /// Path to building footprints
        #[arg(short = 'f', long)]
        footprints: PathBuf,

        /// Path to point cloud file(s)
        #[arg(short = 'p', long)]
        pointcloud: Vec<PathBuf>,

        /// Output directory for cropped point clouds
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Reconstruct a single building from cropped point cloud
    Reconstruct {
        /// Path to building footprint (single feature)
        #[arg(short = 'f', long)]
        footprint: PathBuf,

        /// Path to cropped point cloud
        #[arg(short = 'p', long)]
        pointcloud: PathBuf,

        /// Output CityJSON file
        #[arg(short, long)]
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Setup logging
    let log_level = if cli.quiet {
        tracing::Level::ERROR
    } else {
        match cli.verbose {
            0 => tracing::Level::INFO,
            1 => tracing::Level::DEBUG,
            _ => tracing::Level::TRACE,
        }
    };

    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .with_target(false)
        .init();

    match cli.command {
        Commands::Run {
            config,
            footprints,
            pointcloud,
            output,
            output_srs,
            exclude_process_attributes,
            split_cjseq,
            plane_detect_epsilon,
            plane_detect_k,
            plane_detect_min_points,
            plane_metrics_is_wall_threshold,
            plane_metrics_is_horizontal_threshold,
            plane_merge_angle_degrees,
            plane_merge_distance,
            plane_merge_distance_horizontal,
            plane_merge_inlier_overlap,
            plane_min_inlier_ratio,
            plane_small_plane_max_inliers,
            plane_merge_neighbor_distance,
            wall_min_height_diff,
            wall_snap_tolerance_exp,
            wall_build_all_edges,
            wall_min_edge_length,
            jobs,
        } => run_pipeline(
            config,
            footprints,
            pointcloud,
            output,
            output_srs,
            Some(!exclude_process_attributes),
            split_cjseq,
            PlaneOverrides {
                plane_detect_epsilon,
                plane_detect_k,
                plane_detect_min_points,
                plane_metrics_is_wall_threshold,
                plane_metrics_is_horizontal_threshold,
                plane_merge_angle_degrees,
                plane_merge_distance,
                plane_merge_distance_horizontal,
                plane_merge_inlier_overlap,
                plane_min_inlier_ratio,
                plane_small_plane_max_inliers,
                plane_merge_neighbor_distance,
            },
            wall_min_height_diff,
            wall_snap_tolerance_exp,
            wall_build_all_edges,
            wall_min_edge_length,
            jobs,
        ),
        Commands::Crop {
            footprints,
            pointcloud,
            output,
        } => crop_pointclouds(&footprints, &pointcloud, &output),
        Commands::Reconstruct {
            footprint,
            pointcloud,
            output,
        } => reconstruct_single(&footprint, &pointcloud, &output),
    }
}

/// Run the full reconstruction pipeline.
fn run_pipeline(
    config_path: Option<PathBuf>,
    footprints_path: Option<PathBuf>,
    pointcloud_paths: Option<Vec<PathBuf>>,
    output_path: Option<PathBuf>,
    output_srs: Option<String>,
    include_process_attributes: Option<bool>,
    split_cjseq: bool,
    plane_overrides: PlaneOverrides,
    wall_min_height_diff: Option<f64>,
    wall_snap_tolerance_exp: Option<i32>,
    wall_build_all_edges: Option<bool>,
    wall_min_edge_length: Option<f64>,
    jobs: usize,
) -> Result<()> {
    // Load configuration
    let config = if let Some(ref path) = config_path {
        info!("Loading configuration from {}", path.display());
        Config::from_file(path).context("Failed to load configuration")?
    } else {
        // Build config from CLI arguments
        let footprints = footprints_path.context("--footprints is required when no config file is provided")?;
        let pointclouds = pointcloud_paths.context("--pointcloud is required when no config file is provided")?;
        let output = output_path.context("--output is required when no config file is provided")?;

        Config {
            polygon_source: footprints,
            output_directory: output,
            pointclouds: vec![roofer_rs_io::config::PointCloudConfig {
                name: "input".to_string(),
                source: pointclouds,
            }],
            ..default_config()
        }
    };

    // Allow CLI to override config output CRS
    let output_srs = output_srs.or_else(|| config.output_srs.clone());
    let include_process_attributes =
        include_process_attributes.unwrap_or(config.include_process_attributes);
    let split_cjseq = if split_cjseq {
        true
    } else {
        config.split_cjseq
    };

    let plane_detect_epsilon = plane_overrides
        .plane_detect_epsilon
        .unwrap_or(config.plane_detect_epsilon);
    let plane_detect_k = plane_overrides.plane_detect_k.unwrap_or(config.plane_detect_k);
    let plane_detect_min_points = plane_overrides
        .plane_detect_min_points
        .unwrap_or(config.plane_detect_min_points);
    let plane_metrics_is_wall_threshold = plane_overrides
        .plane_metrics_is_wall_threshold
        .unwrap_or(config.plane_metrics_is_wall_threshold);
    let plane_metrics_is_horizontal_threshold = plane_overrides
        .plane_metrics_is_horizontal_threshold
        .unwrap_or(config.plane_metrics_is_horizontal_threshold);
    let plane_merge_angle_degrees = plane_overrides
        .plane_merge_angle_degrees
        .unwrap_or(config.plane_merge_angle_degrees);
    let plane_merge_distance = plane_overrides
        .plane_merge_distance
        .unwrap_or(config.plane_merge_distance);
    let plane_merge_distance_horizontal = plane_overrides
        .plane_merge_distance_horizontal
        .unwrap_or(config.plane_merge_distance_horizontal);
    let plane_merge_inlier_overlap = plane_overrides
        .plane_merge_inlier_overlap
        .unwrap_or(config.plane_merge_inlier_overlap);
    let plane_min_inlier_ratio = plane_overrides
        .plane_min_inlier_ratio
        .unwrap_or(config.plane_min_inlier_ratio);
    let plane_small_plane_max_inliers = plane_overrides
        .plane_small_plane_max_inliers
        .unwrap_or(config.plane_small_plane_max_inliers);
    let plane_merge_neighbor_distance = plane_overrides
        .plane_merge_neighbor_distance
        .unwrap_or(config.plane_merge_neighbor_distance);

    // Set up thread pool
    if jobs > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build_global()
            .ok();
    }

    // Create output directory
    std::fs::create_dir_all(&config.output_directory)?;

    // Read footprints
    info!("Reading footprints from {}", config.polygon_source.display());
    let vector_reader = VectorReader::new()
        .with_id_attribute(config.id_attribute.clone().unwrap_or_else(|| "fid".to_string()));

    let pc_paths = config.point_cloud_paths();
    let las_reader = LasReader::new();

    // Resolve CRS: config > footprints (vector) > point cloud (LAS/LAZ header)
    let crs = if config.srs.is_some() {
        config.srs.clone()
    } else {
        vector_reader
            .read_crs(&config.polygon_source)
            .context("Failed to read CRS from footprints")?
    };
    let crs = crs.or_else(|| {
        las_reader
            .read_crs_from_first(&pc_paths)
            .ok()
            .flatten()
    });

    let footprints = vector_reader
        .read_footprints(&config.polygon_source)
        .context("Failed to read footprints")?;

    info!("Loaded {} footprints", footprints.len());

    if footprints.is_empty() {
        warn!("No footprints found, nothing to process");
        return Ok(());
    }

    if pc_paths.is_empty() {
        anyhow::bail!("No point cloud files found");
    }

    info!("Found {} point cloud file(s)", pc_paths.len());

    // Compute footprint extent and data offset (first vertex of first footprint, like C++)
    let footprint_extent = compute_footprint_extent(&footprints);
    let data_offset = footprints
        .first()
        .and_then(|fp| fp.polygon.exterior.vertices.first())
        .map(|v| [v.x, v.y, 0.0])
        .unwrap_or([footprint_extent.0, footprint_extent.1, 0.0]);

    // Optional output reprojection (input CRS -> output CRS)
    let mut coord_transform: Option<CoordTransform> = None;
    let mut output_extent = footprint_extent;
    let mut translate = config.cj_translate.unwrap_or(data_offset);
    let mut reference_srs = crs.clone();

    if let (Some(ref input_srs), Some(ref output_srs)) = (&crs, &output_srs) {
        if normalize_epsg(input_srs) != normalize_epsg(output_srs) {
            let transform = build_coord_transform(input_srs, output_srs)
                .context("Failed to build output CRS transform")?;
            translate = transform_point(&transform, translate)
                .context("Failed to transform CityJSON translate")?;
            output_extent = transform_extent(&transform, footprint_extent)
                .context("Failed to transform CityJSON extent")?;
            coord_transform = Some(transform);
            reference_srs = Some(output_srs.clone());
        }
    } else if output_srs.is_some() && crs.is_none() {
        warn!("Output CRS specified but input CRS is unknown; skipping reprojection");
    }

    // Read all points (for now, simple approach - read all into memory)
    info!("Reading point cloud...");
    let point_cloud = las_reader
        .read_files(&pc_paths)
        .context("Failed to read point cloud")?;

    info!("Loaded {} points", point_cloud.len());

    // Build spatial index of footprints
    info!("Building spatial index...");
    let footprint_tree: RTree<Footprint> = RTree::bulk_load(footprints);

    // Compute transform: use config override or data offset from first footprint (like C++)
    let transform = CityJsonTransform {
        scale: config.cj_scale,
        translate,
    };

    // Set up wall configuration
    use roofer_rs_geometry::reconstruction::WallConfig;
    let wall_config = WallConfig {
        min_internal_wall_height_diff: wall_min_height_diff.or(config.wall_min_height_diff),
        snap_tolerance_exp: wall_snap_tolerance_exp.unwrap_or(config.wall_snap_tolerance_exp),
        build_all_edges: wall_build_all_edges.unwrap_or(config.wall_build_all_edges),
        min_edge_length: wall_min_edge_length.unwrap_or(config.wall_min_edge_length),
    };

    // Set up reconstruction
    let recon_config = ReconstructionConfig {
        plane_config: RansacConfig {
            epsilon: plane_detect_epsilon,
            k: plane_detect_k,
            min_points: plane_detect_min_points,
            max_iterations: RansacConfig::default().max_iterations,
            metrics_is_wall_threshold: plane_metrics_is_wall_threshold,
            metrics_is_horizontal_threshold: plane_metrics_is_horizontal_threshold,
            merge_angle_degrees: plane_merge_angle_degrees,
            merge_distance: plane_merge_distance,
            merge_distance_horizontal: plane_merge_distance_horizontal,
            merge_inlier_overlap: plane_merge_inlier_overlap,
            min_inlier_ratio: plane_min_inlier_ratio,
            small_plane_max_inliers: plane_small_plane_max_inliers,
            merge_neighbor_distance: plane_merge_neighbor_distance,
        },
        lod13_step_height: config.lod13_step_height,
        complexity_factor: config.complexity_factor,
        lods: match config.lod {
            Some(12) => vec![Lod::Lod12],
            Some(13) => vec![Lod::Lod12, Lod::Lod13],
            Some(22) => vec![Lod::Lod12, Lod::Lod22],
            _ => vec![Lod::Lod12, Lod::Lod13, Lod::Lod22],
        },
        wall_config,
    };

    let reconstructor = BuildingReconstructor::new(recon_config);

    // Create output file
    let output_file = config.output_directory.join("output.city.jsonl");
    let mut writer = CityJsonWriter::new(&output_file, transform.clone())?
        .with_process_attributes(include_process_attributes);
    
    // Set reference system from input file or config
    if let Some(ref srs) = reference_srs {
        info!("Using CRS: {}", srs);
        writer = writer.with_reference_system(srs);
    } else {
        warn!("No CRS found - output will not have referenceSystem metadata");
    }

    if let Some(transform) = coord_transform {
        writer = writer.with_coord_transform(transform);
    }

    if split_cjseq {
        let metadata_path = config.output_directory.join("metadata.city.json");
        let mut metadata_writer = CityJsonWriter::new(&metadata_path, transform.clone())?
            .with_process_attributes(include_process_attributes);
        if let Some(ref srs) = reference_srs {
            metadata_writer = metadata_writer.with_reference_system(srs);
        }
        metadata_writer.write_header(Some([
            output_extent.0,
            output_extent.1,
            0.0,
            output_extent.2,
            output_extent.3,
            0.0,
        ]))?;
        metadata_writer.finish()?;
        std::fs::create_dir_all(config.output_directory.join("features"))?;
    }

    // Write header with footprint extent (like C++, z=0 for 2D footprints)
    writer.write_header(Some([
        output_extent.0,
        output_extent.1,
        0.0,
        output_extent.2,
        output_extent.3,
        0.0,
    ]))?;

    // Process footprints
    info!("Reconstructing buildings...");
    let progress = ProgressBar::new(footprint_tree.size() as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
            .progress_chars("#>-"),
    );

    let success_count = AtomicUsize::new(0);
    let fail_count = AtomicUsize::new(0);

    // Collect results
    let results: Vec<_> = footprint_tree
        .iter()
        .collect::<Vec<_>>()
        .par_iter()
        .map(|footprint| {
            // Crop points for this footprint
            let cropped = crop_points_for_footprint(&point_cloud, footprint);

            // Reconstruct
            let result = reconstructor.reconstruct(footprint, &cropped);

            progress.inc(1);

            if result.success {
                success_count.fetch_add(1, Ordering::Relaxed);
            } else {
                fail_count.fetch_add(1, Ordering::Relaxed);
            }

            result
        })
        .collect();

    progress.finish();

    // Write results (sequential to maintain order)
    for result in results {
        if result.success || result.geometry.has_geometry() {
            writer.write_feature(&result.geometry)?;
            if split_cjseq {
                let feature_value = writer.build_feature_value(&result.geometry)?;
                let feature_path = config
                    .output_directory
                    .join("features")
                    .join(format!("{}.jsonl", result.geometry.id));
                let mut feature_file = std::io::BufWriter::new(std::fs::File::create(feature_path)?);
                let line = serde_json::to_string(&feature_value)?;
                writeln!(feature_file, "{}", line)?;
            }
        }
    }

    writer.finish()?;

    let success = success_count.load(Ordering::Relaxed);
    let fail = fail_count.load(Ordering::Relaxed);

    info!(
        "Reconstruction complete: {} succeeded, {} failed",
        success, fail
    );
    info!("Output written to {}", output_file.display());

    Ok(())
}

/// Crop point cloud points that fall within a footprint.
fn crop_points_for_footprint(point_cloud: &PointCloud, footprint: &Footprint) -> PointCloud {
    let mut cropped = PointCloud::new();

    for (i, point) in point_cloud.positions.iter().enumerate() {
        if footprint.contains_2d(point.x, point.y) {
            if let Some(ref classification) = point_cloud.classification {
                cropped.push_full(
                    *point,
                    point_cloud.intensity.as_ref().map(|v| v[i]).unwrap_or(0),
                    classification[i],
                    point_cloud.return_number.as_ref().map(|v| v[i]).unwrap_or(0),
                    point_cloud.number_of_returns.as_ref().map(|v| v[i]).unwrap_or(0),
                );
            } else {
                cropped.push(*point);
            }
        }
    }

    cropped
}

/// Crop point clouds to building footprints.
fn crop_pointclouds(
    footprints_path: &PathBuf,
    pointcloud_paths: &[PathBuf],
    output_path: &PathBuf,
) -> Result<()> {
    info!("Crop subcommand not yet fully implemented");
    info!("Footprints: {}", footprints_path.display());
    info!("Point clouds: {:?}", pointcloud_paths);
    info!("Output: {}", output_path.display());

    // TODO: Implement crop-only mode
    Ok(())
}

/// Reconstruct a single building.
fn reconstruct_single(
    footprint_path: &PathBuf,
    pointcloud_path: &PathBuf,
    output_path: &PathBuf,
) -> Result<()> {
    info!("Reading footprint from {}", footprint_path.display());
    let vector_reader = VectorReader::new();
    let footprints = vector_reader
        .read_footprints(footprint_path)
        .context("Failed to read footprint")?;

    let footprint = footprints
        .into_iter()
        .next()
        .context("No footprint found in file")?;

    info!("Reading point cloud from {}", pointcloud_path.display());
    let las_reader = LasReader::new();
    let point_cloud = las_reader
        .read_file(pointcloud_path)
        .context("Failed to read point cloud")?;

    info!("Loaded {} points", point_cloud.len());

    // Reconstruct
    let config = ReconstructionConfig::default();
    let reconstructor = BuildingReconstructor::new(config);
    let result = reconstructor.reconstruct(&footprint, &point_cloud);

    if result.success {
        info!("Reconstruction successful");
    } else {
        warn!(
            "Reconstruction failed: {}",
            result.error_message.unwrap_or_default()
        );
    }

    // Write output
    let stats = point_cloud.compute_statistics();
    let transform = CityJsonTransform {
        scale: [0.001, 0.001, 0.001],
        translate: [stats.bounds_min.x, stats.bounds_min.y, 0.0],
    };

    let mut writer = CityJsonWriter::new(output_path, transform)?;
    writer.write_header(Some([
        stats.bounds_min.x,
        stats.bounds_min.y,
        stats.bounds_min.z,
        stats.bounds_max.x,
        stats.bounds_max.y,
        stats.bounds_max.z,
    ]))?;
    writer.write_feature(&result.geometry)?;
    writer.finish()?;

    info!("Output written to {}", output_path.display());
    Ok(())
}

/// Compute the 2D bounding box of all footprints: (min_x, min_y, max_x, max_y).
fn compute_footprint_extent(footprints: &[Footprint]) -> (f64, f64, f64, f64) {
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;

    for fp in footprints {
        for v in &fp.polygon.exterior.vertices {
            min_x = min_x.min(v.x);
            min_y = min_y.min(v.y);
            max_x = max_x.max(v.x);
            max_y = max_y.max(v.y);
        }
    }

    (min_x, min_y, max_x, max_y)
}

fn normalize_epsg(srs: &str) -> Option<u32> {
    let trimmed = srs.trim();
    let code = trimmed.strip_prefix("EPSG:").unwrap_or(trimmed);
    code.parse::<u32>().ok()
}

fn build_coord_transform(input_srs: &str, output_srs: &str) -> Result<CoordTransform> {
    let input_epsg =
        normalize_epsg(input_srs).context("Input CRS must be EPSG:<code> or <code>")?;
    let output_epsg =
        normalize_epsg(output_srs).context("Output CRS must be EPSG:<code> or <code>")?;
    let source = SpatialRef::from_epsg(input_epsg)?;
    let target = SpatialRef::from_epsg(output_epsg)?;
    Ok(CoordTransform::new(&source, &target)?)
}

fn transform_point(transform: &CoordTransform, point: [f64; 3]) -> Result<[f64; 3]> {
    let mut xs = vec![point[0]];
    let mut ys = vec![point[1]];
    let mut zs = vec![point[2]];
    transform.transform_coords(&mut xs, &mut ys, &mut zs)?;
    Ok([xs[0], ys[0], zs[0]])
}

fn transform_extent(
    transform: &CoordTransform,
    extent: (f64, f64, f64, f64),
) -> Result<(f64, f64, f64, f64)> {
    let (min_x, min_y, max_x, max_y) = extent;
    let mut xs = vec![min_x, max_x, min_x, max_x];
    let mut ys = vec![min_y, min_y, max_y, max_y];
    let mut zs = vec![0.0; 4];
    transform.transform_coords(&mut xs, &mut ys, &mut zs)?;
    let out_min_x = xs.iter().cloned().fold(f64::MAX, f64::min);
    let out_min_y = ys.iter().cloned().fold(f64::MAX, f64::min);
    let out_max_x = xs.iter().cloned().fold(f64::MIN, f64::max);
    let out_max_y = ys.iter().cloned().fold(f64::MIN, f64::max);
    Ok((out_min_x, out_min_y, out_max_x, out_max_y))
}

/// Create a default configuration.
fn default_config() -> Config {
    Config {
        polygon_source: PathBuf::new(),
        polygon_source_layer: None,
        id_attribute: None,
        force_lod11_attribute: None,
        filter: None,
        srs: None,
        output_srs: None,
        roi_box: None,
        ceil_point_density: 20.0,
        tilesize: [1000.0, 1000.0],
        cellsize: 0.5,
        plane_detect_epsilon: 0.3,
        plane_detect_k: 15,
        plane_detect_min_points: 15,
        plane_metrics_is_wall_threshold: 0.3,
        plane_metrics_is_horizontal_threshold: 0.995,
        plane_merge_angle_degrees: 7.5,
        plane_merge_distance: 0.5,
        plane_merge_distance_horizontal: 1.5,
        plane_merge_inlier_overlap: 0.6,
        plane_min_inlier_ratio: 0.04,
        plane_small_plane_max_inliers: 60,
        plane_merge_neighbor_distance: 2.0,
        lod13_step_height: 3.0,
        complexity_factor: 0.7,
        lod: None,
        split_cjseq: false,
        omit_metadata: false,
        cj_translate: None,
        cj_scale: [0.001, 0.001, 0.001],
        output_directory: PathBuf::new(),
        pointclouds: Vec::new(),
        output_attributes: roofer_rs_io::config::OutputAttributes::with_defaults(),
        include_process_attributes: true,
        wall_min_height_diff: None,
        wall_snap_tolerance_exp: 4,
        wall_build_all_edges: true,
        wall_min_edge_length: 0.0,
    }
}

//! I/O layer for Roofer.
//!
//! This crate provides:
//! - LAS/LAZ point cloud reading
//! - Vector file reading (GeoPackage, Shapefile) via GDAL
//! - CityJSON output (JSONL format)
//! - FCB output (binary CityJSON)
//! - GeoParquet output
//! - Configuration file parsing

pub mod cityjson;
pub mod config;
pub mod las_reader;
pub mod vector_reader;

pub use cityjson::{validate_cityjson_feature_ids, CityJsonWriter};
pub use config::Config;
pub use las_reader::LasReader;
pub use vector_reader::VectorReader;

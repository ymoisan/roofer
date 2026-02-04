//! Core geometry, data structures, and reconstruction algorithms for Roofer.
//!
//! This crate provides:
//! - Point cloud data structures with optional Arrow-backed columnar storage
//! - 2D/3D geometry types (polygons, meshes, planes)
//! - Reconstruction algorithms (plane detection, triangulation, arrangement building)
//! - Attribute handling for CityJSON-compatible output

pub mod attributes;
pub mod mesh;
pub mod plane;
pub mod point_cloud;
pub mod polygon;
pub mod reconstruction;

pub use attributes::{AttributeMap, AttributeValue};
pub use mesh::Mesh;
pub use plane::Plane;
pub use point_cloud::PointCloud;
pub use polygon::{LinearRing, Polygon3D};

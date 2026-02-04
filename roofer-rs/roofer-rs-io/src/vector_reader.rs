//! Vector file reader using GDAL.
//!
//! Reads building footprints from various vector formats (GeoPackage, Shapefile, etc.).

use gdal::vector::{Geometry, LayerAccess};
use gdal::Dataset;
use nalgebra::Point3;
use roofer_rs_geometry::polygon::{Footprint, LinearRing, Polygon3D};
use roofer_rs_geometry::AttributeMap;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VectorError {
    #[error("GDAL error: {0}")]
    GdalError(#[from] gdal::errors::GdalError),
    #[error("File not found: {0}")]
    FileNotFound(String),
    #[error("Layer not found: {0}")]
    LayerNotFound(String),
    #[error("Invalid geometry: {0}")]
    InvalidGeometry(String),
    #[error("Missing ID attribute: {0}")]
    MissingIdAttribute(String),
}

/// Reader for vector files (GeoPackage, Shapefile, etc.).
pub struct VectorReader {
    /// Layer name to read (None = first layer)
    pub layer_name: Option<String>,
    /// Attribute to use as building ID
    pub id_attribute: Option<String>,
    /// OGR SQL WHERE clause
    pub filter: Option<String>,
    /// Spatial filter box [min_x, min_y, max_x, max_y]
    pub spatial_filter: Option<[f64; 4]>,
}

impl Default for VectorReader {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorReader {
    pub fn new() -> Self {
        Self {
            layer_name: None,
            id_attribute: None,
            filter: None,
            spatial_filter: None,
        }
    }

    pub fn with_layer(mut self, layer_name: impl Into<String>) -> Self {
        self.layer_name = Some(layer_name.into());
        self
    }

    pub fn with_id_attribute(mut self, attr: impl Into<String>) -> Self {
        self.id_attribute = Some(attr.into());
        self
    }

    pub fn with_filter(mut self, filter: impl Into<String>) -> Self {
        self.filter = Some(filter.into());
        self
    }

    pub fn with_spatial_filter(mut self, bbox: [f64; 4]) -> Self {
        self.spatial_filter = Some(bbox);
        self
    }

    /// Read the coordinate reference system from a vector file.
    /// Returns the EPSG code as a string (e.g., "2961") if available.
    pub fn read_crs(&self, path: &Path) -> Result<Option<String>, VectorError> {
        if !path.exists() {
            return Err(VectorError::FileNotFound(path.display().to_string()));
        }

        let dataset = Dataset::open(path)?;
        let layer = if let Some(ref name) = self.layer_name {
            dataset
                .layer_by_name(name)
                .map_err(|_| VectorError::LayerNotFound(name.clone()))?
        } else {
            dataset.layer(0)?
        };

        // Get the spatial reference from the layer
        if let Some(srs) = layer.spatial_ref() {
            // Try to get the EPSG code
            if let Ok(epsg) = srs.auth_code() {
                tracing::info!("Found CRS: EPSG:{}", epsg);
                return Ok(Some(epsg.to_string()));
            }
            // Fallback: try to identify from WKT
            tracing::warn!("Could not determine EPSG code from spatial reference");
        } else {
            tracing::warn!("No spatial reference found in vector file");
        }

        Ok(None)
    }

    /// Read footprints from a vector file.
    pub fn read_footprints(&self, path: &Path) -> Result<Vec<Footprint>, VectorError> {
        if !path.exists() {
            return Err(VectorError::FileNotFound(path.display().to_string()));
        }

        let dataset = Dataset::open(path)?;
        let mut layer = if let Some(ref name) = self.layer_name {
            dataset
                .layer_by_name(name)
                .map_err(|_| VectorError::LayerNotFound(name.clone()))?
        } else {
            dataset.layer(0)?
        };

        // Apply attribute filter
        if let Some(ref filter) = self.filter {
            layer.set_attribute_filter(filter)?;
        }

        // Apply spatial filter
        if let Some(bbox) = self.spatial_filter {
            layer.set_spatial_filter_rect(bbox[0], bbox[1], bbox[2], bbox[3]);
        }

        let mut footprints = Vec::new();
        let mut feature_id = 0u64;

        for feature in layer.features() {
            // Get ID
            let id = if let Some(ref id_attr) = self.id_attribute {
                feature
                    .field_as_string_by_name(id_attr)
                    .ok()
                    .flatten()
                    .unwrap_or_else(|| feature_id.to_string())
            } else {
                feature.fid().map(|f| f.to_string()).unwrap_or_else(|| feature_id.to_string())
            };

            // Get geometry
            let geometry = match feature.geometry() {
                Some(g) => g,
                None => continue,
            };

            // Convert to Polygon3D
            let polygon = self.geometry_to_polygon3d(geometry)?;
            if polygon.is_empty() {
                continue;
            }

            // Get attributes
            let mut attributes = AttributeMap::new();
            for (name, value) in feature.fields() {
                match value {
                    Some(gdal::vector::FieldValue::IntegerValue(v)) => {
                        attributes.insert(name, v as i64);
                    }
                    Some(gdal::vector::FieldValue::Integer64Value(v)) => {
                        attributes.insert(name, v);
                    }
                    Some(gdal::vector::FieldValue::RealValue(v)) => {
                        attributes.insert(name, v);
                    }
                    Some(gdal::vector::FieldValue::StringValue(v)) => {
                        attributes.insert(name, v);
                    }
                    _ => {}
                }
            }

            let footprint = Footprint::new(id, polygon).with_attributes(attributes);
            footprints.push(footprint);
            feature_id += 1;
        }

        tracing::info!("Read {} footprints from {}", footprints.len(), path.display());
        Ok(footprints)
    }

    /// Convert GDAL Geometry to Polygon3D.
    fn geometry_to_polygon3d(&self, geometry: &Geometry) -> Result<Polygon3D, VectorError> {
        let geom_type = geometry.geometry_type();

        match geom_type {
            gdal::vector::OGRwkbGeometryType::wkbPolygon
            | gdal::vector::OGRwkbGeometryType::wkbPolygon25D
            | gdal::vector::OGRwkbGeometryType::wkbPolygonM
            | gdal::vector::OGRwkbGeometryType::wkbPolygonZM => {
                self.polygon_geometry_to_polygon3d(geometry)
            }
            gdal::vector::OGRwkbGeometryType::wkbMultiPolygon
            | gdal::vector::OGRwkbGeometryType::wkbMultiPolygon25D => {
                // Use first polygon from multi-polygon
                if geometry.geometry_count() > 0 {
                    let first = geometry.get_geometry(0);
                    self.polygon_geometry_to_polygon3d(&first)
                } else {
                    Ok(Polygon3D::default())
                }
            }
            _ => Err(VectorError::InvalidGeometry(format!(
                "Unsupported geometry type: {:?}",
                geom_type
            ))),
        }
    }

    /// Convert a GDAL polygon geometry to Polygon3D.
    fn polygon_geometry_to_polygon3d(&self, geometry: &Geometry) -> Result<Polygon3D, VectorError> {
        let ring_count = geometry.geometry_count();
        if ring_count == 0 {
            return Ok(Polygon3D::default());
        }

        // Exterior ring
        let ext_ring = geometry.get_geometry(0);
        let exterior = self.ring_to_linear_ring(&ext_ring)?;

        // Interior rings (holes)
        let mut interiors = Vec::new();
        for i in 1..ring_count {
            let int_ring = geometry.get_geometry(i);
            let interior = self.ring_to_linear_ring(&int_ring)?;
            interiors.push(interior);
        }

        Ok(Polygon3D::with_interiors(exterior, interiors))
    }

    /// Convert a GDAL ring to LinearRing.
    fn ring_to_linear_ring(&self, geometry: &Geometry) -> Result<LinearRing, VectorError> {
        let points = geometry.get_point_vec();
        let vertices: Vec<Point3<f64>> = points
            .into_iter()
            .map(|(x, y, z)| Point3::new(x, y, z))
            .collect();

        Ok(LinearRing::from_vertices(vertices))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require actual test data files
    // They are skipped if the files don't exist

    #[test]
    fn test_vector_reader_creation() {
        let reader = VectorReader::new()
            .with_layer("buildings")
            .with_id_attribute("fid")
            .with_filter("area > 100");

        assert_eq!(reader.layer_name, Some("buildings".to_string()));
        assert_eq!(reader.id_attribute, Some("fid".to_string()));
        assert_eq!(reader.filter, Some("area > 100".to_string()));
    }
}

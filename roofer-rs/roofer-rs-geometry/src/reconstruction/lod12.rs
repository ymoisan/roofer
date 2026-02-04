//! LoD 1.2 reconstruction (simple extrusion).
//!
//! Creates a building solid by extruding the footprint to a flat roof.

use crate::mesh::{Face, Mesh, SemanticSurface};
use crate::point_cloud::PointCloud;
use crate::polygon::Footprint;
use nalgebra::Point3;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Lod12Error {
    #[error("Empty footprint")]
    EmptyFootprint,
    #[error("Empty point cloud")]
    EmptyPointCloud,
    #[error("Invalid geometry: {0}")]
    InvalidGeometry(String),
}

/// Reconstructor for LoD 1.2 (extruded footprint with flat roof).
pub struct Lod12Reconstructor {}

impl Lod12Reconstructor {
    pub fn new() -> Self {
        Self {}
    }

    /// Reconstruct LoD 1.2 geometry.
    pub fn reconstruct(
        &self,
        footprint: &Footprint,
        point_cloud: &PointCloud,
        h_ground: f64,
    ) -> Result<Mesh, Lod12Error> {
        if footprint.polygon.is_empty() {
            return Err(Lod12Error::EmptyFootprint);
        }

        if point_cloud.is_empty() {
            return Err(Lod12Error::EmptyPointCloud);
        }

        // Compute roof height (use 70th percentile)
        let stats = point_cloud.compute_statistics();
        let h_roof = stats.z_70p;

        self.extrude_footprint(footprint, h_ground, h_roof)
    }

    /// Extrude a footprint to create a solid.
    pub fn extrude_footprint(
        &self,
        footprint: &Footprint,
        h_ground: f64,
        h_roof: f64,
    ) -> Result<Mesh, Lod12Error> {
        let mut mesh = Mesh::new();

        // Add semantic surfaces
        let ground_idx = mesh.add_semantic(SemanticSurface::ground());
        let wall_idx = mesh.add_semantic(SemanticSurface::wall(true));
        let roof_idx = mesh.add_semantic(SemanticSurface::roof());

        let exterior = &footprint.polygon.exterior;
        if exterior.len() < 4 {
            return Err(Lod12Error::InvalidGeometry(
                "Exterior ring too short".to_string(),
            ));
        }

        let n = exterior.len() - 1; // Exclude closing vertex

        // Add bottom vertices (ground level)
        let mut bottom_indices = Vec::with_capacity(n);
        for i in 0..n {
            let v = &exterior.vertices[i];
            let idx = mesh.add_vertex(Point3::new(v.x, v.y, h_ground));
            bottom_indices.push(idx);
        }

        // Add top vertices (roof level)
        let mut top_indices = Vec::with_capacity(n);
        for i in 0..n {
            let v = &exterior.vertices[i];
            let idx = mesh.add_vertex(Point3::new(v.x, v.y, h_roof));
            top_indices.push(idx);
        }

        // Add ground face (reversed winding for downward normal)
        let ground_face = Face::new(bottom_indices.iter().rev().cloned().collect())
            .with_semantic(ground_idx);
        mesh.add_face(ground_face);

        // Add roof face
        let roof_face = Face::new(top_indices.clone()).with_semantic(roof_idx);
        mesh.add_face(roof_face);

        // Add wall faces
        for i in 0..n {
            let j = (i + 1) % n;
            let wall_face = Face::new(vec![
                bottom_indices[i],
                bottom_indices[j],
                top_indices[j],
                top_indices[i],
            ])
            .with_semantic(wall_idx);
            mesh.add_face(wall_face);
        }

        Ok(mesh)
    }
}

impl Default for Lod12Reconstructor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polygon::{LinearRing, Polygon3D};
    use crate::AttributeMap;

    #[test]
    fn test_simple_extrusion() {
        let reconstructor = Lod12Reconstructor::new();

        // Create a simple square footprint
        let exterior = LinearRing::from_vertices(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 0.0),
            Point3::new(0.0, 10.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
        ]);

        let footprint = Footprint::new("test", Polygon3D::new(exterior));

        let mesh = reconstructor
            .extrude_footprint(&footprint, 0.0, 10.0)
            .unwrap();

        // Should have 8 vertices (4 bottom + 4 top)
        assert_eq!(mesh.vertex_count(), 8);

        // Should have 6 faces (1 ground + 1 roof + 4 walls)
        assert_eq!(mesh.face_count(), 6);

        // Volume should be 10 * 10 * 10 = 1000
        let volume = mesh.compute_volume();
        assert!((volume - 1000.0).abs() < 1.0);
    }
}

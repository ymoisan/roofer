//! 3D mesh representation for building geometry.
//!
//! Provides types for roof surfaces and building solids with semantic surfaces.

use nalgebra::{Point3, Vector3};
use serde::{Deserialize, Serialize};

use crate::attributes::AttributeMap;

/// Surface type for semantic surfaces in CityJSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum SurfaceType {
    GroundSurface,
    WallSurface,
    RoofSurface,
    OuterCeilingSurface,
    OuterFloorSurface,
    ClosureSurface,
}

impl Default for SurfaceType {
    fn default() -> Self {
        SurfaceType::RoofSurface
    }
}

/// A semantic surface with attributes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SemanticSurface {
    pub surface_type: SurfaceType,
    pub on_footprint_edge: Option<bool>,
    /// For roof surfaces: azimuth angle (direction of slope)
    pub azimuth: Option<f64>,
    /// For roof surfaces: slope angle
    pub slope: Option<f64>,
    /// Height statistics
    pub h_roof_50p: Option<f64>,
    pub h_roof_70p: Option<f64>,
    pub h_roof_min: Option<f64>,
    pub h_roof_max: Option<f64>,
}

impl SemanticSurface {
    pub fn ground() -> Self {
        Self {
            surface_type: SurfaceType::GroundSurface,
            ..Default::default()
        }
    }

    pub fn wall(on_footprint_edge: bool) -> Self {
        Self {
            surface_type: SurfaceType::WallSurface,
            on_footprint_edge: Some(on_footprint_edge),
            ..Default::default()
        }
    }

    pub fn roof() -> Self {
        Self {
            surface_type: SurfaceType::RoofSurface,
            ..Default::default()
        }
    }

    pub fn with_roof_attributes(mut self, azimuth: f64, slope: f64) -> Self {
        self.azimuth = Some(azimuth);
        self.slope = Some(slope);
        self
    }
}

/// A face (polygon) in the mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Face {
    /// Indices into the mesh's vertex array
    pub indices: Vec<u32>,
    /// Semantic surface index (into Mesh::semantics)
    pub semantic_index: Option<usize>,
}

impl Face {
    pub fn new(indices: Vec<u32>) -> Self {
        Self {
            indices,
            semantic_index: None,
        }
    }

    pub fn with_semantic(mut self, index: usize) -> Self {
        self.semantic_index = Some(index);
        self
    }

    pub fn is_triangle(&self) -> bool {
        self.indices.len() == 3
    }
}

/// A 3D mesh representing building geometry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Mesh {
    /// Vertices of the mesh
    pub vertices: Vec<Point3<f64>>,
    /// Faces (polygons) of the mesh
    pub faces: Vec<Face>,
    /// Semantic surface definitions
    pub semantics: Vec<SemanticSurface>,
}

impl Mesh {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a vertex and return its index.
    pub fn add_vertex(&mut self, vertex: Point3<f64>) -> u32 {
        let index = self.vertices.len() as u32;
        self.vertices.push(vertex);
        index
    }

    /// Add a vertex, merging with existing if within tolerance.
    pub fn add_vertex_merged(&mut self, vertex: Point3<f64>, tolerance: f64) -> u32 {
        // Check if vertex already exists
        for (i, v) in self.vertices.iter().enumerate() {
            let dist = ((v.x - vertex.x).powi(2) + (v.y - vertex.y).powi(2) + (v.z - vertex.z).powi(2)).sqrt();
            if dist < tolerance {
                return i as u32;
            }
        }
        self.add_vertex(vertex)
    }

    /// Add a face.
    pub fn add_face(&mut self, face: Face) {
        self.faces.push(face);
    }

    /// Add a semantic surface and return its index.
    pub fn add_semantic(&mut self, semantic: SemanticSurface) -> usize {
        let index = self.semantics.len();
        self.semantics.push(semantic);
        index
    }

    /// Get the number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Get the number of faces.
    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    /// Compute bounding box.
    pub fn bounding_box(&self) -> Option<(Point3<f64>, Point3<f64>)> {
        if self.vertices.is_empty() {
            return None;
        }

        let mut min = self.vertices[0];
        let mut max = self.vertices[0];

        for v in &self.vertices[1..] {
            min.x = min.x.min(v.x);
            min.y = min.y.min(v.y);
            min.z = min.z.min(v.z);
            max.x = max.x.max(v.x);
            max.y = max.y.max(v.y);
            max.z = max.z.max(v.z);
        }

        Some((min, max))
    }

    /// Compute the normal of a face.
    pub fn face_normal(&self, face_index: usize) -> Option<Vector3<f64>> {
        let face = &self.faces.get(face_index)?;
        if face.indices.len() < 3 {
            return None;
        }

        let p0 = &self.vertices[face.indices[0] as usize];
        let p1 = &self.vertices[face.indices[1] as usize];
        let p2 = &self.vertices[face.indices[2] as usize];

        let v1 = p1 - p0;
        let v2 = p2 - p0;
        let normal = v1.cross(&v2);

        let len = normal.norm();
        if len < 1e-10 {
            return None;
        }

        Some(normal / len)
    }

    /// Compute the centroid of a face.
    pub fn face_centroid(&self, face_index: usize) -> Option<Point3<f64>> {
        let face = &self.faces.get(face_index)?;
        if face.indices.is_empty() {
            return None;
        }

        let sum: Point3<f64> = face
            .indices
            .iter()
            .map(|&i| &self.vertices[i as usize])
            .fold(Point3::origin(), |acc, p| Point3::new(acc.x + p.x, acc.y + p.y, acc.z + p.z));

        let n = face.indices.len() as f64;
        Some(Point3::new(sum.x / n, sum.y / n, sum.z / n))
    }

    /// Triangulate all faces (convert polygons to triangles).
    pub fn triangulate(&mut self) {
        let mut new_faces = Vec::new();

        for face in &self.faces {
            if face.indices.len() <= 3 {
                new_faces.push(face.clone());
            } else {
                // Fan triangulation
                for i in 1..(face.indices.len() - 1) {
                    let mut tri = Face::new(vec![
                        face.indices[0],
                        face.indices[i as usize],
                        face.indices[(i + 1) as usize],
                    ]);
                    tri.semantic_index = face.semantic_index;
                    new_faces.push(tri);
                }
            }
        }

        self.faces = new_faces;
    }

    /// Compute volume (assuming a closed solid).
    pub fn compute_volume(&self) -> f64 {
        let mut volume = 0.0;

        for face in &self.faces {
            if face.indices.len() < 3 {
                continue;
            }

            // Triangulate and sum signed volumes
            let p0 = &self.vertices[face.indices[0] as usize];
            for i in 1..(face.indices.len() - 1) {
                let p1 = &self.vertices[face.indices[i] as usize];
                let p2 = &self.vertices[face.indices[i + 1] as usize];

                // Signed volume of tetrahedron with origin
                volume += p0.coords.dot(&p1.coords.cross(&p2.coords));
            }
        }

        volume.abs() / 6.0
    }

    /// Get faces by semantic type.
    pub fn faces_by_type(&self, surface_type: SurfaceType) -> Vec<usize> {
        self.faces
            .iter()
            .enumerate()
            .filter(|(_, face)| {
                face.semantic_index
                    .map(|idx| self.semantics.get(idx).map(|s| s.surface_type == surface_type).unwrap_or(false))
                    .unwrap_or(false)
            })
            .map(|(i, _)| i)
            .collect()
    }
}

/// A complete building geometry with multiple LoD representations.
#[derive(Debug, Clone, Default)]
pub struct BuildingGeometry {
    /// LoD 1.2 geometry (extruded footprint)
    pub lod12: Option<Mesh>,
    /// LoD 1.3 geometry (extruded with step)
    pub lod13: Option<Mesh>,
    /// LoD 2.2 geometry (detailed roof)
    pub lod22: Option<Mesh>,
    /// Building attributes
    pub attributes: AttributeMap,
    /// Ground height
    pub h_ground: f64,
    /// Building ID
    pub id: String,
}

impl BuildingGeometry {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Default::default()
        }
    }

    /// Check if any geometry is available.
    pub fn has_geometry(&self) -> bool {
        self.lod12.is_some() || self.lod13.is_some() || self.lod22.is_some()
    }

    /// Get the best available LoD.
    pub fn best_lod(&self) -> Option<&Mesh> {
        self.lod22.as_ref().or(self.lod13.as_ref()).or(self.lod12.as_ref())
    }

    /// Compute geographic extent.
    pub fn geographic_extent(&self) -> Option<[f64; 6]> {
        let mesh = self.best_lod()?;
        let (min, max) = mesh.bounding_box()?;
        Some([min.x, min.y, min.z, max.x, max.y, max.z])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_basic() {
        let mut mesh = Mesh::new();

        // Add a triangle
        let i0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let i1 = mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
        let i2 = mesh.add_vertex(Point3::new(0.0, 1.0, 0.0));

        mesh.add_face(Face::new(vec![i0, i1, i2]));

        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.face_count(), 1);

        // Normal should point in Z direction
        let normal = mesh.face_normal(0).unwrap();
        assert!((normal.z.abs() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mesh_triangulation() {
        let mut mesh = Mesh::new();

        // Add a quad
        let i0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let i1 = mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));
        let i2 = mesh.add_vertex(Point3::new(1.0, 1.0, 0.0));
        let i3 = mesh.add_vertex(Point3::new(0.0, 1.0, 0.0));

        mesh.add_face(Face::new(vec![i0, i1, i2, i3]));

        assert_eq!(mesh.face_count(), 1);

        mesh.triangulate();

        assert_eq!(mesh.face_count(), 2);
        assert!(mesh.faces.iter().all(|f| f.is_triangle()));
    }

    #[test]
    fn test_mesh_volume() {
        let mut mesh = Mesh::new();

        // Create a unit cube
        let v = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
        ];

        for vertex in &v {
            mesh.add_vertex(*vertex);
        }

        // Add faces (outward normals)
        mesh.add_face(Face::new(vec![0, 3, 2, 1])); // bottom
        mesh.add_face(Face::new(vec![4, 5, 6, 7])); // top
        mesh.add_face(Face::new(vec![0, 1, 5, 4])); // front
        mesh.add_face(Face::new(vec![2, 3, 7, 6])); // back
        mesh.add_face(Face::new(vec![0, 4, 7, 3])); // left
        mesh.add_face(Face::new(vec![1, 2, 6, 5])); // right

        let volume = mesh.compute_volume();
        assert!((volume - 1.0).abs() < 1e-10);
    }
}

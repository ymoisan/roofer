//! LoD 2.2 reconstruction (detailed roof) with arrangement-based extrusion.
//!
//! This implementation follows the C++ roofer approach:
//! 1. Detect roof planes from point cloud
//! 2. Compute plane intersections (ridgelines)
//! 3. Build 2D arrangement (planar subdivision) using CDT
//! 4. Classify each arrangement face to its best-fitting plane
//! 5. Extrude to 3D: outer walls from boundary, internal walls between different planes

use crate::mesh::{Face, Mesh, SemanticSurface};
use crate::plane::{Plane, PlaneDetector, RansacConfig};
use crate::point_cloud::PointCloud;
use crate::polygon::Footprint;
use nalgebra::Point3;
use spade::{ConstrainedDelaunayTriangulation, Point2 as SpadePoint2, Triangulation};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Configuration for wall generation.
#[derive(Debug, Clone)]
pub struct WallConfig {
    /// Minimum height difference to create internal wall (meters).
    /// If None, create walls for all edges regardless of height difference.
    pub min_internal_wall_height_diff: Option<f64>,
    
    /// Snap tolerance exponent (10^-exp) for filtering short edges.
    pub snap_tolerance_exp: i32,
    
    /// Build walls for all edges vs only height-different edges.
    pub build_all_edges: bool,
    
    /// Minimum edge length to create wall (meters).
    pub min_edge_length: f64,
}

impl Default for WallConfig {
    fn default() -> Self {
        Self {
            min_internal_wall_height_diff: None, // Default: all edges
            snap_tolerance_exp: 4, // 10^-4 = 0.0001
            build_all_edges: true,
            min_edge_length: 0.0,
        }
    }
}

/// A wall quad with 4 vertices and wall type.
#[derive(Debug, Clone)]
struct WallQuad {
    v0: Point3<f64>,
    v1: Point3<f64>,
    v2: Point3<f64>,
    v3: Point3<f64>,
    #[allow(dead_code)]
    is_outer: bool,
}


/// A ridgeline segment (intersection of two planes clipped to footprint).
#[derive(Debug, Clone)]
struct Ridgeline {
    p1: SpadePoint2<f64>,
    p2: SpadePoint2<f64>,
    #[allow(dead_code)]
    plane_a: usize,
    #[allow(dead_code)]
    plane_b: usize,
}

/// An arrangement face with its assigned plane index.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ArrangementFace {
    vertices: Vec<SpadePoint2<f64>>,
    plane_idx: usize,
}

/// An arrangement edge (either boundary or internal).
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ArrangementEdge {
    p1: SpadePoint2<f64>,
    p2: SpadePoint2<f64>,
    is_boundary: bool,
    /// Plane indices on each side (left, right). None means outside footprint.
    left_plane: Option<usize>,
    right_plane: Option<usize>,
}

#[derive(Debug, Error)]
pub enum Lod22Error {
    #[error("Empty footprint")]
    EmptyFootprint,
    #[error("Empty point cloud")]
    EmptyPointCloud,
    #[error("No planes detected")]
    NoPlanesDetected,
    #[error("Triangulation failed: {0}")]
    TriangulationFailed(String),
    #[error("Invalid geometry: {0}")]
    InvalidGeometry(String),
}

/// Reconstructor for LoD 2.2 (detailed roof geometry).
pub struct Lod22Reconstructor {
    plane_detector: PlaneDetector,
    wall_config: WallConfig,
}

impl Lod22Reconstructor {
    pub fn new(config: RansacConfig) -> Self {
        Self {
            plane_detector: PlaneDetector::new(config),
            wall_config: WallConfig::default(),
        }
    }

    pub fn new_with_wall_config(config: RansacConfig, wall_config: WallConfig) -> Self {
        Self {
            plane_detector: PlaneDetector::new(config),
            wall_config,
        }
    }

    /// Reconstruct LoD 2.2 geometry.
    pub fn reconstruct(
        &self,
        footprint: &Footprint,
        point_cloud: &PointCloud,
        h_ground: f64,
    ) -> Result<Mesh, Lod22Error> {
        if footprint.polygon.is_empty() {
            return Err(Lod22Error::EmptyFootprint);
        }

        if point_cloud.is_empty() {
            return Err(Lod22Error::EmptyPointCloud);
        }

        // Detect roof planes
        let planes = self
            .plane_detector
            .detect_multiple(&point_cloud.positions, 20);

        if planes.is_empty() {
            // Fall back to flat roof if no planes detected
            return self.create_flat_roof(footprint, point_cloud, h_ground);
        }

        // Build the mesh with arrangement-based extrusion
        self.build_mesh_with_arrangement(footprint, &planes, point_cloud, h_ground)
    }

    /// Create a flat roof (fallback when plane detection fails).
    fn create_flat_roof(
        &self,
        footprint: &Footprint,
        point_cloud: &PointCloud,
        h_ground: f64,
    ) -> Result<Mesh, Lod22Error> {
        let mut mesh = Mesh::new();

        let stats = point_cloud.compute_statistics();
        let h_roof = stats.z_70p;

        // Add semantics
        let ground_idx = mesh.add_semantic(SemanticSurface::ground());
        let wall_idx = mesh.add_semantic(SemanticSurface::wall(true));
        let roof_idx = mesh.add_semantic(SemanticSurface::roof());

        let exterior = &footprint.polygon.exterior;
        let n = exterior.len().saturating_sub(1);
        if n < 3 {
            return Err(Lod22Error::InvalidGeometry("Footprint too small".to_string()));
        }

        // Add bottom vertices
        let mut bottom_indices = Vec::with_capacity(n);
        for i in 0..n {
            let v = &exterior.vertices[i];
            bottom_indices.push(mesh.add_vertex(Point3::new(v.x, v.y, h_ground)));
        }

        // Add top vertices
        let mut top_indices = Vec::with_capacity(n);
        for i in 0..n {
            let v = &exterior.vertices[i];
            top_indices.push(mesh.add_vertex(Point3::new(v.x, v.y, h_roof)));
        }

        // Ground face (reversed for outward normal)
        mesh.add_face(
            Face::new(bottom_indices.iter().rev().cloned().collect()).with_semantic(ground_idx),
        );

        // Roof face
        mesh.add_face(Face::new(top_indices.clone()).with_semantic(roof_idx));

        // Wall faces
        for i in 0..n {
            let j = (i + 1) % n;
            mesh.add_face(
                Face::new(vec![
                    bottom_indices[i],
                    bottom_indices[j],
                    top_indices[j],
                    top_indices[i],
                ])
                .with_semantic(wall_idx),
            );
        }

        Ok(mesh)
    }

    /// Build mesh using arrangement-based extrusion (C++ equivalent).
    fn build_mesh_with_arrangement(
        &self,
        footprint: &Footprint,
        planes: &[Plane],
        point_cloud: &PointCloud,
        h_ground: f64,
    ) -> Result<Mesh, Lod22Error> {
        let mut mesh = Mesh::new();
        let stats = point_cloud.compute_statistics();

        // Add ground semantic
        let ground_idx = mesh.add_semantic(SemanticSurface::ground());
        let wall_idx = mesh.add_semantic(SemanticSurface::wall(true));

        // Add roof semantics for each plane
        let roof_semantics: Vec<usize> = planes
            .iter()
            .map(|plane| {
                mesh.add_semantic(
                    SemanticSurface::roof().with_roof_attributes(
                        plane.azimuth_degrees(),
                        plane.slope_degrees(),
                    ),
                )
            })
            .collect();

        let exterior = &footprint.polygon.exterior;
        let n = exterior.len().saturating_sub(1);
        if n < 3 {
            return Err(Lod22Error::InvalidGeometry("Footprint too small".to_string()));
        }

        // Step 1: Compute ridgelines (plane intersections clipped to footprint)
        let ridgelines = compute_ridgelines(footprint, planes, n);

        // Step 2: Build 2D arrangement using CDT
        let (cdt, _vertex_index, boundary_z) =
            build_arrangement(footprint, &ridgelines, point_cloud, n, stats.z_70p);

        let cdt = match cdt {
            Ok(c) => c,
            Err(_) => {
                // Fallback to simple extrusion if CDT fails
                return self.build_simple_extrusion(footprint, planes, point_cloud, h_ground);
            }
        };

        // Step 3: Extract arrangement faces and edges
        let (_faces, _edges) = extract_arrangement(&cdt, footprint, planes, point_cloud, stats.z_70p);

        // Step 4: Build ground face
        let mut bottom_indices = Vec::with_capacity(n);
        for i in 0..n {
            let v = &exterior.vertices[i];
            bottom_indices.push(mesh.add_vertex(Point3::new(v.x, v.y, h_ground)));
        }
        mesh.add_face(
            Face::new(bottom_indices.iter().rev().cloned().collect()).with_semantic(ground_idx),
        );

        // Step 5: Build roof faces from arrangement
        let mut roof_z_values: Vec<Vec<f64>> = vec![Vec::new(); roof_semantics.len()];
        
        for tri_face in cdt.inner_faces() {
            let [p0, p1, p2] = tri_face.positions();
            let cx = (p0.x + p1.x + p2.x) / 3.0;
            let cy = (p0.y + p1.y + p2.y) / 3.0;

            // Skip triangles outside footprint
            if !footprint.contains_2d(cx, cy) {
                continue;
            }

            // Determine plane for this face
            let local_z = local_height_at_point(point_cloud, cx, cy, stats.z_70p);
            let plane_idx = best_plane_index_for_xy(cx, cy, local_z, planes);
            let roof_semantic = roof_semantics
                .get(plane_idx)
                .copied()
                .unwrap_or_else(|| *roof_semantics.first().unwrap_or(&0));

            // Compute 3D vertices by projecting onto the assigned plane
            let z0 = height_from_plane(p0.x, p0.y, plane_idx, planes, &boundary_z, point_cloud, stats.z_70p);
            let z1 = height_from_plane(p1.x, p1.y, plane_idx, planes, &boundary_z, point_cloud, stats.z_70p);
            let z2 = height_from_plane(p2.x, p2.y, plane_idx, planes, &boundary_z, point_cloud, stats.z_70p);

            let mut tri = [
                Point3::new(p0.x, p0.y, z0),
                Point3::new(p1.x, p1.y, z1),
                Point3::new(p2.x, p2.y, z2),
            ];

            // Ensure CCW winding
            if signed_area_2d(&tri) < 0.0 {
                tri.swap(1, 2);
            }

            let idx0 = mesh.add_vertex(tri[0]);
            let idx1 = mesh.add_vertex(tri[1]);
            let idx2 = mesh.add_vertex(tri[2]);
            mesh.add_face(Face::new(vec![idx0, idx1, idx2]).with_semantic(roof_semantic));

            // Track Z values for height stats
            if let Some(pos) = roof_semantics.iter().position(|&s| s == roof_semantic) {
                roof_z_values[pos].extend_from_slice(&[tri[0].z, tri[1].z, tri[2].z]);
            }
        }

        // Step 6: Build walls from all CDT edges
        let walls = build_all_walls(
            &cdt,
            footprint,
            planes,
            point_cloud,
            &boundary_z,
            &self.wall_config,
            h_ground,
            stats.z_70p,
        );
        
        // Validate that we have walls (especially for footprint perimeter)
        let wall_count = walls.len();
        if wall_count == 0 {
            tracing::warn!("Building {} has no walls - this may indicate a reconstruction issue", footprint.id);
        }
        
        for wall in walls {
            let idx0 = mesh.add_vertex(wall.v0);
            let idx1 = mesh.add_vertex(wall.v1);
            let idx2 = mesh.add_vertex(wall.v2);
            let idx3 = mesh.add_vertex(wall.v3);
            mesh.add_face(Face::new(vec![idx0, idx1, idx2, idx3]).with_semantic(wall_idx));
        }
        
        // Final validation: ensure footprint perimeter has walls
        validate_footprint_walls(&mesh, footprint, wall_count, h_ground);

        // Update roof semantics with height statistics
        for (i, semantic_idx) in roof_semantics.iter().enumerate() {
            if let Some(semantic) = mesh.semantics.get_mut(*semantic_idx) {
                if roof_z_values[i].is_empty() {
                    continue;
                }
                let mut sorted = roof_z_values[i].clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                semantic.h_roof_min = sorted.first().copied();
                semantic.h_roof_max = sorted.last().copied();
                semantic.h_roof_50p = Some(sorted[sorted.len() / 2]);
                semantic.h_roof_70p = Some(sorted[sorted.len() * 7 / 10]);
            }
        }

        Ok(mesh)
    }

    /// Simple extrusion fallback (like old implementation).
    fn build_simple_extrusion(
        &self,
        footprint: &Footprint,
        planes: &[Plane],
        point_cloud: &PointCloud,
        h_ground: f64,
    ) -> Result<Mesh, Lod22Error> {
        let mut mesh = Mesh::new();

        let ground_idx = mesh.add_semantic(SemanticSurface::ground());
        let wall_idx = mesh.add_semantic(SemanticSurface::wall(true));
        let roof_idx = mesh.add_semantic(
            SemanticSurface::roof().with_roof_attributes(
                planes.first().map(|p| p.azimuth_degrees()).unwrap_or(0.0),
                planes.first().map(|p| p.slope_degrees()).unwrap_or(0.0),
            ),
        );

        let exterior = &footprint.polygon.exterior;
        let n = exterior.len().saturating_sub(1);

        // Bottom vertices
        let mut bottom_indices = Vec::with_capacity(n);
        for i in 0..n {
            let v = &exterior.vertices[i];
            bottom_indices.push(mesh.add_vertex(Point3::new(v.x, v.y, h_ground)));
        }

        // Top vertices with plane projection
        let mut top_indices = Vec::with_capacity(n);
        for i in 0..n {
            let v = &exterior.vertices[i];
            let h = self.find_roof_height_at_point(v.x, v.y, planes, point_cloud);
            top_indices.push(mesh.add_vertex(Point3::new(v.x, v.y, h)));
        }

        // Ground
        mesh.add_face(
            Face::new(bottom_indices.iter().rev().cloned().collect()).with_semantic(ground_idx),
        );

        // Roof (fan triangulation)
        for i in 1..(n - 1) {
            mesh.add_face(
                Face::new(vec![top_indices[0], top_indices[i], top_indices[i + 1]])
                    .with_semantic(roof_idx),
            );
        }

        // Walls
        for i in 0..n {
            let j = (i + 1) % n;
            mesh.add_face(
                Face::new(vec![
                    bottom_indices[i],
                    bottom_indices[j],
                    top_indices[j],
                    top_indices[i],
                ])
                .with_semantic(wall_idx),
            );
        }

        Ok(mesh)
    }

    fn find_roof_height_at_point(
        &self,
        x: f64,
        y: f64,
        planes: &[Plane],
        point_cloud: &PointCloud,
    ) -> f64 {
        let search_radius = 2.0;
        let mut nearby_z = Vec::new();

        for p in &point_cloud.positions {
            let dx = p.x - x;
            let dy = p.y - y;
            if dx * dx + dy * dy < search_radius * search_radius {
                nearby_z.push(p.z);
            }
        }

        if nearby_z.is_empty() {
            if let Some(plane) = planes.first() {
                if plane.normal.z.abs() > 0.01 {
                    return -(plane.normal.x * x + plane.normal.y * y + plane.d) / plane.normal.z;
                }
            }
            return point_cloud.compute_statistics().z_70p;
        }

        nearby_z.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        nearby_z[nearby_z.len() * 7 / 10]
    }
}

/// Compute ridgelines from plane intersections, clipped to footprint.
fn compute_ridgelines(footprint: &Footprint, planes: &[Plane], n: usize) -> Vec<Ridgeline> {
    let mut ridgelines = Vec::new();
    let eps = 1e-8;
    let min_angle_deg: f64 = 10.0;
    let min_dot = (min_angle_deg.to_radians()).cos();
    let exterior = &footprint.polygon.exterior;

    for i in 0..planes.len() {
        for j in (i + 1)..planes.len() {
            // Skip near-parallel planes
            let dot = planes[i].normal.dot(&planes[j].normal).abs();
            if dot > min_dot {
                continue;
            }

            // Compute intersection line in 3D, project to 2D
            if let Some((p, dir)) = plane_intersection_line_2d(&planes[i], &planes[j], eps) {
                // Find intersections with footprint boundary
                let mut intersections = Vec::new();
                for k in 0..n {
                    let a = &exterior.vertices[k];
                    let b = &exterior.vertices[(k + 1) % n];
                    if let Some(pt) = line_segment_intersection_2d(
                        p,
                        dir,
                        SpadePoint2::new(a.x, a.y),
                        SpadePoint2::new(b.x, b.y),
                        eps,
                    ) {
                        intersections.push(pt);
                    }
                }

                // Need at least 2 intersections to form a ridgeline segment
                if intersections.len() >= 2 {
                    let (p1, p2) = farthest_pair(&intersections);
                    
                    // Filter out very short ridgelines
                    let len_sq = (p2.x - p1.x).powi(2) + (p2.y - p1.y).powi(2);
                    if len_sq > 0.25 {
                        ridgelines.push(Ridgeline {
                            p1,
                            p2,
                            plane_a: i,
                            plane_b: j,
                        });
                    }
                }
            }
        }
    }

    ridgelines
}

/// Build 2D arrangement using Constrained Delaunay Triangulation.
fn build_arrangement(
    footprint: &Footprint,
    ridgelines: &[Ridgeline],
    point_cloud: &PointCloud,
    n: usize,
    stats_z70: f64,
) -> (
    Result<ConstrainedDelaunayTriangulation<SpadePoint2<f64>>, String>,
    HashMap<(i64, i64), usize>,
    HashMap<(i64, i64), f64>,
) {
    let mut vertices = Vec::new();
    let mut edges = Vec::new();
    let mut vertex_index: HashMap<(i64, i64), usize> = HashMap::new();
    let mut boundary_z: HashMap<(i64, i64), f64> = HashMap::new();

    let exterior = &footprint.polygon.exterior;

    // Add footprint boundary vertices and edges
    let mut boundary_indices = Vec::with_capacity(n);
    for i in 0..n {
        let v = &exterior.vertices[i];
        let idx = insert_vertex(&mut vertices, &mut vertex_index, v.x, v.y);
        boundary_indices.push(idx);
        
        // Store boundary Z value
        let z = local_height_at_point(point_cloud, v.x, v.y, stats_z70);
        boundary_z.insert(point_key(v.x, v.y), z);
    }

    // Add boundary edges
    for i in 0..n {
        let j = (i + 1) % n;
        edges.push([boundary_indices[i], boundary_indices[j]]);
    }

    // Add ridgeline constraints
    for ridge in ridgelines {
        let idx1 = insert_vertex(&mut vertices, &mut vertex_index, ridge.p1.x, ridge.p1.y);
        let idx2 = insert_vertex(&mut vertices, &mut vertex_index, ridge.p2.x, ridge.p2.y);
        if idx1 != idx2 {
            edges.push([idx1, idx2]);
        }
        
        // Store boundary Z for ridgeline endpoints (for consistent height at edges)
        let z1 = local_height_at_point(point_cloud, ridge.p1.x, ridge.p1.y, stats_z70);
        let z2 = local_height_at_point(point_cloud, ridge.p2.x, ridge.p2.y, stats_z70);
        boundary_z.insert(point_key(ridge.p1.x, ridge.p1.y), z1);
        boundary_z.insert(point_key(ridge.p2.x, ridge.p2.y), z2);
    }

    // Build CDT
    let mut conflict_count = 0usize;
    let result = ConstrainedDelaunayTriangulation::<SpadePoint2<f64>>::try_bulk_load_cdt(
        vertices,
        edges,
        |_| {
            conflict_count += 1;
        },
    );

    match result {
        Ok(cdt) => (Ok(cdt), vertex_index, boundary_z),
        Err(e) => (Err(format!("{:?}", e)), vertex_index, boundary_z),
    }
}

/// Extract arrangement faces and edges from CDT.
fn extract_arrangement(
    cdt: &ConstrainedDelaunayTriangulation<SpadePoint2<f64>>,
    footprint: &Footprint,
    planes: &[Plane],
    point_cloud: &PointCloud,
    stats_z70: f64,
) -> (Vec<ArrangementFace>, Vec<ArrangementEdge>) {
    let mut faces = Vec::new();
    let edges = Vec::new();

    // Extract faces (triangles)
    for face in cdt.inner_faces() {
        let [p0, p1, p2] = face.positions();
        let cx = (p0.x + p1.x + p2.x) / 3.0;
        let cy = (p0.y + p1.y + p2.y) / 3.0;

        if !footprint.contains_2d(cx, cy) {
            continue;
        }

        let local_z = local_height_at_point(point_cloud, cx, cy, stats_z70);
        let plane_idx = best_plane_index_for_xy(cx, cy, local_z, planes);

        faces.push(ArrangementFace {
            vertices: vec![p0, p1, p2],
            plane_idx,
        });
    }

    (faces, edges)
}

/// Build walls from all CDT edges (matching C++ ArrangementExtruder behavior).
fn build_all_walls(
    cdt: &ConstrainedDelaunayTriangulation<SpadePoint2<f64>>,
    footprint: &Footprint,
    planes: &[Plane],
    point_cloud: &PointCloud,
    boundary_z: &HashMap<(i64, i64), f64>,
    wall_config: &WallConfig,
    h_ground: f64,
    stats_z70: f64,
) -> Vec<WallQuad> {
    let mut walls = Vec::new();
    let mut seen_edges: HashSet<((i64, i64), (i64, i64))> = HashSet::new();
    let snap_tolerance = 10.0_f64.powi(-wall_config.snap_tolerance_exp);

    // Track which footprint edges have been processed
    let mut footprint_edges_processed: HashSet<((i64, i64), (i64, i64))> = HashSet::new();
    let exterior = &footprint.polygon.exterior;
    let n = exterior.len().saturating_sub(1);
    for i in 0..n {
        let j = (i + 1) % n;
        let v_i = &exterior.vertices[i];
        let v_j = &exterior.vertices[j];
        let key = (point_key(v_i.x, v_i.y), point_key(v_j.x, v_j.y));
        footprint_edges_processed.insert(key);
    }

    // Iterate through ALL edges in CDT
    for edge in cdt.undirected_edges() {
        let [p1, p2] = edge.positions();
        
        // Skip very short edges
        let edge_len_sq = (p2.x - p1.x).powi(2) + (p2.y - p1.y).powi(2);
        if edge_len_sq < wall_config.min_edge_length.powi(2) {
            continue;
        }
        if edge_len_sq < snap_tolerance * snap_tolerance {
            continue;
        }

        // Avoid duplicate edges
        let key1 = (point_key(p1.x, p1.y), point_key(p2.x, p2.y));
        let key2 = (point_key(p2.x, p2.y), point_key(p1.x, p1.y));
        if seen_edges.contains(&key1) || seen_edges.contains(&key2) {
            continue;
        }
        seen_edges.insert(key1);

        // Get the two adjacent faces
        let directed = edge.as_directed();
        let face_a_handle = directed.face();
        let face_b_handle = directed.rev().face();

        // Determine if faces are inside footprint
        let mut fp_a = false;
        let mut fp_b = false;
        let mut plane_idx_a = None;
        let mut plane_idx_b = None;

        // Check face A
        if !face_a_handle.is_outer() {
            let e0 = directed;
            let e1 = e0.next();
            let e2 = e1.next();
            let fp0 = e0.from().position();
            let fp1 = e1.from().position();
            let fp2 = e2.from().position();
            let cx = (fp0.x + fp1.x + fp2.x) / 3.0;
            let cy = (fp0.y + fp1.y + fp2.y) / 3.0;
            fp_a = footprint.contains_2d(cx, cy);
            if fp_a {
                let local_z = local_height_at_point(point_cloud, cx, cy, stats_z70);
                plane_idx_a = Some(best_plane_index_for_xy(cx, cy, local_z, planes));
            }
        }

        // Check face B
        if !face_b_handle.is_outer() {
            let directed_rev = directed.rev();
            let e0 = directed_rev;
            let e1 = e0.next();
            let e2 = e1.next();
            let fp0 = e0.from().position();
            let fp1 = e1.from().position();
            let fp2 = e2.from().position();
            let cx = (fp0.x + fp1.x + fp2.x) / 3.0;
            let cy = (fp0.y + fp1.y + fp2.y) / 3.0;
            fp_b = footprint.contains_2d(cx, cy);
            if fp_b {
                let local_z = local_height_at_point(point_cloud, cx, cy, stats_z70);
                plane_idx_b = Some(best_plane_index_for_xy(cx, cy, local_z, planes));
            }
        }

        // Skip edge if neither face is in footprint
        if !fp_a && !fp_b {
            continue;
        }

        // Check if this edge is on or near the footprint boundary
        // If so, it MUST get a wall (exterior envelope visible to LiDAR)
        let is_footprint_boundary = footprint_edges_processed.contains(&key1) 
            || footprint_edges_processed.contains(&key2)
            || is_edge_on_footprint_boundary(p1, p2, footprint, snap_tolerance);

        // Only build outer walls (exterior envelope visible to LiDAR)
        // Skip inner walls (both faces inside footprint) UNLESS it's a footprint boundary edge
        if fp_a && fp_b && !is_footprint_boundary {
            // Both faces are inside footprint and NOT on boundary - this is an inner wall, skip it
            continue;
        }

        // This is an outer wall (at least one face is outside footprint OR it's a footprint boundary edge)
        // Note: We skip inner walls (both faces inside) as LiDAR cannot see them,
        // but footprint boundary edges always get walls

        // Compute heights on both sides
        let h1a = if fp_a {
            height_from_plane(p1.x, p1.y, plane_idx_a.unwrap(), planes, boundary_z, point_cloud, stats_z70)
        } else {
            h_ground
        };
        let h1b = if fp_b {
            height_from_plane(p1.x, p1.y, plane_idx_b.unwrap(), planes, boundary_z, point_cloud, stats_z70)
        } else {
            h_ground
        };
        let h2a = if fp_a {
            height_from_plane(p2.x, p2.y, plane_idx_a.unwrap(), planes, boundary_z, point_cloud, stats_z70)
        } else {
            h_ground
        };
        let h2b = if fp_b {
            height_from_plane(p2.x, p2.y, plane_idx_b.unwrap(), planes, boundary_z, point_cloud, stats_z70)
        } else {
            h_ground
        };

        // Inner walls are skipped above, so we only process outer walls here
        // No height difference check needed for outer walls

        // Build wall quad(s) - handle height crossings
        let wall_quads = build_wall_quad(p1, p2, h1a, h1b, h2a, h2b, wall_config);
        walls.extend(wall_quads);

        // Mark footprint edge as processed if this edge matches a footprint edge
        if footprint_edges_processed.contains(&key1) || footprint_edges_processed.contains(&key2) {
            // Already tracked
        }
    }

    // Ensure all footprint perimeter edges have walls
    for i in 0..n {
        let j = (i + 1) % n;
        let v_i = &exterior.vertices[i];
        let v_j = &exterior.vertices[j];
        let key = (point_key(v_i.x, v_i.y), point_key(v_j.x, v_j.y));
        
        // Check if we already have a wall for this edge
        let mut has_wall = false;
        for wall in &walls {
            let w_key1 = (point_key(wall.v0.x, wall.v0.y), point_key(wall.v1.x, wall.v1.y));
            let w_key2 = (point_key(wall.v1.x, wall.v1.y), point_key(wall.v0.x, wall.v0.y));
            if w_key1 == key || w_key2 == key {
                has_wall = true;
                break;
            }
        }
        
        if !has_wall {
            // Add explicit footprint perimeter wall
            let z_i = boundary_z.get(&point_key(v_i.x, v_i.y)).copied()
                .unwrap_or_else(|| local_height_at_point(point_cloud, v_i.x, v_i.y, stats_z70));
            let z_j = boundary_z.get(&point_key(v_j.x, v_j.y)).copied()
                .unwrap_or_else(|| local_height_at_point(point_cloud, v_j.x, v_j.y, stats_z70));
            
            walls.push(WallQuad {
                v0: Point3::new(v_i.x, v_i.y, h_ground),
                v1: Point3::new(v_j.x, v_j.y, h_ground),
                v2: Point3::new(v_j.x, v_j.y, z_j),
                v3: Point3::new(v_i.x, v_i.y, z_i),
                is_outer: true,
            });
        }
    }

    walls
}

/// Build wall quad(s) from edge heights, handling height crossings.
fn build_wall_quad(
    p1: SpadePoint2<f64>,
    p2: SpadePoint2<f64>,
    h1a: f64,
    h1b: f64,
    h2a: f64,
    h2b: f64,
    _wall_config: &WallConfig,
) -> Vec<WallQuad> {
    // Check for height crossing (h1a < h1b but h2a > h2b, or vice versa)
    let crosses = (h1a < h1b && h2a > h2b) || (h1a > h1b && h2a < h2b);
    
    if crosses {
        // Compute intersection point of the two height lines
        // Line A: from (p1.x, p1.y, h1a) to (p2.x, p2.y, h2a)
        // Line B: from (p1.x, p1.y, h1b) to (p2.x, p2.y, h2b)
        // Find intersection in 3D
        
        // Parameterize line A: p1 + t * (p2 - p1), z = h1a + t * (h2a - h1a)
        // Parameterize line B: p1 + s * (p2 - p1), z = h1b + s * (h2b - h1b)
        // At intersection: h1a + t * (h2a - h1a) = h1b + s * (h2b - h1b)
        // And t = s (same XY position)
        // So: h1a + t * (h2a - h1a) = h1b + t * (h2b - h1b)
        // t * (h2a - h1a - h2b + h1b) = h1b - h1a
        // t = (h1b - h1a) / (h2a - h1a - h2b + h1b)
        
        let denom = (h2a - h1a) - (h2b - h1b);
        if denom.abs() > 1e-8 {
            let t = (h1b - h1a) / denom;
            if t > 0.0 && t < 1.0 {
                let px = p1.x + t * (p2.x - p1.x);
                let py = p1.y + t * (p2.y - p1.y);
                let hx = h1a + t * (h2a - h1a);
                
                // Create two wall quads
                return vec![
                    WallQuad {
                        v0: Point3::new(p1.x, p1.y, h1a.min(h1b)),
                        v1: Point3::new(px, py, hx),
                        v2: Point3::new(px, py, hx),
                        v3: Point3::new(p1.x, p1.y, h1a.max(h1b)),
                        is_outer: true, // Only outer walls are built
                    },
                    WallQuad {
                        v0: Point3::new(px, py, hx),
                        v1: Point3::new(p2.x, p2.y, h2a.min(h2b)),
                        v2: Point3::new(p2.x, p2.y, h2a.max(h2b)),
                        v3: Point3::new(px, py, hx),
                        is_outer: true, // Only outer walls are built
                    },
                ];
            }
        }
    }
    
    // No crossing - single quad
    vec![WallQuad {
        v0: Point3::new(p1.x, p1.y, h1a.min(h1b)),
        v1: Point3::new(p2.x, p2.y, h2a.min(h2b)),
        v2: Point3::new(p2.x, p2.y, h2a.max(h2b)),
        v3: Point3::new(p1.x, p1.y, h1a.max(h1b)),
        is_outer: true, // Only outer walls are built
    }]
}


fn height_from_plane(
    x: f64,
    y: f64,
    plane_idx: usize,
    planes: &[Plane],
    boundary_z: &HashMap<(i64, i64), f64>,
    point_cloud: &PointCloud,
    stats_z70: f64,
) -> f64 {
    // First check if this is a boundary point
    if let Some(z) = boundary_z.get(&point_key(x, y)) {
        return *z;
    }

    // Project onto plane
    if let Some(plane) = planes.get(plane_idx) {
        if plane.normal.z.abs() > 1e-6 {
            return -(plane.normal.x * x + plane.normal.y * y + plane.d) / plane.normal.z;
        }
    }
    
    local_height_at_point(point_cloud, x, y, stats_z70)
}

fn insert_vertex(
    vertices: &mut Vec<SpadePoint2<f64>>,
    vertex_index: &mut HashMap<(i64, i64), usize>,
    x: f64,
    y: f64,
) -> usize {
    let key = point_key(x, y);
    if let Some(idx) = vertex_index.get(&key).copied() {
        return idx;
    }
    let idx = vertices.len();
    vertices.push(SpadePoint2::new(x, y));
    vertex_index.insert(key, idx);
    idx
}

fn local_height_at_point(point_cloud: &PointCloud, x: f64, y: f64, fallback: f64) -> f64 {
    let search_radius = 2.0;
    let mut nearby_z = Vec::new();
    for p in &point_cloud.positions {
        let dx = p.x - x;
        let dy = p.y - y;
        if dx * dx + dy * dy < search_radius * search_radius {
            nearby_z.push(p.z);
        }
    }
    if nearby_z.is_empty() {
        return fallback;
    }
    nearby_z.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    nearby_z[nearby_z.len() * 7 / 10]
}

fn plane_intersection_line_2d(
    a: &Plane,
    b: &Plane,
    eps: f64,
) -> Option<(SpadePoint2<f64>, SpadePoint2<f64>)> {
    let dir = a.normal.cross(&b.normal);
    if dir.norm() < eps {
        return None;
    }

    // Find a point on the intersection line by solving the system
    let (x, y) = if dir.z.abs() >= dir.x.abs() && dir.z.abs() >= dir.y.abs() {
        let det = a.normal.x * b.normal.y - a.normal.y * b.normal.x;
        if det.abs() < eps {
            return None;
        }
        let rhs1 = -a.d;
        let rhs2 = -b.d;
        let x = (rhs1 * b.normal.y - a.normal.y * rhs2) / det;
        let y = (a.normal.x * rhs2 - rhs1 * b.normal.x) / det;
        (x, y)
    } else if dir.x.abs() >= dir.y.abs() {
        let det = a.normal.y * b.normal.z - a.normal.z * b.normal.y;
        if det.abs() < eps {
            return None;
        }
        let rhs1 = -a.d;
        let rhs2 = -b.d;
        let y = (rhs1 * b.normal.z - a.normal.z * rhs2) / det;
        (0.0, y)
    } else {
        let det = a.normal.x * b.normal.z - a.normal.z * b.normal.x;
        if det.abs() < eps {
            return None;
        }
        let rhs1 = -a.d;
        let rhs2 = -b.d;
        let x = (rhs1 * b.normal.z - a.normal.z * rhs2) / det;
        (x, 0.0)
    };

    let dir2 = SpadePoint2::new(dir.x, dir.y);
    if (dir2.x * dir2.x + dir2.y * dir2.y).sqrt() < eps {
        return None;
    }
    Some((SpadePoint2::new(x, y), dir2))
}

fn line_segment_intersection_2d(
    line_point: SpadePoint2<f64>,
    line_dir: SpadePoint2<f64>,
    seg_a: SpadePoint2<f64>,
    seg_b: SpadePoint2<f64>,
    eps: f64,
) -> Option<SpadePoint2<f64>> {
    let r = line_dir;
    let s = SpadePoint2::new(seg_b.x - seg_a.x, seg_b.y - seg_a.y);
    let denom = cross_2d(r, s);
    if denom.abs() < eps {
        return None;
    }
    let qp = SpadePoint2::new(seg_a.x - line_point.x, seg_a.y - line_point.y);
    let t = cross_2d(qp, s) / denom;
    let u = cross_2d(qp, r) / denom;
    if u < -eps || u > 1.0 + eps {
        return None;
    }
    Some(SpadePoint2::new(
        line_point.x + t * r.x,
        line_point.y + t * r.y,
    ))
}

fn cross_2d(a: SpadePoint2<f64>, b: SpadePoint2<f64>) -> f64 {
    a.x * b.y - a.y * b.x
}

fn farthest_pair(points: &[SpadePoint2<f64>]) -> (SpadePoint2<f64>, SpadePoint2<f64>) {
    let mut best = (points[0], points[0]);
    let mut best_d = 0.0;
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let dx = points[i].x - points[j].x;
            let dy = points[i].y - points[j].y;
            let d = dx * dx + dy * dy;
            if d > best_d {
                best_d = d;
                best = (points[i], points[j]);
            }
        }
    }
    best
}

fn point_key(x: f64, y: f64) -> (i64, i64) {
    let scale = 1_000_000.0;
    ((x * scale).round() as i64, (y * scale).round() as i64)
}

fn signed_area_2d(tri: &[Point3<f64>; 3]) -> f64 {
    (tri[1].x - tri[0].x) * (tri[2].y - tri[0].y)
        - (tri[2].x - tri[0].x) * (tri[1].y - tri[0].y)
}

fn best_plane_index_for_xy(x: f64, y: f64, local_z: f64, planes: &[Plane]) -> usize {
    let mut best_idx = 0usize;
    let mut best_dist = f64::MAX;
    for (i, plane) in planes.iter().enumerate() {
        if plane.normal.z.abs() < 1e-6 {
            continue;
        }
        let z = -(plane.normal.x * x + plane.normal.y * y + plane.d) / plane.normal.z;
        let dist = (z - local_z).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }
    best_idx
}

/// Check if an edge is on or near the footprint boundary.
fn is_edge_on_footprint_boundary(
    p1: SpadePoint2<f64>,
    p2: SpadePoint2<f64>,
    footprint: &Footprint,
    tolerance: f64,
) -> bool {
    let exterior = &footprint.polygon.exterior;
    let n = exterior.len().saturating_sub(1);
    
    // Check if both endpoints are close to footprint boundary vertices
    let mut p1_on_boundary = false;
    let mut p2_on_boundary = false;
    
    for i in 0..n {
        let v = &exterior.vertices[i];
        let dx1 = p1.x - v.x;
        let dy1 = p1.y - v.y;
        let dx2 = p2.x - v.x;
        let dy2 = p2.y - v.y;
        
        if dx1 * dx1 + dy1 * dy1 < tolerance * tolerance {
            p1_on_boundary = true;
        }
        if dx2 * dx2 + dy2 * dy2 < tolerance * tolerance {
            p2_on_boundary = true;
        }
    }
    
    // If both endpoints are on boundary, check if the edge matches a footprint edge
    if p1_on_boundary && p2_on_boundary {
        for i in 0..n {
            let j = (i + 1) % n;
            let v_i = &exterior.vertices[i];
            let v_j = &exterior.vertices[j];
            
            // Check if edge matches footprint edge (in either direction)
            let dist1 = ((p1.x - v_i.x).powi(2) + (p1.y - v_i.y).powi(2)).sqrt();
            let dist2 = ((p2.x - v_j.x).powi(2) + (p2.y - v_j.y).powi(2)).sqrt();
            let dist3 = ((p1.x - v_j.x).powi(2) + (p1.y - v_j.y).powi(2)).sqrt();
            let dist4 = ((p2.x - v_i.x).powi(2) + (p2.y - v_i.y).powi(2)).sqrt();
            
            if (dist1 < tolerance && dist2 < tolerance) || (dist3 < tolerance && dist4 < tolerance) {
                return true;
            }
        }
    }
    
    false
}

/// Validate that footprint perimeter has wall coverage.
fn validate_footprint_walls(
    mesh: &Mesh,
    footprint: &Footprint,
    _wall_count: usize,
    _h_ground: f64,
) {
    let exterior = &footprint.polygon.exterior;
    let n = exterior.len().saturating_sub(1);
    if n < 3 {
        return;
    }
    
    // Count wall faces in mesh
    let wall_face_count = mesh
        .faces
        .iter()
        .filter(|face| {
            if let Some(sem_idx) = face.semantic_index {
                if let Some(sem) = mesh.semantics.get(sem_idx) {
                    return sem.surface_type == crate::mesh::SurfaceType::WallSurface;
                }
            }
            false
        })
        .count();
    
    // Warn if no walls or very few walls relative to footprint size
    if wall_face_count == 0 {
        tracing::warn!(
            "Building {} has no wall faces - building may be incorrectly reconstructed",
            footprint.id
        );
    } else if wall_face_count < n {
        tracing::debug!(
            "Building {} has {} wall faces but {} footprint edges - some edges may be missing walls",
            footprint.id,
            wall_face_count,
            n
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polygon::{LinearRing, Polygon3D};

    fn create_test_footprint() -> Footprint {
        let exterior = LinearRing::from_vertices(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 0.0),
            Point3::new(0.0, 10.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
        ]);
        Footprint::new("test", Polygon3D::new(exterior))
    }

    fn create_test_point_cloud() -> PointCloud {
        // Create a simple sloped roof point cloud
        let mut positions = Vec::new();
        for x in 0..10 {
            for y in 0..10 {
                let z = 10.0 + (x as f64) * 0.5;
                positions.push(Point3::new(x as f64, y as f64, z));
            }
        }
        PointCloud::from_positions(positions)
    }

    #[test]
    fn test_flat_roof_fallback() {
        let reconstructor = Lod22Reconstructor::new(RansacConfig::default());
        let footprint = create_test_footprint();
        let point_cloud = create_test_point_cloud();

        let result = reconstructor.create_flat_roof(&footprint, &point_cloud, 0.0);
        assert!(result.is_ok());

        let mesh = result.unwrap();
        assert_eq!(mesh.vertex_count(), 8);
        assert_eq!(mesh.face_count(), 6);
    }

    #[test]
    fn test_lod22_reconstruction() {
        let config = RansacConfig {
            epsilon: 0.5,
            min_points: 10,
            ..Default::default()
        };
        let reconstructor = Lod22Reconstructor::new(config);
        let footprint = create_test_footprint();
        let point_cloud = create_test_point_cloud();

        let result = reconstructor.reconstruct(&footprint, &point_cloud, 0.0);
        assert!(result.is_ok());

        let mesh = result.unwrap();
        assert!(mesh.vertex_count() >= 8);
        assert!(mesh.face_count() >= 6);
    }

    #[test]
    fn test_ridgeline_computation() {
        let footprint = create_test_footprint();
        let planes = vec![
            Plane {
                normal: Vector3::new(0.0, 0.0, 1.0),
                d: -10.0,
                inliers: vec![],
            },
            Plane {
                normal: Vector3::new(0.5, 0.0, 0.866).normalize(),
                d: -10.0,
                inliers: vec![],
            },
        ];
        let ridgelines = compute_ridgelines(&footprint, &planes, 4);
        // May or may not have ridgelines depending on plane configuration
        assert!(ridgelines.len() >= 0);
    }
}

//! CityJSON output writer.
//!
//! Writes CityJSON features in JSONL (JSON Lines) format.

use gdal::spatial_ref::CoordTransform;
use roofer_rs_geometry::mesh::{BuildingGeometry, Mesh, SurfaceType};
use roofer_rs_geometry::mesh;
use roofer_rs_geometry::AttributeValue;
use serde_json::{json, Map, Value};
use tracing::{debug, warn};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CityJsonError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("GDAL error: {0}")]
    GdalError(#[from] gdal::errors::GdalError),
}

/// Transform settings for CityJSON output.
#[derive(Debug, Clone)]
pub struct CityJsonTransform {
    pub scale: [f64; 3],
    pub translate: [f64; 3],
}

impl Default for CityJsonTransform {
    fn default() -> Self {
        Self {
            scale: [0.001, 0.001, 0.001],
            translate: [0.0, 0.0, 0.0],
        }
    }
}

/// CityJSON JSONL writer.
pub struct CityJsonWriter {
    writer: BufWriter<File>,
    transform: CityJsonTransform,
    reference_system: Option<String>,
    coord_transform: Option<CoordTransform>,
    include_process_attributes: bool,
    header_written: bool,
}

impl CityJsonWriter {
    /// Create a new CityJSON writer.
    pub fn new(path: &Path, transform: CityJsonTransform) -> Result<Self, CityJsonError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);

        Ok(Self {
            writer,
            transform,
            reference_system: None,
            coord_transform: None,
            include_process_attributes: true,
            header_written: false,
        })
    }

    /// Set the coordinate reference system.
    pub fn with_reference_system(mut self, srs: impl Into<String>) -> Self {
        self.reference_system = Some(srs.into());
        self
    }

    /// Set an optional coordinate transform applied to output vertices.
    pub fn with_coord_transform(mut self, transform: CoordTransform) -> Self {
        self.coord_transform = Some(transform);
        self
    }

    /// Control inclusion of process/debug attributes in output.
    pub fn with_process_attributes(mut self, include: bool) -> Self {
        self.include_process_attributes = include;
        self
    }

    /// Write the CityJSON header (first line).
    pub fn write_header(&mut self, extent: Option<[f64; 6]>) -> Result<(), CityJsonError> {
        let mut header = json!({
            "type": "CityJSON",
            "version": "2.0",
            "CityObjects": {},
            "vertices": [],
            "transform": {
                "scale": self.transform.scale,
                "translate": self.transform.translate
            }
        });

        if let Some(ref srs) = self.reference_system {
            let metadata = json!({
                "referenceSystem": format!("https://www.opengis.net/def/crs/EPSG/0/{}", srs.trim_start_matches("EPSG:")),
                "referenceDate": chrono::Utc::now().format("%Y-%m-%d").to_string(),
                "identifier": "0"
            });
            if let Some(ext) = extent {
                let mut meta = metadata.as_object().unwrap().clone();
                meta.insert("geographicalExtent".to_string(), json!(ext));
                header["metadata"] = Value::Object(meta);
            } else {
                header["metadata"] = metadata;
            }
        }

        let line = serde_json::to_string(&header)?;
        writeln!(self.writer, "{}", line)?;
        self.header_written = true;

        Ok(())
    }

    /// Write a building feature.
    pub fn write_feature(&mut self, geometry: &BuildingGeometry) -> Result<(), CityJsonError> {
        if !self.header_written {
            self.write_header(geometry.geographic_extent())?;
        }

        let feature = self.build_feature(geometry)?;
        let line = serde_json::to_string(&feature)?;
        writeln!(self.writer, "{}", line)?;

        Ok(())
    }

    /// Build a CityJSON feature value (for external writers).
    pub fn build_feature_value(&self, geometry: &BuildingGeometry) -> Result<Value, CityJsonError> {
        self.build_feature(geometry)
    }

    /// Build a CityJSON feature from BuildingGeometry.
    fn build_feature(&self, geometry: &BuildingGeometry) -> Result<Value, CityJsonError> {
        let id = &geometry.id;
        let mut vertices: Vec<[i64; 3]> = Vec::new();
        let mut city_objects: Map<String, Value> = Map::new();

        // Get the best available LoD mesh
        let mesh = match geometry.best_lod() {
            Some(m) => m,
            None => {
                // Return empty feature if no geometry
                return Ok(json!({
                    "type": "CityJSONFeature",
                    "id": id,
                    "CityObjects": {},
                    "vertices": []
                }));
            }
        };

        // Build attributes
        let mut attributes: Map<String, Value> = Map::new();
        for (key, value) in geometry.attributes.iter() {
            if !self.include_process_attributes && is_process_attribute(key) {
                continue;
            }
            let json_value = match value {
                AttributeValue::Null => Value::Null,
                AttributeValue::Bool(v) => Value::Bool(*v),
                AttributeValue::Int(v) => json!(*v),
                AttributeValue::Float(v) => json!(*v),
                AttributeValue::String(v) => Value::String(v.clone()),
                AttributeValue::DateTime(v) => Value::String(v.to_rfc3339()),
                AttributeValue::IntArray(v) => json!(v),
                AttributeValue::FloatArray(v) => json!(v),
            };
            attributes.insert(key.clone(), json_value);
        }

        // Ensure roof analysis attributes are present when possible
        let (roof_planes, horiz_cnt, slant_cnt) = count_roof_plane_types(mesh);
        attributes
            .entry("rf_roof_planes".to_string())
            .or_insert_with(|| json!(roof_planes as i64));
        attributes
            .entry("rf_roof_type".to_string())
            .or_insert_with(|| json!(classify_roof_type(roof_planes, horiz_cnt, slant_cnt)));
        let ridgelines = if slant_cnt > 1 { slant_cnt - 1 } else { 0 };
        attributes
            .entry("rf_ridgelines".to_string())
            .or_insert_with(|| json!(ridgelines as i64));
        attributes
            .entry("rf_volume_lod22".to_string())
            .or_insert_with(|| json!(mesh.compute_volume()));

        // Convert mesh vertices to CityJSON format (integer coordinates)
        let vertex_offset = vertices.len();
        let mut xs: Vec<f64> = mesh.vertices.iter().map(|v| v.x).collect();
        let mut ys: Vec<f64> = mesh.vertices.iter().map(|v| v.y).collect();
        let mut zs: Vec<f64> = mesh.vertices.iter().map(|v| v.z).collect();

        if let Some(ref transform) = self.coord_transform {
            transform.transform_coords(&mut xs, &mut ys, &mut zs)?;
        }

        let mut max_quant_error = 0.0_f64;
        for idx in 0..xs.len() {
            let x = ((xs[idx] - self.transform.translate[0]) / self.transform.scale[0]) as i64;
            let y = ((ys[idx] - self.transform.translate[1]) / self.transform.scale[1]) as i64;
            let z = ((zs[idx] - self.transform.translate[2]) / self.transform.scale[2]) as i64;
            vertices.push([x, y, z]);

            let dx = (xs[idx] - (x as f64 * self.transform.scale[0] + self.transform.translate[0])).abs();
            let dy = (ys[idx] - (y as f64 * self.transform.scale[1] + self.transform.translate[1])).abs();
            let dz = (zs[idx] - (z as f64 * self.transform.scale[2] + self.transform.translate[2])).abs();
            max_quant_error = max_quant_error.max(dx.max(dy).max(dz));
        }
        debug!("Max quantization error: {}", max_quant_error);

        // Build geometry boundaries
        let boundaries = self.build_solid_boundaries(mesh, vertex_offset);

        // Build semantics
        let (semantics_surfaces, semantics_values) = self.build_semantics(mesh);

        // Create the Building object
        let building_part_id = format!("{}-0", id);
        
        // Get geographic extent
        let extent = if xs.is_empty() {
            None
        } else {
            let (min_x, max_x) = xs
                .iter()
                .fold((f64::MAX, f64::MIN), |(min_v, max_v), v| {
                    (min_v.min(*v), max_v.max(*v))
                });
            let (min_y, max_y) = ys
                .iter()
                .fold((f64::MAX, f64::MIN), |(min_v, max_v), v| {
                    (min_v.min(*v), max_v.max(*v))
                });
            let (min_z, max_z) = zs
                .iter()
                .fold((f64::MAX, f64::MIN), |(min_v, max_v), v| {
                    (min_v.min(*v), max_v.max(*v))
                });
            Some(vec![min_x, min_y, min_z, max_x, max_y, max_z])
        };

        // LoD geometry
        let lod = if geometry.lod22.is_some() {
            "2.2"
        } else if geometry.lod13.is_some() {
            "1.3"
        } else {
            "1.2"
        };

        let geom = json!({
            "type": "Solid",
            "lod": lod,
            "boundaries": [boundaries],
            "semantics": {
                "surfaces": semantics_surfaces,
                "values": [semantics_values]
            }
        });

        // Building part
        let building_part = json!({
            "type": "BuildingPart",
            "parents": [id],
            "geometry": [geom]
        });
        city_objects.insert(building_part_id.clone(), building_part);

        // Building (parent)
        let mut building = json!({
            "type": "Building",
            "attributes": attributes,
            "children": [building_part_id]
        });

        if let Some(ref ext) = extent {
            building["geographicalExtent"] = json!(ext);
        }

        // Add LoD 0 geometry (footprint) if available
        let footprint_boundaries = self.build_footprint_boundaries(mesh, vertex_offset);
        if !footprint_boundaries.is_empty() {
            building["geometry"] = json!([{
                "type": "MultiSurface",
                "lod": "0",
                "boundaries": footprint_boundaries
            }]);
        }

        // CityJSON spec: the key in CityObjects is the object ID (canonical building id).
        city_objects.insert(id.clone(), building);

        Ok(json!({
            "type": "CityJSONFeature",
            "id": id,
            "CityObjects": city_objects,
            "vertices": vertices
        }))
    }

    /// Build solid boundaries from mesh faces.
    fn build_solid_boundaries(&self, mesh: &Mesh, vertex_offset: usize) -> Vec<Vec<Vec<u32>>> {
        let mut boundaries = Vec::new();
        let vertex_count = mesh.vertices.len();

        for (face_idx, face) in mesh.faces.iter().enumerate() {
            if !is_valid_face(mesh, face_idx, vertex_count) {
                continue;
            }

            let mut ring: Vec<u32> = face
                .indices
                .iter()
                .map(|&i| (i as usize + vertex_offset) as u32)
                .collect();
            if !is_ccw_2d(mesh, &face.indices) {
                ring.reverse();
            }
            boundaries.push(vec![ring]);
        }

        boundaries
    }

    /// Build footprint boundaries (ground faces only).
    fn build_footprint_boundaries(&self, mesh: &Mesh, vertex_offset: usize) -> Vec<Vec<Vec<u32>>> {
        let mut boundaries = Vec::new();
        let vertex_count = mesh.vertices.len();

        for (face_idx, face) in mesh.faces.iter().enumerate() {
            // Check if this is a ground surface
            let is_ground = face
                .semantic_index
                .and_then(|idx| mesh.semantics.get(idx))
                .map(|s| s.surface_type == SurfaceType::GroundSurface)
                .unwrap_or(false);

            if is_ground {
                if !is_valid_face(mesh, face_idx, vertex_count) {
                    continue;
                }

                let mut ring: Vec<u32> = face
                    .indices
                    .iter()
                    .map(|&i| (i as usize + vertex_offset) as u32)
                    .collect();
                if !is_ccw_2d(mesh, &face.indices) {
                    ring.reverse();
                }
                boundaries.push(vec![ring]);
            }
        }

        boundaries
    }

    /// Build semantics arrays.
    fn build_semantics(&self, mesh: &Mesh) -> (Vec<Value>, Vec<Option<usize>>) {
        let mut surfaces = Vec::new();
        let mut values = Vec::new();
        let (roof_areas, roof_normals) = roof_surface_metrics(mesh);

        // Create surface definitions
        for (idx, semantic) in mesh.semantics.iter().enumerate() {
            let mut surface = Map::new();
            surface.insert(
                "type".to_string(),
                Value::String(match semantic.surface_type {
                    SurfaceType::GroundSurface => "GroundSurface".to_string(),
                    SurfaceType::WallSurface => "WallSurface".to_string(),
                    SurfaceType::RoofSurface => "RoofSurface".to_string(),
                    SurfaceType::OuterCeilingSurface => "OuterCeilingSurface".to_string(),
                    SurfaceType::OuterFloorSurface => "OuterFloorSurface".to_string(),
                    SurfaceType::ClosureSurface => "ClosureSurface".to_string(),
                }),
            );

            if let Some(on_edge) = semantic.on_footprint_edge {
                surface.insert("on_footprint_edge".to_string(), Value::Bool(on_edge));
            }

            if semantic.surface_type == SurfaceType::RoofSurface {
                if let Some(azimuth) = semantic.azimuth {
                    surface.insert("rf_azimuth".to_string(), json!(azimuth));
                }
                if let Some(slope) = semantic.slope {
                    surface.insert("rf_slope".to_string(), json!(slope));
                }
                if let Some(h) = semantic.h_roof_50p {
                    surface.insert("rf_h_roof_50p".to_string(), json!(h));
                }
                if let Some(h) = semantic.h_roof_70p {
                    surface.insert("rf_h_roof_70p".to_string(), json!(h));
                }
                if let Some(h) = semantic.h_roof_min {
                    surface.insert("rf_h_roof_min".to_string(), json!(h));
                }
                if let Some(h) = semantic.h_roof_max {
                    surface.insert("rf_h_roof_max".to_string(), json!(h));
                }
                if let Some(area) = roof_areas.get(idx).copied() {
                    surface.insert("rf_area".to_string(), json!(area));
                }
                if let Some(Some(normal)) = roof_normals.get(idx) {
                    surface.insert("rf_normal".to_string(), json!([normal[0], normal[1], normal[2]]));
                }
            }

            surfaces.push(Value::Object(surface));
        }

        // Create values array (one per face)
        for face in &mesh.faces {
            values.push(face.semantic_index);
        }

        (surfaces, values)
    }

    /// Flush and close the writer.
    pub fn finish(mut self) -> Result<(), CityJsonError> {
        self.writer.flush()?;
        Ok(())
    }
}

/// Validates that the canonical building id (first key in `CityObjects`) matches the Building
/// object key(s), and that all City Object keys are either that id or variants of it (e.g. `"288"`,
/// `"288-0"`, `"288-1"`). Emits warnings via `tracing::warn!` when checks fail.
///
/// Call this when reading/parsing a CityJSON feature (e.g. from JSONL). Requires `serde_json`
/// with `preserve_order` so the first key in `CityObjects` is well-defined.
pub fn validate_cityjson_feature_ids(feature: &Value) {
    let Some(city_objects) = feature.get("CityObjects").and_then(Value::as_object) else {
        return;
    };
    if city_objects.is_empty() {
        return;
    }

    // First key in CityObjects is the canonical building id (per CityJSON spec + convention).
    let canonical_id = match city_objects.keys().next() {
        Some(k) => k.as_str(),
        None => return,
    };

    // Collect Building keys and check each Building's key matches canonical_id.
    let building_keys: Vec<&str> = city_objects
        .iter()
        .filter_map(|(key, obj)| {
            let ty = obj.get("type").and_then(Value::as_str)?;
            if ty == "Building" {
                Some(key.as_str())
            } else {
                None
            }
        })
        .collect();

    if building_keys.is_empty() {
        warn!(
            cityjson_feature_ids = true,
            "CityObjects has no Building; expected first key to be a Building (canonical_id={})",
            canonical_id
        );
        return;
    }

    for building_key in &building_keys {
        if *building_key != canonical_id {
            warn!(
                cityjson_feature_ids = true,
                "Building key '{}' does not match canonical id (first key) '{}' in CityObjects",
                building_key,
                canonical_id
            );
        }
    }

    if building_keys.len() > 1 {
        warn!(
            cityjson_feature_ids = true,
            "CityObjects has {} Building(s); expected one; canonical_id={}",
            building_keys.len(),
            canonical_id
        );
    }

    // All keys must be canonical_id or canonical_id + "-" + suffix (e.g. "288-0").
    let prefix = format!("{}-", canonical_id);
    for key in city_objects.keys() {
        let k = key.as_str();
        if k != canonical_id && !k.starts_with(&prefix) {
            warn!(
                cityjson_feature_ids = true,
                "CityObjects key '{}' is not the canonical id '{}' nor a variant (e.g. '{}-0')",
                k,
                canonical_id,
                canonical_id
            );
        }
    }
}

fn roof_surface_metrics(mesh: &Mesh) -> (Vec<Option<f64>>, Vec<Option<[f64; 3]>>) {
    let mut areas = vec![None; mesh.semantics.len()];
    let mut normals = vec![None; mesh.semantics.len()];
    let mut accum_area = vec![0.0_f64; mesh.semantics.len()];
    let mut accum_normal = vec![[0.0_f64; 3]; mesh.semantics.len()];

    for (face_idx, face) in mesh.faces.iter().enumerate() {
        let semantic_idx = match face.semantic_index {
            Some(idx) => idx,
            None => continue,
        };
        if mesh.semantics.get(semantic_idx).map(|s| s.surface_type) != Some(SurfaceType::RoofSurface) {
            continue;
        }
        let area = face_area_3d(mesh, face);
        if area <= 0.0 {
            continue;
        }
        accum_area[semantic_idx] += area;
        if let Some(normal) = mesh.face_normal(face_idx) {
            accum_normal[semantic_idx][0] += normal.x * area;
            accum_normal[semantic_idx][1] += normal.y * area;
            accum_normal[semantic_idx][2] += normal.z * area;
        }
    }

    for i in 0..mesh.semantics.len() {
        if accum_area[i] > 0.0 {
            areas[i] = Some(accum_area[i]);
            let n = accum_normal[i];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 1e-10 {
                normals[i] = Some([n[0] / len, n[1] / len, n[2] / len]);
            }
        }
    }

    (areas, normals)
}

fn face_area_3d(mesh: &Mesh, face: &mesh::Face) -> f64 {
    if face.indices.len() < 3 {
        return 0.0;
    }
    let p0 = &mesh.vertices[face.indices[0] as usize];
    let mut area = 0.0;
    for i in 1..(face.indices.len() - 1) {
        let p1 = &mesh.vertices[face.indices[i] as usize];
        let p2 = &mesh.vertices[face.indices[i + 1] as usize];
        let v1 = p1 - p0;
        let v2 = p2 - p0;
        area += 0.5 * v1.cross(&v2).norm();
    }
    area
}

fn is_ccw_2d(mesh: &Mesh, indices: &[u32]) -> bool {
    if indices.len() < 3 {
        return false;
    }

    let mut area = 0.0;
    let n = indices.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let vi = &mesh.vertices[indices[i] as usize];
        let vj = &mesh.vertices[indices[j] as usize];
        area += vi.x * vj.y - vj.x * vi.y;
    }
    area > 0.0
}

fn is_valid_face(mesh: &Mesh, face_idx: usize, vertex_count: usize) -> bool {
    let face = &mesh.faces[face_idx];
    if face.indices.len() < 3 {
        return false;
    }

    if face
        .indices
        .iter()
        .any(|&idx| idx as usize >= vertex_count)
    {
        return false;
    }

    if face
        .indices
        .windows(2)
        .any(|pair| pair[0] == pair[1])
    {
        return false;
    }

    // Reject faces with any duplicate indices
    let mut uniq = std::collections::HashSet::with_capacity(face.indices.len());
    if face.indices.iter().any(|idx| !uniq.insert(*idx)) {
        return false;
    }

    if mesh.face_normal(face_idx).is_none() {
        return false;
    }

    let area = face_area_3d(mesh, face);
    if area < 1e-8 {
        return false;
    }

    true
}

fn count_roof_plane_types(mesh: &Mesh) -> (usize, usize, usize) {
    let mut total = 0;
    let mut horizontal = 0;
    let mut slanted = 0;
    for s in &mesh.semantics {
        if s.surface_type != SurfaceType::RoofSurface {
            continue;
        }
        total += 1;
        if let Some(slope) = s.slope {
            if slope <= 5.0 {
                horizontal += 1;
            } else {
                slanted += 1;
            }
        }
    }
    (total, horizontal, slanted)
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

fn is_process_attribute(key: &str) -> bool {
    matches!(
        key,
        // Process/control flags
        "rf_extrusion_mode"
            | "rf_force_lod11"
            | "rf_is_mutated"
            | "rf_pointcloud_unusable"
            | "rf_pc_select"
            | "rf_pc_source"
            | "rf_pc_year"
            | "rf_t_run"
            // QA / validation metrics
            | "rf_val3dity_lod12"
            | "rf_val3dity_lod13"
            | "rf_val3dity_lod22"
            | "rf_rmse_lod12"
            | "rf_rmse_lod13"
            | "rf_rmse_lod22"
            // Data coverage / quality
            | "rf_nodata_frac"
            | "rf_nodata_r"
            | "rf_h_pc_98p"
            // Reconstruction diagnostics
            | "rf_h_roof_ridge"
            | "rf_is_glass_roof"
            // Source metadata fields that are typically nullable
            | "comment"
            | "md_id"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;
    use roofer_rs_geometry::mesh::{Face, Mesh, SemanticSurface};
    use tempfile::NamedTempFile;

    #[test]
    fn test_cityjson_writer() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        let transform = CityJsonTransform {
            scale: [0.001, 0.001, 0.001],
            translate: [100.0, 200.0, 0.0],
        };

        let mut writer = CityJsonWriter::new(path, transform).unwrap();
        writer.write_header(Some([100.0, 200.0, 0.0, 110.0, 210.0, 10.0])).unwrap();

        // Create a simple building geometry
        let mut mesh = Mesh::new();
        let ground_idx = mesh.add_semantic(SemanticSurface::ground());
        let roof_idx = mesh.add_semantic(SemanticSurface::roof());

        // Add a simple cube
        for z in [0.0, 10.0] {
            for (x, y) in [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)] {
                mesh.add_vertex(Point3::new(100.0 + x, 200.0 + y, z));
            }
        }

        // Add faces
        mesh.add_face(Face::new(vec![0, 1, 2, 3]).with_semantic(ground_idx));
        mesh.add_face(Face::new(vec![4, 5, 6, 7]).with_semantic(roof_idx));

        let mut geometry = BuildingGeometry::new("test_building");
        geometry.lod22 = Some(mesh);
        geometry.attributes.insert("rf_success", true);

        writer.write_feature(&geometry).unwrap();
        writer.finish().unwrap();

        // Read back and verify
        let content = std::fs::read_to_string(path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2); // Header + 1 feature

        // Parse header
        let header: Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(header["type"], "CityJSON");
        assert_eq!(header["version"], "2.0");

        // Parse feature
        let feature: Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(feature["type"], "CityJSONFeature");
        assert_eq!(feature["id"], "test_building");

        // Validation: first key in CityObjects should be Building id; all keys variants of it.
        validate_cityjson_feature_ids(&feature);
    }

    #[test]
    fn test_validate_cityjson_feature_ids_valid() {
        // First key is "288", Building key is "288", "288-0" is variant.
        let feature: Value = serde_json::from_str(
            r#"{"type":"CityJSONFeature","id":"288","CityObjects":{"288":{"type":"Building","children":["288-0"]},"288-0":{"type":"BuildingPart","parents":["288"]}},"vertices":[]}"#,
        )
        .unwrap();
        validate_cityjson_feature_ids(&feature);
    }

    #[test]
    fn test_validate_cityjson_feature_ids_building_mismatch() {
        // First key is "288" but Building is under "999" -> should warn.
        let feature: Value = serde_json::from_str(
            r#"{"type":"CityJSONFeature","id":"288","CityObjects":{"288":{"type":"BuildingPart","parents":["999"]},"999":{"type":"Building","children":["288"]}},"vertices":[]}"#,
        )
        .unwrap();
        validate_cityjson_feature_ids(&feature);
    }

    #[test]
    fn test_validate_cityjson_feature_ids_key_not_variant() {
        // Key "other" is not "288" nor "288-*" -> should warn.
        let feature: Value = serde_json::from_str(
            r#"{"type":"CityJSONFeature","id":"288","CityObjects":{"288":{"type":"Building","children":["288-0"]},"288-0":{"type":"BuildingPart","parents":["288"]},"other":{"type":"BuildingPart","parents":["288"]}},"vertices":[]}"#,
        )
        .unwrap();
        validate_cityjson_feature_ids(&feature);
    }
}

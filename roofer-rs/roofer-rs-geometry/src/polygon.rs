//! 2D and 3D polygon representations.
//!
//! Provides types for building footprints and roof surfaces.

use geo_types::{Coord, LineString as GeoLineString, Polygon as GeoPolygon};
use nalgebra::{Point2, Point3, Vector3};
use rstar::{RTreeObject, AABB};
use serde::{Deserialize, Serialize};

/// A 3D linear ring (closed polyline).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LinearRing {
    /// Vertices of the ring (first and last should be the same for closed ring)
    pub vertices: Vec<Point3<f64>>,
}

impl LinearRing {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_vertices(vertices: Vec<Point3<f64>>) -> Self {
        Self { vertices }
    }

    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    pub fn push(&mut self, vertex: Point3<f64>) {
        self.vertices.push(vertex);
    }

    /// Check if the ring is closed (first and last vertex are the same).
    pub fn is_closed(&self) -> bool {
        if self.vertices.len() < 2 {
            return false;
        }
        let first = &self.vertices[0];
        let last = &self.vertices[self.vertices.len() - 1];
        (first.x - last.x).abs() < 1e-10
            && (first.y - last.y).abs() < 1e-10
            && (first.z - last.z).abs() < 1e-10
    }

    /// Close the ring by adding the first vertex at the end if needed.
    pub fn close(&mut self) {
        if !self.is_closed() && !self.vertices.is_empty() {
            self.vertices.push(self.vertices[0]);
        }
    }

    /// Project to 2D (drop Z coordinate).
    pub fn to_2d(&self) -> Vec<Point2<f64>> {
        self.vertices.iter().map(|p| Point2::new(p.x, p.y)).collect()
    }

    /// Convert to geo_types LineString (2D).
    pub fn to_geo_linestring(&self) -> GeoLineString<f64> {
        let coords: Vec<Coord<f64>> = self
            .vertices
            .iter()
            .map(|p| Coord { x: p.x, y: p.y })
            .collect();
        GeoLineString::new(coords)
    }

    /// Compute the 2D bounding box.
    pub fn bounding_box_2d(&self) -> Option<(Point2<f64>, Point2<f64>)> {
        if self.vertices.is_empty() {
            return None;
        }

        let mut min = Point2::new(f64::MAX, f64::MAX);
        let mut max = Point2::new(f64::MIN, f64::MIN);

        for v in &self.vertices {
            min.x = min.x.min(v.x);
            min.y = min.y.min(v.y);
            max.x = max.x.max(v.x);
            max.y = max.y.max(v.y);
        }

        Some((min, max))
    }

    /// Compute 2D signed area (positive for counter-clockwise).
    pub fn signed_area_2d(&self) -> f64 {
        if self.vertices.len() < 3 {
            return 0.0;
        }

        let mut area = 0.0;
        let n = self.vertices.len();
        for i in 0..n {
            let j = (i + 1) % n;
            area += self.vertices[i].x * self.vertices[j].y;
            area -= self.vertices[j].x * self.vertices[i].y;
        }
        area / 2.0
    }

    /// Check if the ring is counter-clockwise (when viewed from above).
    pub fn is_ccw(&self) -> bool {
        self.signed_area_2d() > 0.0
    }

    /// Reverse the vertex order.
    pub fn reverse(&mut self) {
        self.vertices.reverse();
    }

    /// Compute centroid (2D).
    pub fn centroid_2d(&self) -> Option<Point2<f64>> {
        if self.vertices.is_empty() {
            return None;
        }

        let sum: Point2<f64> = self
            .vertices
            .iter()
            .fold(Point2::origin(), |acc, p| Point2::new(acc.x + p.x, acc.y + p.y));

        let n = self.vertices.len() as f64;
        Some(Point2::new(sum.x / n, sum.y / n))
    }

    /// Compute the normal vector (for a planar ring).
    pub fn compute_normal(&self) -> Option<Vector3<f64>> {
        if self.vertices.len() < 3 {
            return None;
        }

        // Use Newell's method for robustness
        let mut normal: Vector3<f64> = Vector3::zeros();
        let n = self.vertices.len();

        for i in 0..n {
            let curr = &self.vertices[i];
            let next = &self.vertices[(i + 1) % n];

            normal.x += (curr.y - next.y) * (curr.z + next.z);
            normal.y += (curr.z - next.z) * (curr.x + next.x);
            normal.z += (curr.x - next.x) * (curr.y + next.y);
        }

        let norm = normal.norm();
        if norm < 1e-10 {
            return None;
        }

        Some(normal / norm)
    }
}

/// A 3D polygon with an exterior ring and optional interior rings (holes).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Polygon3D {
    /// Exterior boundary ring
    pub exterior: LinearRing,
    /// Interior rings (holes)
    pub interiors: Vec<LinearRing>,
}

impl Polygon3D {
    pub fn new(exterior: LinearRing) -> Self {
        Self {
            exterior,
            interiors: Vec::new(),
        }
    }

    pub fn with_interiors(exterior: LinearRing, interiors: Vec<LinearRing>) -> Self {
        Self { exterior, interiors }
    }

    /// Add an interior ring (hole).
    pub fn add_interior(&mut self, ring: LinearRing) {
        self.interiors.push(ring);
    }

    /// Check if the polygon is empty.
    pub fn is_empty(&self) -> bool {
        self.exterior.is_empty()
    }

    /// Convert to geo_types Polygon (2D).
    pub fn to_geo_polygon(&self) -> GeoPolygon<f64> {
        let exterior = self.exterior.to_geo_linestring();
        let interiors: Vec<GeoLineString<f64>> = self
            .interiors
            .iter()
            .map(|r| r.to_geo_linestring())
            .collect();
        GeoPolygon::new(exterior, interiors)
    }

    /// Compute 2D bounding box.
    pub fn bounding_box_2d(&self) -> Option<(Point2<f64>, Point2<f64>)> {
        self.exterior.bounding_box_2d()
    }

    /// Compute the signed area (2D).
    pub fn signed_area_2d(&self) -> f64 {
        let mut area = self.exterior.signed_area_2d();
        for interior in &self.interiors {
            area -= interior.signed_area_2d().abs();
        }
        area
    }

    /// Compute the area (absolute value).
    pub fn area_2d(&self) -> f64 {
        self.signed_area_2d().abs()
    }

    /// Check if a 2D point is inside the polygon.
    pub fn contains_2d(&self, point: &Point2<f64>) -> bool {
        // Use ray casting algorithm
        if !self.point_in_ring_2d(point, &self.exterior) {
            return false;
        }

        // Check if point is in any hole
        for interior in &self.interiors {
            if self.point_in_ring_2d(point, interior) {
                return false;
            }
        }

        true
    }

    /// Ray casting point-in-polygon test for a single ring.
    fn point_in_ring_2d(&self, point: &Point2<f64>, ring: &LinearRing) -> bool {
        let vertices = &ring.vertices;
        if vertices.len() < 3 {
            return false;
        }

        let mut inside = false;
        let n = vertices.len();

        let mut j = n - 1;
        for i in 0..n {
            let vi = &vertices[i];
            let vj = &vertices[j];

            if ((vi.y > point.y) != (vj.y > point.y))
                && (point.x < (vj.x - vi.x) * (point.y - vi.y) / (vj.y - vi.y) + vi.x)
            {
                inside = !inside;
            }
            j = i;
        }

        inside
    }
}

/// A building footprint with ID and attributes.
#[derive(Debug, Clone)]
pub struct Footprint {
    /// Unique identifier
    pub id: String,
    /// The footprint polygon
    pub polygon: Polygon3D,
    /// Attributes from the source file
    pub attributes: crate::AttributeMap,
}

impl Footprint {
    pub fn new(id: impl Into<String>, polygon: Polygon3D) -> Self {
        Self {
            id: id.into(),
            polygon,
            attributes: crate::AttributeMap::new(),
        }
    }

    pub fn with_attributes(mut self, attributes: crate::AttributeMap) -> Self {
        self.attributes = attributes;
        self
    }

    /// Get the 2D bounding box.
    pub fn bounding_box_2d(&self) -> Option<(Point2<f64>, Point2<f64>)> {
        self.polygon.bounding_box_2d()
    }

    /// Check if a point (x, y) is inside the footprint.
    pub fn contains_2d(&self, x: f64, y: f64) -> bool {
        self.polygon.contains_2d(&Point2::new(x, y))
    }
}

/// Implement RTreeObject for spatial indexing of footprints.
impl RTreeObject for Footprint {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        if let Some((min, max)) = self.bounding_box_2d() {
            AABB::from_corners([min.x, min.y], [max.x, max.y])
        } else {
            AABB::from_point([0.0, 0.0])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_ring_area() {
        // Simple square: (0,0) -> (1,0) -> (1,1) -> (0,1) -> (0,0)
        let ring = LinearRing::from_vertices(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
        ]);

        let area = ring.signed_area_2d();
        assert!((area.abs() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_polygon_contains() {
        let exterior = LinearRing::from_vertices(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 0.0),
            Point3::new(0.0, 10.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
        ]);

        let polygon = Polygon3D::new(exterior);

        assert!(polygon.contains_2d(&Point2::new(5.0, 5.0)));
        assert!(!polygon.contains_2d(&Point2::new(15.0, 5.0)));
        assert!(!polygon.contains_2d(&Point2::new(-1.0, 5.0)));
    }

    #[test]
    fn test_polygon_with_hole() {
        let exterior = LinearRing::from_vertices(vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 0.0),
            Point3::new(0.0, 10.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
        ]);

        let hole = LinearRing::from_vertices(vec![
            Point3::new(3.0, 3.0, 0.0),
            Point3::new(7.0, 3.0, 0.0),
            Point3::new(7.0, 7.0, 0.0),
            Point3::new(3.0, 7.0, 0.0),
            Point3::new(3.0, 3.0, 0.0),
        ]);

        let polygon = Polygon3D::with_interiors(exterior, vec![hole]);

        // Point inside exterior but outside hole
        assert!(polygon.contains_2d(&Point2::new(1.0, 1.0)));
        // Point inside hole (should be outside polygon)
        assert!(!polygon.contains_2d(&Point2::new(5.0, 5.0)));
    }
}

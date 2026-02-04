//! Attribute storage for building features.
//!
//! Provides nullable attribute values compatible with CityJSON output.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single attribute value that can be null or one of several types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AttributeValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    DateTime(DateTime<Utc>),
    IntArray(Vec<i64>),
    FloatArray(Vec<f64>),
}

impl AttributeValue {
    pub fn is_null(&self) -> bool {
        matches!(self, AttributeValue::Null)
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            AttributeValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            AttributeValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            AttributeValue::Float(v) => Some(*v),
            AttributeValue::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            AttributeValue::String(v) => Some(v.as_str()),
            _ => None,
        }
    }
}

impl From<bool> for AttributeValue {
    fn from(v: bool) -> Self {
        AttributeValue::Bool(v)
    }
}

impl From<i64> for AttributeValue {
    fn from(v: i64) -> Self {
        AttributeValue::Int(v)
    }
}

impl From<i32> for AttributeValue {
    fn from(v: i32) -> Self {
        AttributeValue::Int(v as i64)
    }
}

impl From<f64> for AttributeValue {
    fn from(v: f64) -> Self {
        AttributeValue::Float(v)
    }
}

impl From<f32> for AttributeValue {
    fn from(v: f32) -> Self {
        AttributeValue::Float(v as f64)
    }
}

impl From<String> for AttributeValue {
    fn from(v: String) -> Self {
        AttributeValue::String(v)
    }
}

impl From<&str> for AttributeValue {
    fn from(v: &str) -> Self {
        AttributeValue::String(v.to_string())
    }
}

impl<T: Into<AttributeValue>> From<Option<T>> for AttributeValue {
    fn from(v: Option<T>) -> Self {
        match v {
            Some(val) => val.into(),
            None => AttributeValue::Null,
        }
    }
}

/// A map of attribute names to values for a single feature.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AttributeMap(HashMap<String, AttributeValue>);

impl AttributeMap {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<AttributeValue>) {
        self.0.insert(key.into(), value.into());
    }

    pub fn get(&self, key: &str) -> Option<&AttributeValue> {
        self.0.get(key)
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }

    pub fn remove(&mut self, key: &str) -> Option<AttributeValue> {
        self.0.remove(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &AttributeValue)> {
        self.0.iter()
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.0.keys()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Merge another AttributeMap into this one (overwrites existing keys).
    pub fn merge(&mut self, other: AttributeMap) {
        self.0.extend(other.0);
    }

    /// Convert to the inner HashMap.
    pub fn into_inner(self) -> HashMap<String, AttributeValue> {
        self.0
    }
}

impl FromIterator<(String, AttributeValue)> for AttributeMap {
    fn from_iter<T: IntoIterator<Item = (String, AttributeValue)>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl IntoIterator for AttributeMap {
    type Item = (String, AttributeValue);
    type IntoIter = std::collections::hash_map::IntoIter<String, AttributeValue>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attribute_value_conversions() {
        let v: AttributeValue = 42i64.into();
        assert_eq!(v.as_int(), Some(42));

        let v: AttributeValue = 3.14f64.into();
        assert_eq!(v.as_float(), Some(3.14));

        let v: AttributeValue = "hello".into();
        assert_eq!(v.as_str(), Some("hello"));

        let v: AttributeValue = true.into();
        assert_eq!(v.as_bool(), Some(true));

        let v: AttributeValue = Option::<i64>::None.into();
        assert!(v.is_null());
    }

    #[test]
    fn test_attribute_map() {
        let mut map = AttributeMap::new();
        map.insert("id", 123i64);
        map.insert("name", "building");
        map.insert("height", 10.5f64);

        assert_eq!(map.get("id").and_then(|v| v.as_int()), Some(123));
        assert_eq!(map.get("name").and_then(|v| v.as_str()), Some("building"));
        assert_eq!(map.get("height").and_then(|v| v.as_float()), Some(10.5));
    }
}

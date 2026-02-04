# Building roofer-rs (no Nix)

Plain `cargo` build. No Nix or other tooling required.

## Requirements

- **Rust** (e.g. `rustup`): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **GDAL** development headers and libraries (for vector/footprint I/O). The build uses `bindgen` and needs GDAL’s C headers (e.g. `cpl_atomic_ops.h`).
  - **Ubuntu/Debian (including WSL):** `sudo apt update && sudo apt install libgdal-dev pkg-config`
  - **Fedora:** `sudo dnf install gdal-devel pkg-config`
  - **macOS (Homebrew):** `brew install gdal pkg-config`

Optional (only if you enable the FCB feature):

- **OpenSSL** dev (Ubuntu: `libssl-dev`, Fedora: `openssl-devel`)

## Build

```bash
cd roofer-rs
cargo build --release
```

Binary: `target/release/roofer-rs`.

To enable FCB output (adds OpenSSL dependency; install `libssl-dev` first):

```bash
cargo build --release --features fcb
```

## CRS (Coordinate Reference System)

The CLI resolves the output CRS in this order:

1. `srs` in config (e.g. `srs = "25832"`)
2. CRS from the footprint vector file (GeoPackage/Shapefile layer)
3. CRS from the first point cloud file (LAS/LAZ header: GeoTIFF or WKT VLRs)

If none is found, CityJSON is written without `referenceSystem` metadata.

## Troubleshooting

### `cpl_atomic_ops.h` (or other GDAL header) not found

You need GDAL’s **development** package and headers, not only the runtime library.

- **Ubuntu/Debian / WSL:**  
  `sudo apt update && sudo apt install libgdal-dev pkg-config`  
  Then run `cargo build --release` again.

- If GDAL is installed in a custom prefix, point the build at it:  
  `export GDAL_INCLUDE_DIR=/path/to/gdal/include`  
  `export GDAL_LIB_DIR=/path/to/gdal/lib`  
  (or set `GDAL_HOME` if your layout uses a single root.)

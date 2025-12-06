//! Alpamayo WASM Module
//!
//! High-performance LiDAR BEV processing in WebAssembly.
//! Provides ~10x speedup over JavaScript for point cloud processing.

use wasm_bindgen::prelude::*;
use js_sys::{Float32Array, Uint8ClampedArray};
// use web_sys::ImageData;  // Reserved for future use

#[cfg(feature = "console_error_panic_hook")]
use console_error_panic_hook;

// Initialize panic hook for better error messages
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// BEV Grid configuration
#[wasm_bindgen]
pub struct BEVConfig {
    width: usize,
    height: usize,
    resolution: f32,
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
}

#[wasm_bindgen]
impl BEVConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            width: 240,
            height: 200,
            resolution: 0.05,  // 5cm
            x_min: -2.0,
            x_max: 10.0,
            y_min: -5.0,
            y_max: 5.0,
        }
    }

    pub fn set_grid_size(&mut self, width: usize, height: usize) {
        self.width = width;
        self.height = height;
    }

    pub fn set_range(&mut self, x_min: f32, x_max: f32, y_min: f32, y_max: f32) {
        self.x_min = x_min;
        self.x_max = x_max;
        self.y_min = y_min;
        self.y_max = y_max;
    }
}

/// LiDAR BEV Processor - High-performance point cloud to BEV grid conversion
#[wasm_bindgen]
pub struct LiDARProcessor {
    config: BEVConfig,
    occupancy: Vec<f32>,
    intensity: Vec<f32>,
    decay_factor: f32,
}

#[wasm_bindgen]
impl LiDARProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let config = BEVConfig::new();
        let size = config.width * config.height;
        Self {
            config,
            occupancy: vec![0.0; size],
            intensity: vec![0.0; size],
            decay_factor: 0.95,
        }
    }

    pub fn with_config(config: BEVConfig) -> Self {
        let size = config.width * config.height;
        Self {
            config,
            occupancy: vec![0.0; size],
            intensity: vec![0.0; size],
            decay_factor: 0.95,
        }
    }

    pub fn set_decay(&mut self, decay: f32) {
        self.decay_factor = decay.clamp(0.0, 1.0);
    }

    /// Process LaserScan data
    /// ranges: array of distances
    /// intensities: array of intensity values (optional, can be empty)
    /// angle_min, angle_increment: scan parameters
    #[wasm_bindgen]
    pub fn process_scan(
        &mut self,
        ranges: &Float32Array,
        intensities: &Float32Array,
        angle_min: f32,
        angle_increment: f32,
    ) -> usize {
        let cfg = &self.config;

        // Apply decay to existing grid
        for i in 0..self.occupancy.len() {
            self.occupancy[i] *= self.decay_factor;
            self.intensity[i] *= self.decay_factor;
        }

        let ranges_len = ranges.length() as usize;
        let has_intensities = intensities.length() > 0;

        let mut point_count = 0;

        for i in 0..ranges_len {
            let range = ranges.get_index(i as u32);

            // Skip invalid readings
            if range < 0.1 || range > 12.0 || !range.is_finite() {
                continue;
            }

            let angle = angle_min + (i as f32) * angle_increment;
            let x = range * angle.cos();
            let y = range * angle.sin();

            // Check bounds
            if x < cfg.x_min || x > cfg.x_max || y < cfg.y_min || y > cfg.y_max {
                continue;
            }

            // Convert to grid coordinates
            let gx = ((x - cfg.x_min) / cfg.resolution) as usize;
            let gy = ((y - cfg.y_min) / cfg.resolution) as usize;

            if gx < cfg.width && gy < cfg.height {
                let idx = gy * cfg.width + gx;
                self.occupancy[idx] = 1.0;

                if has_intensities {
                    self.intensity[idx] = intensities.get_index(i as u32);
                } else {
                    self.intensity[idx] = 0.5;
                }

                point_count += 1;
            }
        }

        point_count
    }

    /// Get occupancy grid as Float32Array
    #[wasm_bindgen]
    pub fn get_occupancy(&self) -> Float32Array {
        let arr = Float32Array::new_with_length(self.occupancy.len() as u32);
        for (i, &val) in self.occupancy.iter().enumerate() {
            arr.set_index(i as u32, val);
        }
        arr
    }

    /// Get intensity grid as Float32Array
    #[wasm_bindgen]
    pub fn get_intensity(&self) -> Float32Array {
        let arr = Float32Array::new_with_length(self.intensity.len() as u32);
        for (i, &val) in self.intensity.iter().enumerate() {
            arr.set_index(i as u32, val);
        }
        arr
    }

    /// Get combined 2-channel BEV tensor [occupancy, intensity]
    #[wasm_bindgen]
    pub fn get_bev_tensor(&self) -> Float32Array {
        let size = self.occupancy.len();
        let arr = Float32Array::new_with_length((size * 2) as u32);

        for (i, &val) in self.occupancy.iter().enumerate() {
            arr.set_index(i as u32, val);
        }
        for (i, &val) in self.intensity.iter().enumerate() {
            arr.set_index((size + i) as u32, val);
        }
        arr
    }

    /// Render BEV to RGBA image data (for canvas)
    #[wasm_bindgen]
    pub fn render_to_rgba(&self, color_mode: u8) -> Uint8ClampedArray {
        let size = self.occupancy.len();
        let arr = Uint8ClampedArray::new_with_length((size * 4) as u32);

        for i in 0..size {
            let occ = self.occupancy[i];
            let int = self.intensity[i];
            let idx = (i * 4) as u32;

            let (r, g, b, a) = match color_mode {
                // Distance-based (green to red)
                0 => {
                    if occ > 0.1 {
                        let h = (1.0 - int) * 120.0;  // Green to red
                        let (r, g, b) = hsv_to_rgb(h, 1.0, occ);
                        (r, g, b, 255)
                    } else {
                        (10, 10, 15, 255)
                    }
                }
                // Intensity-based (amber)
                1 => {
                    if occ > 0.1 {
                        let brightness = (int * 255.0) as u8;
                        (brightness, (brightness as f32 * 0.8) as u8, 0, 255)
                    } else {
                        (10, 10, 15, 255)
                    }
                }
                // Green
                _ => {
                    if occ > 0.1 {
                        let brightness = (occ * 255.0) as u8;
                        (34, brightness.max(100), 94, 255)
                    } else {
                        (10, 10, 15, 255)
                    }
                }
            };

            arr.set_index(idx, r);
            arr.set_index(idx + 1, g);
            arr.set_index(idx + 2, b);
            arr.set_index(idx + 3, a);
        }

        arr
    }

    /// Count occupied cells
    #[wasm_bindgen]
    pub fn count_occupied(&self) -> usize {
        self.occupancy.iter().filter(|&&v| v > 0.1).count()
    }

    /// Clear the grid
    #[wasm_bindgen]
    pub fn clear(&mut self) {
        self.occupancy.fill(0.0);
        self.intensity.fill(0.0);
    }

    /// Get grid dimensions
    #[wasm_bindgen]
    pub fn width(&self) -> usize {
        self.config.width
    }

    #[wasm_bindgen]
    pub fn height(&self) -> usize {
        self.config.height
    }
}

/// Flow field computation for motion estimation
#[wasm_bindgen]
pub struct FlowEstimator {
    prev_occupancy: Vec<f32>,
    width: usize,
    height: usize,
    flow_x: Vec<f32>,
    flow_y: Vec<f32>,
}

#[wasm_bindgen]
impl FlowEstimator {
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize) -> Self {
        let size = width * height;
        Self {
            prev_occupancy: vec![0.0; size],
            width,
            height,
            flow_x: vec![0.0; size],
            flow_y: vec![0.0; size],
        }
    }

    /// Compute flow between current and previous occupancy grids
    pub fn compute_flow(&mut self, current: &Float32Array) -> Float32Array {
        let size = self.width * self.height;
        let search_radius = 3;

        // Simple block matching for flow estimation
        for y in search_radius..self.height - search_radius {
            for x in search_radius..self.width - search_radius {
                let idx = y * self.width + x;
                let curr_val = current.get_index(idx as u32);

                if curr_val < 0.1 {
                    self.flow_x[idx] = 0.0;
                    self.flow_y[idx] = 0.0;
                    continue;
                }

                // Find best match in previous frame
                let mut best_dx = 0i32;
                let mut best_dy = 0i32;
                let mut best_diff = f32::MAX;

                for dy in -(search_radius as i32)..=(search_radius as i32) {
                    for dx in -(search_radius as i32)..=(search_radius as i32) {
                        let prev_x = (x as i32 + dx) as usize;
                        let prev_y = (y as i32 + dy) as usize;
                        let prev_idx = prev_y * self.width + prev_x;

                        let diff = (curr_val - self.prev_occupancy[prev_idx]).abs();
                        if diff < best_diff {
                            best_diff = diff;
                            best_dx = dx;
                            best_dy = dy;
                        }
                    }
                }

                self.flow_x[idx] = best_dx as f32 * 0.05;  // Convert to meters
                self.flow_y[idx] = best_dy as f32 * 0.05;
            }
        }

        // Update previous frame
        for i in 0..size {
            self.prev_occupancy[i] = current.get_index(i as u32);
        }

        // Return combined flow [flow_x, flow_y]
        let result = Float32Array::new_with_length((size * 2) as u32);
        for i in 0..size {
            result.set_index(i as u32, self.flow_x[i]);
            result.set_index((size + i) as u32, self.flow_y[i]);
        }
        result
    }
}

/// Risk map computation
#[wasm_bindgen]
pub fn compute_risk_map(
    occupancy: &Float32Array,
    flow_x: &Float32Array,
    flow_y: &Float32Array,
    width: usize,
    height: usize,
) -> Float32Array {
    let size = width * height;
    let result = Float32Array::new_with_length(size as u32);

    // Front region is higher risk
    let front_start = height / 2;
    let center_x = width / 2;

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let occ = occupancy.get_index(idx as u32);

            if occ < 0.1 {
                result.set_index(idx as u32, 0.0);
                continue;
            }

            // Base risk from occupancy
            let mut risk = occ;

            // Higher risk for objects in front
            if y > front_start {
                let front_factor = (y - front_start) as f32 / (height - front_start) as f32;
                risk *= 1.0 + front_factor;
            }

            // Higher risk for objects in path (center)
            let center_dist = ((x as i32 - center_x as i32).abs() as f32) / (width as f32 / 2.0);
            risk *= 1.0 + (1.0 - center_dist) * 0.5;

            // Higher risk for moving objects
            let flow_mag = (flow_x.get_index(idx as u32).powi(2)
                         + flow_y.get_index(idx as u32).powi(2)).sqrt();
            risk *= 1.0 + flow_mag * 2.0;

            result.set_index(idx as u32, risk.min(1.0));
        }
    }

    result
}

/// Simple path planning using cost map
#[wasm_bindgen]
pub fn plan_trajectory(
    risk_map: &Float32Array,
    width: usize,
    height: usize,
    num_waypoints: usize,
) -> Float32Array {
    let result = Float32Array::new_with_length((num_waypoints * 2) as u32);

    let center_x = width / 2;
    let start_y = 10;  // Start from bottom
    let end_y = height - 10;

    let mut current_x = center_x as f32;
    let step_y = (end_y - start_y) as f32 / num_waypoints as f32;

    for i in 0..num_waypoints {
        let y = (start_y as f32 + step_y * i as f32) as usize;

        // Find lowest risk path at this y level
        let mut best_x = current_x as usize;
        let mut best_risk = f32::MAX;

        let search_range = 20;
        let min_x = (current_x as i32 - search_range).max(0) as usize;
        let max_x = (current_x as i32 + search_range).min(width as i32 - 1) as usize;

        for x in min_x..=max_x {
            let idx = y * width + x;
            let risk = risk_map.get_index(idx as u32);

            // Prefer staying near current path
            let deviation_cost = ((x as f32 - current_x).abs() / search_range as f32) * 0.3;
            let total_cost = risk + deviation_cost;

            if total_cost < best_risk {
                best_risk = total_cost;
                best_x = x;
            }
        }

        // Convert to world coordinates (meters)
        let world_x = (best_x as f32 - center_x as f32) * 0.05;  // 5cm resolution
        let world_y = y as f32 * 0.05;

        result.set_index((i * 2) as u32, world_x);
        result.set_index((i * 2 + 1) as u32, world_y);

        current_x = best_x as f32;
    }

    result
}

// Helper: HSV to RGB conversion
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match h_prime as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processor_creation() {
        let processor = LiDARProcessor::new();
        assert_eq!(processor.width(), 240);
        assert_eq!(processor.height(), 200);
    }
}

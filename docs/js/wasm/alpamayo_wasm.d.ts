/* tslint:disable */
/* eslint-disable */

export class BEVConfig {
  free(): void;
  [Symbol.dispose](): void;
  constructor();
  set_grid_size(width: number, height: number): void;
  set_range(x_min: number, x_max: number, y_min: number, y_max: number): void;
}

export class FlowEstimator {
  free(): void;
  [Symbol.dispose](): void;
  constructor(width: number, height: number);
  /**
   * Compute flow between current and previous occupancy grids
   */
  compute_flow(current: Float32Array): Float32Array;
}

export class LiDARProcessor {
  free(): void;
  [Symbol.dispose](): void;
  constructor();
  static with_config(config: BEVConfig): LiDARProcessor;
  set_decay(decay: number): void;
  /**
   * Process LaserScan data
   * ranges: array of distances
   * intensities: array of intensity values (optional, can be empty)
   * angle_min, angle_increment: scan parameters
   */
  process_scan(ranges: Float32Array, intensities: Float32Array, angle_min: number, angle_increment: number): number;
  /**
   * Get occupancy grid as Float32Array
   */
  get_occupancy(): Float32Array;
  /**
   * Get intensity grid as Float32Array
   */
  get_intensity(): Float32Array;
  /**
   * Get combined 2-channel BEV tensor [occupancy, intensity]
   */
  get_bev_tensor(): Float32Array;
  /**
   * Render BEV to RGBA image data (for canvas)
   */
  render_to_rgba(color_mode: number): Uint8ClampedArray;
  /**
   * Count occupied cells
   */
  count_occupied(): number;
  /**
   * Clear the grid
   */
  clear(): void;
  /**
   * Get grid dimensions
   */
  width(): number;
  height(): number;
}

/**
 * Risk map computation
 */
export function compute_risk_map(occupancy: Float32Array, flow_x: Float32Array, flow_y: Float32Array, width: number, height: number): Float32Array;

export function init(): void;

/**
 * Simple path planning using cost map
 */
export function plan_trajectory(risk_map: Float32Array, width: number, height: number, num_waypoints: number): Float32Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_bevconfig_free: (a: number, b: number) => void;
  readonly bevconfig_new: () => number;
  readonly bevconfig_set_grid_size: (a: number, b: number, c: number) => void;
  readonly bevconfig_set_range: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly __wbg_lidarprocessor_free: (a: number, b: number) => void;
  readonly lidarprocessor_new: () => number;
  readonly lidarprocessor_with_config: (a: number) => number;
  readonly lidarprocessor_set_decay: (a: number, b: number) => void;
  readonly lidarprocessor_process_scan: (a: number, b: any, c: any, d: number, e: number) => number;
  readonly lidarprocessor_get_occupancy: (a: number) => any;
  readonly lidarprocessor_get_intensity: (a: number) => any;
  readonly lidarprocessor_get_bev_tensor: (a: number) => any;
  readonly lidarprocessor_render_to_rgba: (a: number, b: number) => any;
  readonly lidarprocessor_count_occupied: (a: number) => number;
  readonly lidarprocessor_clear: (a: number) => void;
  readonly lidarprocessor_width: (a: number) => number;
  readonly lidarprocessor_height: (a: number) => number;
  readonly __wbg_flowestimator_free: (a: number, b: number) => void;
  readonly flowestimator_new: (a: number, b: number) => number;
  readonly flowestimator_compute_flow: (a: number, b: any) => any;
  readonly compute_risk_map: (a: any, b: any, c: any, d: number, e: number) => any;
  readonly plan_trajectory: (a: any, b: number, c: number, d: number) => any;
  readonly init: () => void;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_externrefs: WebAssembly.Table;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;

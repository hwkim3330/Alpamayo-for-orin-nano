let wasm;

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    }
}

let WASM_VECTOR_LEN = 0;

const BEVConfigFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_bevconfig_free(ptr >>> 0, 1));

const BEVPipelineFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_bevpipeline_free(ptr >>> 0, 1));

const BEVResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_bevresult_free(ptr >>> 0, 1));

const FlowEstimatorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_flowestimator_free(ptr >>> 0, 1));

const LiDARProcessorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_lidarprocessor_free(ptr >>> 0, 1));

/**
 * BEV Grid configuration
 */
export class BEVConfig {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BEVConfigFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_bevconfig_free(ptr, 0);
    }
    constructor() {
        const ret = wasm.bevconfig_new();
        this.__wbg_ptr = ret >>> 0;
        BEVConfigFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} width
     * @param {number} height
     */
    set_grid_size(width, height) {
        wasm.bevconfig_set_grid_size(this.__wbg_ptr, width, height);
    }
    /**
     * @param {number} x_min
     * @param {number} x_max
     * @param {number} y_min
     * @param {number} y_max
     */
    set_range(x_min, x_max, y_min, y_max) {
        wasm.bevconfig_set_range(this.__wbg_ptr, x_min, x_max, y_min, y_max);
    }
}
if (Symbol.dispose) BEVConfig.prototype[Symbol.dispose] = BEVConfig.prototype.free;

/**
 * Full BEV Pipeline - Combines all processing stages
 */
export class BEVPipeline {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BEVPipelineFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_bevpipeline_free(ptr, 0);
    }
    constructor() {
        const ret = wasm.bevpipeline_new();
        this.__wbg_ptr = ret >>> 0;
        BEVPipelineFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Process complete pipeline: scan -> occupancy -> flow -> risk -> trajectory
     * @param {Float32Array} ranges
     * @param {Float32Array} intensities
     * @param {number} angle_min
     * @param {number} angle_increment
     * @returns {BEVResult}
     */
    process_frame(ranges, intensities, angle_min, angle_increment) {
        const ret = wasm.bevpipeline_process_frame(this.__wbg_ptr, ranges, intensities, angle_min, angle_increment);
        return BEVResult.__wrap(ret);
    }
    /**
     * Render complete visualization with all layers
     * @param {boolean} show_flow
     * @param {boolean} show_risk
     * @param {boolean} show_trajectory
     * @returns {Uint8ClampedArray}
     */
    render_full(show_flow, show_risk, show_trajectory) {
        const ret = wasm.bevpipeline_render_full(this.__wbg_ptr, show_flow, show_risk, show_trajectory);
        return ret;
    }
    /**
     * Get trajectory as Float32Array
     * @returns {Float32Array}
     */
    get_trajectory() {
        const ret = wasm.bevpipeline_get_trajectory(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get risk map as Float32Array
     * @returns {Float32Array}
     */
    get_risk_map() {
        const ret = wasm.bevpipeline_get_risk_map(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get flow as Float32Array [flow_x, flow_y]
     * @returns {Float32Array}
     */
    get_flow() {
        const ret = wasm.bevpipeline_get_flow(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    width() {
        const ret = wasm.bevpipeline_width(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    height() {
        const ret = wasm.bevpipeline_height(this.__wbg_ptr);
        return ret >>> 0;
    }
    clear() {
        wasm.bevpipeline_clear(this.__wbg_ptr);
    }
}
if (Symbol.dispose) BEVPipeline.prototype[Symbol.dispose] = BEVPipeline.prototype.free;

/**
 * Result from BEV processing
 */
export class BEVResult {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(BEVResult.prototype);
        obj.__wbg_ptr = ptr;
        BEVResultFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BEVResultFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_bevresult_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get point_count() {
        const ret = wasm.__wbg_get_bevresult_point_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set point_count(arg0) {
        wasm.__wbg_set_bevresult_point_count(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get occupied_cells() {
        const ret = wasm.__wbg_get_bevresult_occupied_cells(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set occupied_cells(arg0) {
        wasm.__wbg_set_bevresult_occupied_cells(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get max_risk() {
        const ret = wasm.__wbg_get_bevresult_max_risk(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set max_risk(arg0) {
        wasm.__wbg_set_bevresult_max_risk(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get frame_id() {
        const ret = wasm.__wbg_get_bevresult_frame_id(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} arg0
     */
    set frame_id(arg0) {
        wasm.__wbg_set_bevresult_frame_id(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) BEVResult.prototype[Symbol.dispose] = BEVResult.prototype.free;

/**
 * Flow field computation for motion estimation
 */
export class FlowEstimator {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        FlowEstimatorFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_flowestimator_free(ptr, 0);
    }
    /**
     * @param {number} width
     * @param {number} height
     */
    constructor(width, height) {
        const ret = wasm.flowestimator_new(width, height);
        this.__wbg_ptr = ret >>> 0;
        FlowEstimatorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Compute flow between current and previous occupancy grids
     * @param {Float32Array} current
     * @returns {Float32Array}
     */
    compute_flow(current) {
        const ret = wasm.flowestimator_compute_flow(this.__wbg_ptr, current);
        return ret;
    }
}
if (Symbol.dispose) FlowEstimator.prototype[Symbol.dispose] = FlowEstimator.prototype.free;

/**
 * LiDAR BEV Processor - High-performance point cloud to BEV grid conversion
 */
export class LiDARProcessor {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(LiDARProcessor.prototype);
        obj.__wbg_ptr = ptr;
        LiDARProcessorFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        LiDARProcessorFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_lidarprocessor_free(ptr, 0);
    }
    constructor() {
        const ret = wasm.lidarprocessor_new();
        this.__wbg_ptr = ret >>> 0;
        LiDARProcessorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {BEVConfig} config
     * @returns {LiDARProcessor}
     */
    static with_config(config) {
        _assertClass(config, BEVConfig);
        var ptr0 = config.__destroy_into_raw();
        const ret = wasm.lidarprocessor_with_config(ptr0);
        return LiDARProcessor.__wrap(ret);
    }
    /**
     * @param {number} decay
     */
    set_decay(decay) {
        wasm.lidarprocessor_set_decay(this.__wbg_ptr, decay);
    }
    /**
     * Process LaserScan data
     * ranges: array of distances
     * intensities: array of intensity values (optional, can be empty)
     * angle_min, angle_increment: scan parameters
     * @param {Float32Array} ranges
     * @param {Float32Array} intensities
     * @param {number} angle_min
     * @param {number} angle_increment
     * @returns {number}
     */
    process_scan(ranges, intensities, angle_min, angle_increment) {
        const ret = wasm.lidarprocessor_process_scan(this.__wbg_ptr, ranges, intensities, angle_min, angle_increment);
        return ret >>> 0;
    }
    /**
     * Get occupancy grid as Float32Array
     * @returns {Float32Array}
     */
    get_occupancy() {
        const ret = wasm.lidarprocessor_get_occupancy(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get intensity grid as Float32Array
     * @returns {Float32Array}
     */
    get_intensity() {
        const ret = wasm.lidarprocessor_get_intensity(this.__wbg_ptr);
        return ret;
    }
    /**
     * Get combined 2-channel BEV tensor [occupancy, intensity]
     * @returns {Float32Array}
     */
    get_bev_tensor() {
        const ret = wasm.lidarprocessor_get_bev_tensor(this.__wbg_ptr);
        return ret;
    }
    /**
     * Render BEV to RGBA image data (for canvas)
     * @param {number} color_mode
     * @returns {Uint8ClampedArray}
     */
    render_to_rgba(color_mode) {
        const ret = wasm.lidarprocessor_render_to_rgba(this.__wbg_ptr, color_mode);
        return ret;
    }
    /**
     * Count occupied cells
     * @returns {number}
     */
    count_occupied() {
        const ret = wasm.lidarprocessor_count_occupied(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Clear the grid
     */
    clear() {
        wasm.lidarprocessor_clear(this.__wbg_ptr);
    }
    /**
     * Get grid dimensions
     * @returns {number}
     */
    width() {
        const ret = wasm.bevpipeline_width(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    height() {
        const ret = wasm.bevpipeline_height(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) LiDARProcessor.prototype[Symbol.dispose] = LiDARProcessor.prototype.free;

/**
 * Risk map computation
 * @param {Float32Array} occupancy
 * @param {Float32Array} flow_x
 * @param {Float32Array} flow_y
 * @param {number} width
 * @param {number} height
 * @returns {Float32Array}
 */
export function compute_risk_map(occupancy, flow_x, flow_y, width, height) {
    const ret = wasm.compute_risk_map(occupancy, flow_x, flow_y, width, height);
    return ret;
}

export function init() {
    wasm.init();
}

/**
 * Simple path planning using cost map
 * @param {Float32Array} risk_map
 * @param {number} width
 * @param {number} height
 * @param {number} num_waypoints
 * @returns {Float32Array}
 */
export function plan_trajectory(risk_map, width, height, num_waypoints) {
    const ret = wasm.plan_trajectory(risk_map, width, height, num_waypoints);
    return ret;
}

const EXPECTED_RESPONSE_TYPES = new Set(['basic', 'cors', 'default']);

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && EXPECTED_RESPONSE_TYPES.has(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbg___wbindgen_throw_dd24417ed36fc46e = function(arg0, arg1) {
        throw new Error(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbg_error_7534b8e9a36f1ab4 = function(arg0, arg1) {
        let deferred0_0;
        let deferred0_1;
        try {
            deferred0_0 = arg0;
            deferred0_1 = arg1;
            console.error(getStringFromWasm0(arg0, arg1));
        } finally {
            wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
        }
    };
    imports.wbg.__wbg_get_index_865ef6c029b35e97 = function(arg0, arg1) {
        const ret = arg0[arg1 >>> 0];
        return ret;
    };
    imports.wbg.__wbg_length_86ce4877baf913bb = function(arg0) {
        const ret = arg0.length;
        return ret;
    };
    imports.wbg.__wbg_new_8a6f238a6ece86ea = function() {
        const ret = new Error();
        return ret;
    };
    imports.wbg.__wbg_new_with_length_30843b434774b4c6 = function(arg0) {
        const ret = new Uint8ClampedArray(arg0 >>> 0);
        return ret;
    };
    imports.wbg.__wbg_new_with_length_95ba657dfb7d3dfb = function(arg0) {
        const ret = new Float32Array(arg0 >>> 0);
        return ret;
    };
    imports.wbg.__wbg_set_index_165b46b0114d368c = function(arg0, arg1, arg2) {
        arg0[arg1 >>> 0] = arg2;
    };
    imports.wbg.__wbg_set_index_692c683816d95946 = function(arg0, arg1, arg2) {
        arg0[arg1 >>> 0] = arg2;
    };
    imports.wbg.__wbg_stack_0ed75d68575b0f3c = function(arg0, arg1) {
        const ret = arg1.stack;
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbindgen_init_externref_table = function() {
        const table = wasm.__wbindgen_externrefs;
        const offset = table.grow(4);
        table.set(0, undefined);
        table.set(offset + 0, undefined);
        table.set(offset + 1, null);
        table.set(offset + 2, true);
        table.set(offset + 3, false);
    };

    return imports;
}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedDataViewMemory0 = null;
    cachedUint8ArrayMemory0 = null;


    wasm.__wbindgen_start();
    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (typeof module !== 'undefined') {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (typeof module_or_path !== 'undefined') {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (typeof module_or_path === 'undefined') {
        module_or_path = new URL('alpamayo_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;

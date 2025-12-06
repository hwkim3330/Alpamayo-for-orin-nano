/**
 * WASM-accelerated LiDAR BEV Processor
 * Provides ~10x speedup over pure JavaScript for point cloud processing
 */

class WASMLiDARProcessor {
    constructor() {
        this.wasmModule = null;
        this.processor = null;
        this.flowEstimator = null;
        this.isReady = false;
        this.useWasm = false;  // Fallback to JS if WASM fails

        // Performance tracking
        this.processingTimes = [];
        this.lastBenchmark = { wasm: 0, js: 0 };
    }

    /**
     * Initialize WASM module
     * @returns {Promise<boolean>} True if WASM loaded successfully
     */
    async init() {
        try {
            // Dynamic import of WASM module
            const wasmPath = this.getWasmPath();
            console.log(`[WASM] Loading from: ${wasmPath}`);

            this.wasmModule = await import(wasmPath);
            await this.wasmModule.default();  // Initialize WASM

            // Create processor instances
            this.processor = new this.wasmModule.LiDARProcessor();
            this.flowEstimator = new this.wasmModule.FlowEstimator(
                this.processor.width(),
                this.processor.height()
            );

            this.isReady = true;
            this.useWasm = true;

            console.log(`[WASM] Initialized successfully - Grid: ${this.processor.width()}x${this.processor.height()}`);
            return true;
        } catch (error) {
            console.warn('[WASM] Failed to load, falling back to JavaScript:', error);
            this.useWasm = false;
            return false;
        }
    }

    /**
     * Get WASM module path based on current location
     */
    getWasmPath() {
        // Detect if running locally or on GitHub Pages
        const currentPath = window.location.pathname;
        if (currentPath.includes('/fsd/')) {
            return '../js/wasm/alpamayo_wasm.js';
        }
        return './js/wasm/alpamayo_wasm.js';
    }

    /**
     * Process LaserScan data using WASM
     * @param {Object} scan - ROS LaserScan message
     * @returns {Object} Processing result with occupancy, timing, etc.
     */
    processScan(scan) {
        if (!this.useWasm || !this.processor) {
            return this.processScanJS(scan);
        }

        const startTime = performance.now();

        try {
            // Convert ranges to Float32Array
            const ranges = new Float32Array(scan.ranges);
            const intensities = scan.intensities
                ? new Float32Array(scan.intensities)
                : new Float32Array(0);

            // Process with WASM
            const pointCount = this.processor.process_scan(
                ranges,
                intensities,
                scan.angle_min,
                scan.angle_increment
            );

            const processingTime = performance.now() - startTime;
            this.trackPerformance(processingTime, true);

            return {
                pointCount,
                occupiedCells: this.processor.count_occupied(),
                processingTime,
                useWasm: true,
                width: this.processor.width(),
                height: this.processor.height()
            };
        } catch (error) {
            console.error('[WASM] Processing error:', error);
            return this.processScanJS(scan);
        }
    }

    /**
     * Fallback JavaScript processing
     */
    processScanJS(scan) {
        const startTime = performance.now();

        // Simple JS implementation (for fallback)
        const width = 240, height = 200;
        const resolution = 0.05;
        const xMin = -2.0, xMax = 10.0;
        const yMin = -5.0, yMax = 5.0;

        let pointCount = 0;
        let occupiedCells = 0;

        for (let i = 0; i < scan.ranges.length; i++) {
            const range = scan.ranges[i];
            if (range < 0.1 || range > 12.0 || !isFinite(range)) continue;

            const angle = scan.angle_min + i * scan.angle_increment;
            const x = range * Math.cos(angle);
            const y = range * Math.sin(angle);

            if (x >= xMin && x <= xMax && y >= yMin && y <= yMax) {
                pointCount++;
            }
        }

        const processingTime = performance.now() - startTime;
        this.trackPerformance(processingTime, false);

        return {
            pointCount,
            occupiedCells: Math.floor(pointCount * 0.7),  // Estimate
            processingTime,
            useWasm: false,
            width, height
        };
    }

    /**
     * Get occupancy grid as Float32Array
     */
    getOccupancy() {
        if (!this.useWasm || !this.processor) return null;
        return this.processor.get_occupancy();
    }

    /**
     * Get intensity grid as Float32Array
     */
    getIntensity() {
        if (!this.useWasm || !this.processor) return null;
        return this.processor.get_intensity();
    }

    /**
     * Get combined BEV tensor [occupancy, intensity]
     */
    getBEVTensor() {
        if (!this.useWasm || !this.processor) return null;
        return this.processor.get_bev_tensor();
    }

    /**
     * Render BEV to RGBA image data for canvas
     * @param {number} colorMode - 0: distance, 1: intensity, 2: green
     */
    renderToRGBA(colorMode = 0) {
        if (!this.useWasm || !this.processor) return null;
        return this.processor.render_to_rgba(colorMode);
    }

    /**
     * Compute flow between frames
     */
    computeFlow() {
        if (!this.useWasm || !this.processor || !this.flowEstimator) return null;

        const occupancy = this.getOccupancy();
        if (!occupancy) return null;

        return this.flowEstimator.compute_flow(occupancy);
    }

    /**
     * Compute risk map from occupancy and flow
     */
    computeRiskMap() {
        if (!this.useWasm || !this.wasmModule) return null;

        const occupancy = this.getOccupancy();
        const flow = this.computeFlow();

        if (!occupancy || !flow) return null;

        const width = this.processor.width();
        const height = this.processor.height();
        const size = width * height;

        // Extract flow_x and flow_y from combined flow
        const flowX = new Float32Array(size);
        const flowY = new Float32Array(size);
        for (let i = 0; i < size; i++) {
            flowX[i] = flow[i];
            flowY[i] = flow[size + i];
        }

        return this.wasmModule.compute_risk_map(
            occupancy, flowX, flowY, width, height
        );
    }

    /**
     * Plan trajectory through risk map
     * @param {number} numWaypoints - Number of waypoints to generate
     */
    planTrajectory(numWaypoints = 8) {
        if (!this.useWasm || !this.wasmModule) return null;

        const riskMap = this.computeRiskMap();
        if (!riskMap) return null;

        const width = this.processor.width();
        const height = this.processor.height();

        return this.wasmModule.plan_trajectory(
            riskMap, width, height, numWaypoints
        );
    }

    /**
     * Set decay factor for temporal persistence
     */
    setDecay(decay) {
        if (this.processor) {
            this.processor.set_decay(decay);
        }
    }

    /**
     * Clear all grids
     */
    clear() {
        if (this.processor) {
            this.processor.clear();
        }
    }

    /**
     * Track processing performance
     */
    trackPerformance(time, isWasm) {
        this.processingTimes.push({ time, isWasm, timestamp: Date.now() });

        // Keep last 100 samples
        if (this.processingTimes.length > 100) {
            this.processingTimes.shift();
        }

        // Update benchmark
        const wasmTimes = this.processingTimes.filter(t => t.isWasm).map(t => t.time);
        const jsTimes = this.processingTimes.filter(t => !t.isWasm).map(t => t.time);

        if (wasmTimes.length > 0) {
            this.lastBenchmark.wasm = wasmTimes.reduce((a, b) => a + b, 0) / wasmTimes.length;
        }
        if (jsTimes.length > 0) {
            this.lastBenchmark.js = jsTimes.reduce((a, b) => a + b, 0) / jsTimes.length;
        }
    }

    /**
     * Get performance statistics
     */
    getPerformanceStats() {
        const stats = {
            useWasm: this.useWasm,
            avgProcessingTime: 0,
            wasmAvg: this.lastBenchmark.wasm,
            jsAvg: this.lastBenchmark.js,
            speedup: 1
        };

        if (this.processingTimes.length > 0) {
            stats.avgProcessingTime = this.processingTimes.slice(-10)
                .reduce((a, b) => a + b.time, 0) / Math.min(10, this.processingTimes.length);
        }

        if (stats.jsAvg > 0 && stats.wasmAvg > 0) {
            stats.speedup = stats.jsAvg / stats.wasmAvg;
        }

        return stats;
    }

    /**
     * Get grid dimensions
     */
    getDimensions() {
        if (this.processor) {
            return {
                width: this.processor.width(),
                height: this.processor.height()
            };
        }
        return { width: 240, height: 200 };
    }
}

// Export for ES modules
export { WASMLiDARProcessor };

// Also expose globally for non-module usage
if (typeof window !== 'undefined') {
    window.WASMLiDARProcessor = WASMLiDARProcessor;
}

/**
 * LiDAR BEV Viewer - Canvas-based 2D LiDAR visualization
 * Renders LaserScan data as Bird's Eye View
 */

class LiDARViewer {
    constructor(canvasId, options = {}) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');

        // BEV grid settings (matching model)
        this.config = {
            gridWidth: options.gridWidth || 240,      // cells
            gridHeight: options.gridHeight || 200,    // cells
            resolution: options.resolution || 0.05,   // 5cm per cell
            xMin: options.xMin || -2.0,               // meters
            xMax: options.xMax || 10.0,
            yMin: options.yMin || -5.0,
            yMax: options.yMax || 5.0,
            maxRange: options.maxRange || 12.0,
            pointSize: options.pointSize || 2,
            colorScheme: options.colorScheme || 'intensity',  // 'intensity', 'distance', 'green'
            showGrid: options.showGrid !== false,
            showEgo: options.showEgo !== false,
            decay: options.decay || 0.95              // Trail decay factor
        };

        // State
        this.points = [];
        this.bevGrid = new Float32Array(this.config.gridWidth * this.config.gridHeight);
        this.intensityGrid = new Float32Array(this.config.gridWidth * this.config.gridHeight);

        // Resize canvas
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }

    resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;

        // Calculate scale to fit BEV grid
        const worldWidth = this.config.xMax - this.config.xMin;
        const worldHeight = this.config.yMax - this.config.yMin;

        this.scale = Math.min(
            this.canvas.width / worldWidth,
            this.canvas.height / worldHeight
        ) * 0.9;

        this.offsetX = this.canvas.width / 2;
        this.offsetY = this.canvas.height * 0.8;  // Ego at bottom
    }

    // Convert world coordinates to canvas
    worldToCanvas(x, y) {
        return {
            cx: this.offsetX + y * this.scale,  // Y is lateral
            cy: this.offsetY - x * this.scale   // X is forward
        };
    }

    // Convert LaserScan to points
    processLaserScan(scan) {
        this.points = [];

        const angleMin = scan.angle_min;
        const angleIncrement = scan.angle_increment;
        const ranges = scan.ranges;
        const intensities = scan.intensities || [];

        // Decay existing grid
        for (let i = 0; i < this.bevGrid.length; i++) {
            this.bevGrid[i] *= this.config.decay;
            this.intensityGrid[i] *= this.config.decay;
        }

        for (let i = 0; i < ranges.length; i++) {
            const range = ranges[i];

            // Skip invalid readings
            if (range < 0.1 || range > this.config.maxRange || !isFinite(range)) {
                continue;
            }

            const angle = angleMin + i * angleIncrement;
            const x = range * Math.cos(angle);
            const y = range * Math.sin(angle);
            const intensity = intensities[i] || 0.5;

            // Check bounds
            if (x < this.config.xMin || x > this.config.xMax ||
                y < this.config.yMin || y > this.config.yMax) {
                continue;
            }

            this.points.push({ x, y, range, intensity });

            // Update BEV grid
            const gx = Math.floor((x - this.config.xMin) / this.config.resolution);
            const gy = Math.floor((y - this.config.yMin) / this.config.resolution);

            if (gx >= 0 && gx < this.config.gridWidth &&
                gy >= 0 && gy < this.config.gridHeight) {
                const idx = gy * this.config.gridWidth + gx;
                this.bevGrid[idx] = 1.0;
                this.intensityGrid[idx] = intensity;
            }
        }

        return this.points;
    }

    // Get color based on scheme
    getPointColor(point) {
        switch (this.config.colorScheme) {
            case 'distance':
                const distRatio = point.range / this.config.maxRange;
                const hue = (1 - distRatio) * 120;  // Green to red
                return `hsl(${hue}, 100%, 50%)`;

            case 'intensity':
                const brightness = Math.min(255, Math.floor(point.intensity * 255));
                return `rgb(${brightness}, ${Math.floor(brightness * 0.8)}, 0)`;

            case 'green':
            default:
                return '#22c55e';
        }
    }

    // Render the visualization
    render() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        // Clear with dark background
        ctx.fillStyle = '#0a0a0f';
        ctx.fillRect(0, 0, w, h);

        // Draw grid
        if (this.config.showGrid) {
            this.drawGrid();
        }

        // Draw range circles
        this.drawRangeCircles();

        // Draw LiDAR points
        this.drawPoints();

        // Draw ego vehicle
        if (this.config.showEgo) {
            this.drawEgo();
        }

        // Draw info
        this.drawInfo();
    }

    drawGrid() {
        const ctx = this.ctx;
        ctx.strokeStyle = 'rgba(60, 60, 80, 0.3)';
        ctx.lineWidth = 1;

        // Vertical lines (every 1m)
        for (let y = this.config.yMin; y <= this.config.yMax; y += 1) {
            const p1 = this.worldToCanvas(this.config.xMin, y);
            const p2 = this.worldToCanvas(this.config.xMax, y);
            ctx.beginPath();
            ctx.moveTo(p1.cx, p1.cy);
            ctx.lineTo(p2.cx, p2.cy);
            ctx.stroke();
        }

        // Horizontal lines (every 1m)
        for (let x = this.config.xMin; x <= this.config.xMax; x += 1) {
            const p1 = this.worldToCanvas(x, this.config.yMin);
            const p2 = this.worldToCanvas(x, this.config.yMax);
            ctx.beginPath();
            ctx.moveTo(p1.cx, p1.cy);
            ctx.lineTo(p2.cx, p2.cy);
            ctx.stroke();
        }
    }

    drawRangeCircles() {
        const ctx = this.ctx;
        const ego = this.worldToCanvas(0, 0);

        ctx.strokeStyle = 'rgba(59, 130, 246, 0.2)';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);

        // Draw circles at 2m, 5m, 10m
        [2, 5, 10].forEach(range => {
            const radius = range * this.scale;
            ctx.beginPath();
            ctx.arc(ego.cx, ego.cy, radius, 0, Math.PI * 2);
            ctx.stroke();

            // Label
            ctx.fillStyle = 'rgba(136, 136, 136, 0.5)';
            ctx.font = '10px Inter';
            ctx.fillText(`${range}m`, ego.cx + radius + 5, ego.cy);
        });

        ctx.setLineDash([]);
    }

    drawPoints() {
        const ctx = this.ctx;

        this.points.forEach(point => {
            const { cx, cy } = this.worldToCanvas(point.x, point.y);

            ctx.fillStyle = this.getPointColor(point);
            ctx.beginPath();
            ctx.arc(cx, cy, this.config.pointSize, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    drawEgo() {
        const ctx = this.ctx;
        const ego = this.worldToCanvas(0, 0);

        // Car shape
        ctx.fillStyle = '#3b82f6';
        ctx.beginPath();
        ctx.moveTo(ego.cx, ego.cy - 15);
        ctx.lineTo(ego.cx - 8, ego.cy + 10);
        ctx.lineTo(ego.cx + 8, ego.cy + 10);
        ctx.closePath();
        ctx.fill();

        // Glow
        ctx.shadowColor = '#3b82f6';
        ctx.shadowBlur = 10;
        ctx.fill();
        ctx.shadowBlur = 0;
    }

    drawInfo() {
        const ctx = this.ctx;

        ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
        ctx.font = '11px JetBrains Mono';
        ctx.fillText(`Points: ${this.points.length}`, 10, 20);
        ctx.fillText(`Grid: ${this.config.gridWidth}x${this.config.gridHeight}`, 10, 35);
        ctx.fillText(`Res: ${this.config.resolution * 100}cm`, 10, 50);
    }

    // Get BEV grid for model input
    getBEVGrid() {
        return {
            occupancy: this.bevGrid,
            intensity: this.intensityGrid,
            width: this.config.gridWidth,
            height: this.config.gridHeight
        };
    }
}

/**
 * LiDAR3DViewer - Three.js based 3D point cloud visualization
 */
class LiDAR3DViewer {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);

        this.config = {
            maxPoints: options.maxPoints || 5000,
            pointSize: options.pointSize || 0.03,
            fov: options.fov || 60,
            cameraHeight: options.cameraHeight || 5,
            cameraDistance: options.cameraDistance || 10
        };

        this.init();
    }

    init() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0f);

        // Camera
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(this.config.fov, aspect, 0.1, 100);
        this.camera.position.set(0, this.config.cameraHeight, -this.config.cameraDistance);
        this.camera.lookAt(0, 0, 5);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.container.appendChild(this.renderer.domElement);

        // Points
        this.geometry = new THREE.BufferGeometry();
        this.positions = new Float32Array(this.config.maxPoints * 3);
        this.colors = new Float32Array(this.config.maxPoints * 3);

        this.geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
        this.geometry.setAttribute('color', new THREE.BufferAttribute(this.colors, 3));

        const material = new THREE.PointsMaterial({
            size: this.config.pointSize,
            vertexColors: true
        });

        this.points = new THREE.Points(this.geometry, material);
        this.scene.add(this.points);

        // Grid helper
        const gridHelper = new THREE.GridHelper(20, 20, 0x3a3a4a, 0x2a2a3a);
        gridHelper.rotation.x = Math.PI / 2;
        gridHelper.position.z = 5;
        this.scene.add(gridHelper);

        // Ego marker
        const egoGeometry = new THREE.ConeGeometry(0.3, 0.8, 4);
        const egoMaterial = new THREE.MeshBasicMaterial({ color: 0x3b82f6 });
        this.egoMarker = new THREE.Mesh(egoGeometry, egoMaterial);
        this.egoMarker.rotation.x = Math.PI / 2;
        this.scene.add(this.egoMarker);

        // Handle resize
        window.addEventListener('resize', () => this.onResize());

        // Animation
        this.animate();
    }

    onResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    updatePoints(laserScan) {
        const angleMin = laserScan.angle_min;
        const angleIncrement = laserScan.angle_increment;
        const ranges = laserScan.ranges;
        const intensities = laserScan.intensities || [];

        let pointIndex = 0;

        for (let i = 0; i < ranges.length && pointIndex < this.config.maxPoints; i++) {
            const range = ranges[i];

            if (range < 0.1 || range > 12 || !isFinite(range)) {
                continue;
            }

            const angle = angleMin + i * angleIncrement;
            const x = range * Math.sin(angle);  // Lateral
            const y = 0;                         // Height (2D LiDAR)
            const z = range * Math.cos(angle);  // Forward

            const idx = pointIndex * 3;
            this.positions[idx] = x;
            this.positions[idx + 1] = y;
            this.positions[idx + 2] = z;

            // Color based on distance
            const intensity = intensities[i] || 0.5;
            const hue = (1 - range / 12) * 0.3;  // Green to yellow
            const color = new THREE.Color().setHSL(hue, 1, 0.5 + intensity * 0.3);

            this.colors[idx] = color.r;
            this.colors[idx + 1] = color.g;
            this.colors[idx + 2] = color.b;

            pointIndex++;
        }

        // Clear remaining points
        for (let i = pointIndex; i < this.config.maxPoints; i++) {
            const idx = i * 3;
            this.positions[idx] = 0;
            this.positions[idx + 1] = -1000;  // Hide
            this.positions[idx + 2] = 0;
        }

        this.geometry.attributes.position.needsUpdate = true;
        this.geometry.attributes.color.needsUpdate = true;
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        // Slowly rotate camera
        // this.camera.position.x = Math.sin(Date.now() * 0.0001) * 3;
        // this.camera.lookAt(0, 0, 5);

        this.renderer.render(this.scene, this.camera);
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { LiDARViewer, LiDAR3DViewer };
}

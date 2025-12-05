/**
 * ROS2 Web Dashboard - JavaScript
 * Uses roslibjs and ros3djs for ROS2 integration
 */

class ROSDashboard {
    constructor() {
        // Configuration
        this.config = {
            rosbridgeUrl: 'ws://localhost:9090',
            videoServerUrl: 'http://localhost:8080',
            cameraTopic: '/camera/image_web',
            bevTopic: '/lidar/bev_image',
            autoReconnect: true,
            reconnectDelay: 3000
        };

        // State
        this.ros = null;
        this.connected = false;
        this.subscribers = {};
        this.viewer3d = null;
        this.viewerMini = null;

        // DOM Elements
        this.elements = {
            connectionStatus: document.getElementById('connection-status'),
            systemTime: document.getElementById('system-time'),
            speedValue: document.getElementById('speed-value'),
            steeringValue: document.getElementById('steering-value'),
            batteryValue: document.getElementById('battery-value'),
            powerValue: document.getElementById('power-value'),
            tempValue: document.getElementById('temp-value'),
            cpuBar: document.getElementById('cpu-bar'),
            cpuValue: document.getElementById('cpu-value'),
            gpuBar: document.getElementById('gpu-bar'),
            gpuValue: document.getElementById('gpu-value'),
            ramBar: document.getElementById('ram-bar'),
            ramValue: document.getElementById('ram-value'),
            detectionList: document.getElementById('detection-list'),
            detectionCount: document.getElementById('detection-count'),
            console: document.getElementById('console'),
            confidenceFill: document.getElementById('confidence-fill'),
            confidenceValue: document.getElementById('confidence-value'),
            driveMode: document.getElementById('drive-mode'),
            topicCount: document.getElementById('topic-count'),
            latencyValue: document.getElementById('latency-value'),
            renderFps: document.getElementById('render-fps'),
            bevImage: document.getElementById('bev-image'),
            bevImageMini: document.getElementById('bev-image-mini'),
            cameraImage: document.getElementById('camera-image'),
            cameraImageMini: document.getElementById('camera-image-mini')
        };

        // Initialize
        this.init();
    }

    init() {
        this.loadConfig();
        this.setupUI();
        this.startTimeUpdate();
        this.connect();

        // Initialize Lucide icons
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
    }

    loadConfig() {
        const saved = localStorage.getItem('ros_dashboard_config');
        if (saved) {
            Object.assign(this.config, JSON.parse(saved));
        }

        // Update settings inputs
        document.getElementById('rosbridge-url').value = this.config.rosbridgeUrl;
        document.getElementById('video-url').value = this.config.videoServerUrl;
        document.getElementById('camera-topic').value = this.config.cameraTopic;
        document.getElementById('bev-topic').value = this.config.bevTopic;
        document.getElementById('auto-reconnect').checked = this.config.autoReconnect;
    }

    saveConfig() {
        this.config.rosbridgeUrl = document.getElementById('rosbridge-url').value;
        this.config.videoServerUrl = document.getElementById('video-url').value;
        this.config.cameraTopic = document.getElementById('camera-topic').value;
        this.config.bevTopic = document.getElementById('bev-topic').value;
        this.config.autoReconnect = document.getElementById('auto-reconnect').checked;

        localStorage.setItem('ros_dashboard_config', JSON.stringify(this.config));
    }

    setupUI() {
        // View tabs
        document.querySelectorAll('.viz-tab').forEach(tab => {
            tab.addEventListener('click', () => this.switchView(tab.dataset.view));
        });

        // Settings modal
        document.getElementById('btn-settings').addEventListener('click', () => {
            document.getElementById('settings-modal').classList.add('active');
        });

        document.getElementById('btn-close-settings').addEventListener('click', () => {
            document.getElementById('settings-modal').classList.remove('active');
        });

        document.getElementById('btn-cancel-settings').addEventListener('click', () => {
            document.getElementById('settings-modal').classList.remove('active');
        });

        document.getElementById('btn-save-settings').addEventListener('click', () => {
            this.saveConfig();
            document.getElementById('settings-modal').classList.remove('active');
            this.reconnect();
        });

        // Gear indicators
        document.querySelectorAll('.gear').forEach(gear => {
            gear.addEventListener('click', () => {
                document.querySelectorAll('.gear').forEach(g => g.classList.remove('active'));
                gear.classList.add('active');
            });
        });
    }

    switchView(viewId) {
        // Update tabs
        document.querySelectorAll('.viz-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.view === viewId);
        });

        // Update views
        document.querySelectorAll('.viz-view').forEach(view => {
            view.classList.toggle('active', view.id === `view-${viewId}`);
        });

        // Initialize 3D viewer if needed
        if (viewId === '3d' && !this.viewer3d) {
            this.init3DViewer('ros3d-viewer');
        } else if (viewId === 'split' && !this.viewerMini) {
            setTimeout(() => {
                this.init3DViewer('ros3d-viewer-mini', true);
            }, 100);
        }
    }

    startTimeUpdate() {
        setInterval(() => {
            const now = new Date();
            this.elements.systemTime.textContent = now.toLocaleTimeString('en-US', { hour12: false });
        }, 1000);
    }

    connect() {
        this.log('Connecting to ' + this.config.rosbridgeUrl, 'info');

        this.ros = new ROSLIB.Ros({
            url: this.config.rosbridgeUrl
        });

        this.ros.on('connection', () => {
            this.connected = true;
            this.updateConnectionStatus(true);
            this.log('Connected to ROS bridge', 'info');
            this.setupSubscribers();
            this.setupVideoStreams();
            this.init3DViewer('ros3d-viewer');
        });

        this.ros.on('error', (error) => {
            this.log('Connection error: ' + error, 'error');
        });

        this.ros.on('close', () => {
            this.connected = false;
            this.updateConnectionStatus(false);
            this.log('Connection closed', 'warn');

            if (this.config.autoReconnect) {
                setTimeout(() => this.connect(), this.config.reconnectDelay);
            }
        });
    }

    reconnect() {
        if (this.ros) {
            this.ros.close();
        }
        setTimeout(() => this.connect(), 500);
    }

    updateConnectionStatus(connected) {
        const el = this.elements.connectionStatus;
        if (connected) {
            el.classList.add('connected');
            el.classList.remove('disconnected');
            el.innerHTML = '<i data-lucide="wifi"></i><span>Connected</span>';
        } else {
            el.classList.remove('connected');
            el.classList.add('disconnected');
            el.innerHTML = '<i data-lucide="wifi-off"></i><span>Disconnected</span>';
        }
        lucide.createIcons();
    }

    setupSubscribers() {
        // Vehicle state
        this.subscribe('/vehicle/state_json', 'std_msgs/msg/String', (msg) => {
            try {
                const state = JSON.parse(msg.data);
                this.updateVehicleState(state);
            } catch (e) {
                console.error('Parse error:', e);
            }
        });

        // Detections
        this.subscribe('/planner/detections_json', 'std_msgs/msg/String', (msg) => {
            try {
                const detections = JSON.parse(msg.data);
                this.updateDetections(detections);
            } catch (e) {
                console.error('Parse error:', e);
            }
        });

        // Speed
        this.subscribe('/vehicle/speed', 'std_msgs/msg/Float32', (msg) => {
            this.elements.speedValue.textContent = Math.round(msg.data);
        });

        // Steering
        this.subscribe('/vehicle/steering', 'std_msgs/msg/Float32', (msg) => {
            this.elements.steeringValue.textContent = msg.data.toFixed(1) + '°';
        });

        // Get topic count
        this.ros.getTopics((result) => {
            this.elements.topicCount.textContent = result.topics.length;
        });
    }

    subscribe(topic, messageType, callback) {
        if (this.subscribers[topic]) {
            this.subscribers[topic].unsubscribe();
        }

        const subscriber = new ROSLIB.Topic({
            ros: this.ros,
            name: topic,
            messageType: messageType
        });

        subscriber.subscribe(callback);
        this.subscribers[topic] = subscriber;

        this.log(`Subscribed to ${topic}`, 'info');
    }

    setupVideoStreams() {
        // Camera stream via web_video_server
        const cameraUrl = `${this.config.videoServerUrl}/stream?topic=${this.config.cameraTopic}&type=mjpeg&quality=70`;
        this.elements.cameraImage.src = cameraUrl;
        if (this.elements.cameraImageMini) {
            this.elements.cameraImageMini.src = cameraUrl;
        }

        // BEV stream
        const bevUrl = `${this.config.videoServerUrl}/stream?topic=${this.config.bevTopic}&type=mjpeg&quality=80`;
        this.elements.bevImage.src = bevUrl;
        if (this.elements.bevImageMini) {
            this.elements.bevImageMini.src = bevUrl;
        }

        this.log('Video streams configured', 'info');
    }

    init3DViewer(containerId, isMini = false) {
        const container = document.getElementById(containerId);
        if (!container || !this.connected) return;

        const viewer = new ROS3D.Viewer({
            divID: containerId,
            width: container.clientWidth,
            height: container.clientHeight,
            antialias: true,
            background: '#0a0a0f'
        });

        // Set camera position
        viewer.camera.position.set(5, 5, 5);
        viewer.camera.lookAt(0, 0, 0);

        // Add grid
        viewer.addObject(new ROS3D.Grid({
            ros: this.ros,
            color: '#333344',
            cellSize: 1.0,
            num_cells: 20
        }));

        // Add TF client
        const tfClient = new ROSLIB.TFClient({
            ros: this.ros,
            fixedFrame: 'base_link',
            angularThres: 0.01,
            transThres: 0.01
        });

        // Add path visualization
        new ROS3D.Path({
            ros: this.ros,
            tfClient: tfClient,
            rootObject: viewer.scene,
            topic: '/planner/path',
            color: 0x22c55e
        });

        // Add marker array (detections)
        new ROS3D.MarkerArrayClient({
            ros: this.ros,
            tfClient: tfClient,
            rootObject: viewer.scene,
            topic: '/planner/markers'
        });

        // Add axes helper
        const axesHelper = new THREE.AxesHelper(1);
        viewer.scene.add(axesHelper);

        // Add ego vehicle representation
        const egoGeometry = new THREE.BoxGeometry(0.4, 0.2, 0.15);
        const egoMaterial = new THREE.MeshBasicMaterial({ color: 0x06b6d4, wireframe: true });
        const egoMesh = new THREE.Mesh(egoGeometry, egoMaterial);
        egoMesh.position.set(0, 0, 0.075);
        viewer.scene.add(egoMesh);

        if (isMini) {
            this.viewerMini = viewer;
        } else {
            this.viewer3d = viewer;
        }

        // Handle resize
        window.addEventListener('resize', () => {
            viewer.resize(container.clientWidth, container.clientHeight);
        });

        this.log('3D viewer initialized', 'info');

        // Update FPS counter
        let frameCount = 0;
        let lastTime = performance.now();
        const updateFps = () => {
            frameCount++;
            const now = performance.now();
            if (now - lastTime >= 1000) {
                this.elements.renderFps.textContent = frameCount;
                frameCount = 0;
                lastTime = now;
            }
            requestAnimationFrame(updateFps);
        };
        updateFps();
    }

    updateVehicleState(state) {
        // Speed
        this.elements.speedValue.textContent = Math.round(state.speed_kmh || 0);

        // Steering
        this.elements.steeringValue.textContent = (state.steering_angle || 0).toFixed(1) + '°';

        // Battery
        this.elements.batteryValue.textContent = Math.round(state.battery_percent || 100) + '%';

        // Power
        const power = state.power || 0;
        this.elements.powerValue.textContent = (power >= 0 ? '+' : '') + Math.round(power) + 'W';

        // Gear
        const gear = state.gear || 'N';
        document.querySelectorAll('.gear').forEach(g => {
            g.classList.toggle('active', g.dataset.gear === gear);
        });

        // Mode
        this.elements.driveMode.textContent = state.mode || 'Manual';

        // Simulated system stats
        const cpu = 30 + Math.random() * 40;
        const gpu = 40 + Math.random() * 50;
        const ram = 50 + Math.random() * 30;

        this.elements.cpuBar.style.width = cpu + '%';
        this.elements.cpuValue.textContent = Math.round(cpu) + '%';
        this.elements.gpuBar.style.width = gpu + '%';
        this.elements.gpuValue.textContent = Math.round(gpu) + '%';
        this.elements.ramBar.style.width = ram + '%';
        this.elements.ramValue.textContent = Math.round(ram) + '%';

        // Update latency (simulated)
        this.elements.latencyValue.textContent = (5 + Math.random() * 10).toFixed(0) + 'ms';
    }

    updateDetections(detections) {
        if (!Array.isArray(detections) || detections.length === 0) {
            this.elements.detectionList.innerHTML = `
                <div class="empty-state">
                    <i data-lucide="eye-off"></i>
                    <span>No detections</span>
                </div>
            `;
            this.elements.detectionCount.textContent = '0';
            lucide.createIcons();
            return;
        }

        this.elements.detectionCount.textContent = detections.length;

        const icons = {
            'vehicle': '&#x1F697;',
            'pedestrian': '&#x1F6B6;',
            'cyclist': '&#x1F6B4;',
            'unknown': '&#x2753;'
        };

        this.elements.detectionList.innerHTML = detections.map(det => `
            <div class="detection-item ${det.type}">
                <div class="detection-icon">${icons[det.type] || icons.unknown}</div>
                <div class="detection-info">
                    <div class="detection-label">${det.label || det.type}</div>
                    <div class="detection-meta">Conf: ${((det.confidence || 0) * 100).toFixed(0)}%</div>
                </div>
                <div class="detection-distance">${det.distance}m</div>
            </div>
        `).join('');

        // Update confidence
        const avgConf = detections.reduce((sum, d) => sum + (d.confidence || 0), 0) / detections.length;
        this.elements.confidenceValue.textContent = Math.round(avgConf * 100) + '%';
        this.elements.confidenceFill.style.width = (avgConf * 100) + '%';
    }

    log(message, level = 'info') {
        const console = this.elements.console;
        const line = document.createElement('div');
        line.className = `console-line ${level}`;

        const time = new Date().toLocaleTimeString('en-US', { hour12: false });
        line.textContent = `[${time}] ${message}`;

        console.appendChild(line);
        console.scrollTop = console.scrollHeight;

        // Limit lines
        while (console.children.length > 50) {
            console.removeChild(console.firstChild);
        }
    }
}

// Demo mode for testing without ROS
class DemoMode {
    constructor(dashboard) {
        this.dashboard = dashboard;
        this.time = 0;
    }

    start() {
        this.dashboard.log('Demo mode started', 'warn');
        this.dashboard.updateConnectionStatus(true);

        setInterval(() => {
            this.time += 0.1;
            this.updateDemo();
        }, 100);
    }

    updateDemo() {
        // Simulate vehicle state
        const state = {
            speed_kmh: 45 + Math.sin(this.time * 0.5) * 20,
            steering_angle: Math.sin(this.time * 0.3) * 15,
            battery_percent: 78,
            power: 30 + Math.sin(this.time * 0.4) * 25,
            gear: 'D',
            mode: 'FSD Demo'
        };
        this.dashboard.updateVehicleState(state);

        // Simulate detections
        const detections = [
            {
                type: 'vehicle',
                label: 'Tesla Model 3',
                distance: (8 + Math.sin(this.time) * 2).toFixed(1),
                confidence: 0.95
            },
            {
                type: 'pedestrian',
                label: 'Pedestrian',
                distance: (6 + Math.cos(this.time) * 1).toFixed(1),
                confidence: 0.88
            },
            {
                type: 'cyclist',
                label: 'Cyclist',
                distance: (12 + Math.sin(this.time * 0.5) * 3).toFixed(1),
                confidence: 0.92
            }
        ];
        this.dashboard.updateDetections(detections);
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    const dashboard = new ROSDashboard();

    // Check if running without ROS, start demo mode
    setTimeout(() => {
        if (!dashboard.connected) {
            const demo = new DemoMode(dashboard);
            demo.start();
        }
    }, 5000);
});

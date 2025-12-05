/**
 * Alpamayo FSD Dashboard
 * ROS2 WebSocket + Video Server Connection
 */

// Configuration
const CONFIG = {
    rosbridgeUrl: '',
    videoServerUrl: '',
    cameraTopic: '/camera/image_raw',
    maxSteeringAngle: 30
};

// State
const state = {
    targetSpeed: 0,
    currentSpeed: 0,
    targetSteering: 0,
    currentSteering: 0,
    sampleCount: 0,
    connected: false,
    demoMode: false,
    detections: []
};

// DOM Elements
let elements = {};

// ROS Connection
let ros = null;
let cmdTopic = null;
let feedbackTopic = null;
let detectionsTopic = null;

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    elements = {
        rosStatus: document.getElementById('rosStatus'),
        rosStatusDot: document.getElementById('rosStatusDot'),
        speedValue: document.getElementById('speedValue'),
        steeringValue: document.getElementById('steeringValue'),
        steeringIndicator: document.getElementById('steeringIndicator'),
        modeChip: document.getElementById('modeChip'),
        sampleCount: document.getElementById('sampleCount'),
        egoCar: document.getElementById('egoCar'),
        pathLine: document.getElementById('pathLine'),
        detectionList: document.getElementById('detectionList'),
        detectionBoxes: document.getElementById('detectionBoxes'),
        cameraFeed: document.getElementById('cameraFeed'),
        cameraPlaceholder: document.getElementById('cameraPlaceholder'),
        cpuValue: document.getElementById('cpuValue'),
        gpuValue: document.getElementById('gpuValue'),
        latencyValue: document.getElementById('latencyValue'),
        inferenceValue: document.getElementById('inferenceValue')
    };

    loadConfig();
    animate();
});

// Config Management
function loadConfig() {
    const saved = localStorage.getItem('alpamayo_config');
    if (saved) {
        const parsed = JSON.parse(saved);
        document.getElementById('rosHost').value = parsed.rosHost || '';
        document.getElementById('rosPort').value = parsed.rosPort || '9090';
        document.getElementById('videoHost').value = parsed.videoHost || '';
        document.getElementById('videoPort').value = parsed.videoPort || '8080';
    }
}

function saveConfig() {
    const config = {
        rosHost: document.getElementById('rosHost').value,
        rosPort: document.getElementById('rosPort').value,
        videoHost: document.getElementById('videoHost').value,
        videoPort: document.getElementById('videoPort').value
    };
    localStorage.setItem('alpamayo_config', JSON.stringify(config));
}

function setPreset(type) {
    const presets = {
        'jetson': { host: '192.168.4.1' },
        'localhost': { host: 'localhost' },
        'orin': { host: '192.168.55.1' }
    };
    const p = presets[type];
    document.getElementById('rosHost').value = p.host;
    document.getElementById('videoHost').value = p.host;
}

// Modal Control
function showConnectionModal() {
    document.getElementById('connectionModal').classList.remove('hidden');
}

function hideConnectionModal() {
    document.getElementById('connectionModal').classList.add('hidden');
}

// ROS Connection
function connectToROS() {
    const rosHost = document.getElementById('rosHost').value || 'localhost';
    const rosPort = document.getElementById('rosPort').value || '9090';
    const videoHost = document.getElementById('videoHost').value || rosHost;
    const videoPort = document.getElementById('videoPort').value || '8080';

    CONFIG.rosbridgeUrl = `ws://${rosHost}:${rosPort}`;
    CONFIG.videoServerUrl = `http://${videoHost}:${videoPort}`;

    saveConfig();
    hideConnectionModal();
    connect();
}

function connect() {
    if (!CONFIG.rosbridgeUrl) {
        showConnectionModal();
        return;
    }

    elements.rosStatus.textContent = 'Connecting...';

    ros = new ROSLIB.Ros({ url: CONFIG.rosbridgeUrl });

    ros.on('connection', () => {
        state.connected = true;
        state.demoMode = false;
        updateConnectionStatus(true);
        setupTopics();
        setupCameraFeed();
    });

    ros.on('close', () => {
        state.connected = false;
        updateConnectionStatus(false);
        setTimeout(() => {
            if (!state.demoMode) connect();
        }, 3000);
    });

    ros.on('error', (error) => {
        console.error('ROS error:', error);
        elements.rosStatus.textContent = 'Connection Error';
    });
}

function updateConnectionStatus(connected) {
    elements.rosStatus.textContent = connected ? 'Connected' : 'Disconnected';
    elements.rosStatusDot.classList.toggle('disconnected', !connected);
}

function setupTopics() {
    // Planner command
    cmdTopic = new ROSLIB.Topic({
        ros: ros,
        name: '/planner/cmd',
        messageType: 'geometry_msgs/Vector3'
    });

    cmdTopic.subscribe((msg) => {
        state.sampleCount++;
        elements.sampleCount.textContent = state.sampleCount;

        state.targetSteering = msg.x * 180 / Math.PI;
        state.targetSpeed = msg.y;

        if (Math.abs(state.targetSpeed) > 0.1) {
            elements.modeChip.textContent = 'FSD';
            elements.modeChip.classList.add('active');
        } else {
            elements.modeChip.textContent = 'MANUAL';
            elements.modeChip.classList.remove('active');
        }
    });

    // Feedback topic
    feedbackTopic = new ROSLIB.Topic({
        ros: ros,
        name: '/training/feedback',
        messageType: 'std_msgs/String'
    });

    // Detections
    detectionsTopic = new ROSLIB.Topic({
        ros: ros,
        name: '/planner/detections_json',
        messageType: 'std_msgs/String'
    });

    detectionsTopic.subscribe((msg) => {
        try {
            state.detections = JSON.parse(msg.data);
            updateDetections();
        } catch (e) {
            console.error('Detection parse error:', e);
        }
    });
}

function setupCameraFeed() {
    const url = `${CONFIG.videoServerUrl}/stream?topic=${CONFIG.cameraTopic}&type=mjpeg&quality=60`;
    elements.cameraFeed.src = url;
    elements.cameraFeed.style.display = 'block';
    elements.cameraPlaceholder.style.display = 'none';

    elements.cameraFeed.onerror = () => {
        elements.cameraFeed.style.display = 'none';
        elements.cameraPlaceholder.style.display = 'flex';
    };
}

function sendFeedback(type) {
    if (feedbackTopic && state.connected) {
        feedbackTopic.publish({ data: type });
    }

    const btn = event.target;
    btn.style.background = type === 'good' ? 'rgba(34, 197, 94, 0.3)' : 'rgba(239, 68, 68, 0.3)';
    setTimeout(() => { btn.style.background = ''; }, 200);
}

// Detection Updates
function updateDetections() {
    const icons = {
        vehicle: '🚗',
        pedestrian: '🚶',
        cyclist: '🚴',
        unknown: '❓'
    };

    if (state.detections.length === 0) {
        elements.detectionList.innerHTML = `
            <div class="detection-item">
                <span class="detection-icon">✓</span>
                <div class="detection-info">
                    <div class="detection-name">Clear path</div>
                    <div class="detection-meta">No obstacles detected</div>
                </div>
            </div>
        `;
        elements.detectionBoxes.innerHTML = '';
        return;
    }

    elements.detectionList.innerHTML = state.detections.map(det => `
        <div class="detection-item ${det.type || 'unknown'}">
            <span class="detection-icon">${icons[det.type] || icons.unknown}</span>
            <div class="detection-info">
                <div class="detection-name">${det.label || det.type || 'Object'}</div>
                <div class="detection-meta">Conf: ${((det.confidence || 0) * 100).toFixed(0)}%</div>
            </div>
            <span class="detection-distance">${det.distance || '--'}m</span>
        </div>
    `).join('');

    updateDetectionBoxes();
}

function updateDetectionBoxes() {
    elements.detectionBoxes.innerHTML = state.detections.slice(0, 5).map((det, i) => {
        const dist = parseFloat(det.distance) || 10;
        const scale = Math.max(0.3, 1 - dist / 20);
        const top = 20 + (1 - scale) * 40;
        const left = 40 + (i - 2) * 15;

        return `
            <div class="detection-box ${det.type || ''}"
                 style="width: ${40 * scale}px; height: ${60 * scale}px;
                        top: ${top}%; left: ${left}%;
                        transform: translate(-50%, -50%);">
                <div class="detection-label">${det.distance}m</div>
            </div>
        `;
    }).join('');
}

// Animation Loop
function animate() {
    const lerp = 0.15;
    state.currentSpeed += (state.targetSpeed - state.currentSpeed) * lerp;
    state.currentSteering += (state.targetSteering - state.currentSteering) * lerp;

    elements.speedValue.textContent = Math.abs(state.currentSpeed).toFixed(1);
    elements.steeringValue.textContent = state.currentSteering.toFixed(0) + '°';

    const clampedSteering = Math.max(-CONFIG.maxSteeringAngle,
                                     Math.min(CONFIG.maxSteeringAngle, state.currentSteering));

    const steeringPercent = 50 + (clampedSteering / CONFIG.maxSteeringAngle) * 40;
    elements.steeringIndicator.style.left = steeringPercent + '%';

    elements.egoCar.style.transform = `translateX(-50%) rotate(${clampedSteering * 0.5}deg)`;
    elements.pathLine.style.transform = `skewX(${-clampedSteering * 0.8}deg)`;

    if (state.connected || state.demoMode) {
        elements.cpuValue.textContent = Math.round(40 + Math.random() * 30);
        elements.gpuValue.textContent = Math.round(50 + Math.random() * 40);
        elements.latencyValue.textContent = Math.round(5 + Math.random() * 10);
        elements.inferenceValue.textContent = Math.round(8 + Math.random() * 6);
    }

    requestAnimationFrame(animate);
}

// Demo Mode
function startDemoMode() {
    hideConnectionModal();
    state.demoMode = true;
    state.connected = false;

    elements.rosStatus.textContent = 'Demo Mode';
    elements.rosStatusDot.classList.remove('disconnected');
    elements.modeChip.textContent = 'DEMO';
    elements.modeChip.classList.add('active');

    let t = 0;
    setInterval(() => {
        t += 0.05;
        state.targetSpeed = 1.5 + Math.sin(t * 0.3) * 0.8;
        state.targetSteering = Math.sin(t * 0.4) * 20;
        state.sampleCount++;
        elements.sampleCount.textContent = state.sampleCount;

        state.detections = [
            { type: 'vehicle', label: 'Car', distance: (8 + Math.sin(t) * 2).toFixed(1), confidence: 0.95 },
            { type: 'pedestrian', label: 'Person', distance: (5 + Math.cos(t) * 1).toFixed(1), confidence: 0.88 },
            { type: 'cyclist', label: 'Bike', distance: (12 + Math.sin(t * 0.5) * 3).toFixed(1), confidence: 0.91 }
        ];
        updateDetections();
    }, 100);
}

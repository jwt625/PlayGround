// Escalator 3D Visualization
let scene, camera, renderer, controls;
let escalatorGroup;

// Default parameters (in mm, converted to scene units: 1 unit = 100mm)
const params = {
    stepWidth: 1000,
    stepDepth: 400,
    stepRise: 200,
    grooveCount: 24,
    grooveWidth: 4,
    grooveDepth: 5,
    overhang: 15,
    stepCount: 18,
    incline: 30
};

function init() {
    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);

    // Camera
    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(15, 12, 20);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    document.getElementById('container').appendChild(renderer.domElement);

    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 20, 10);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    // Ground plane
    const groundGeometry = new THREE.PlaneGeometry(50, 50);
    const groundMaterial = new THREE.MeshStandardMaterial({ color: 0xcccccc });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.1;
    ground.receiveShadow = true;
    scene.add(ground);

    // Build escalator
    buildEscalator();

    // Event listeners
    setupControls();
    window.addEventListener('resize', onWindowResize);

    animate();
}

function buildEscalator() {
    if (escalatorGroup) {
        scene.remove(escalatorGroup);
        escalatorGroup.traverse((obj) => {
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) obj.material.dispose();
        });
    }

    escalatorGroup = new THREE.Group();
    const scale = 0.01; // Convert mm to scene units

    const stepW = params.stepWidth * scale;
    const stepD = params.stepDepth * scale;
    const stepH = params.stepRise * scale;
    const overhang = params.overhang * scale;
    const grooveW = params.grooveWidth * scale;
    const grooveD = params.grooveDepth * scale;
    const inclineRad = (params.incline * Math.PI) / 180;

    for (let i = 0; i < params.stepCount; i++) {
        const stepGroup = createStep(stepW, stepD, stepH, overhang, grooveW, grooveD, params.grooveCount);
        
        // Position along incline
        const xOffset = i * stepD * Math.cos(inclineRad);
        const yOffset = i * stepH;
        
        stepGroup.position.set(xOffset, yOffset, 0);
        escalatorGroup.add(stepGroup);
    }

    // Center the escalator
    const totalLength = params.stepCount * stepD * Math.cos(inclineRad);
    const totalHeight = params.stepCount * stepH;
    escalatorGroup.position.set(-totalLength / 2, 0, 0);

    scene.add(escalatorGroup);
}

function createStep(width, depth, height, overhang, grooveWidth, grooveDepth, grooveCount) {
    const stepGroup = new THREE.Group();
    const stepMaterial = new THREE.MeshStandardMaterial({ color: 0xffffff });
    const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x000000 });
    const grooveMaterial = new THREE.MeshStandardMaterial({ color: 0x333333 });

    // Main step body (tread)
    const treadThickness = height * 0.3;
    const treadGeometry = new THREE.BoxGeometry(depth, treadThickness, width);
    const tread = new THREE.Mesh(treadGeometry, stepMaterial);
    tread.position.set(depth / 2, height - treadThickness / 2, 0);
    tread.castShadow = true;
    stepGroup.add(tread);

    // Add edges to tread
    const treadEdges = new THREE.EdgesGeometry(treadGeometry);
    const treadLine = new THREE.LineSegments(treadEdges, edgeMaterial);
    treadLine.position.copy(tread.position);
    stepGroup.add(treadLine);

    // Riser (vertical part)
    const riserGeometry = new THREE.BoxGeometry(0.02, height - treadThickness, width);
    const riser = new THREE.Mesh(riserGeometry, stepMaterial);
    riser.position.set(0.01, (height - treadThickness) / 2, 0);
    stepGroup.add(riser);

    // Grooves on tread surface
    const grooveSpacing = width / (grooveCount + 1);
    for (let g = 1; g <= grooveCount; g++) {
        const zPos = -width / 2 + g * grooveSpacing;
        
        // Groove on tread
        const grooveGeom = new THREE.BoxGeometry(depth, grooveDepth, grooveWidth);
        const groove = new THREE.Mesh(grooveGeom, grooveMaterial);
        groove.position.set(depth / 2, height - grooveDepth / 2 + 0.001, zPos);
        stepGroup.add(groove);

        // Overhang groove (extends past the front edge)
        const overhangGeom = new THREE.BoxGeometry(overhang, grooveDepth * 1.5, grooveWidth);
        const overhangGroove = new THREE.Mesh(overhangGeom, grooveMaterial);
        overhangGroove.position.set(depth + overhang / 2, height - grooveDepth * 0.75, zPos);
        stepGroup.add(overhangGroove);
    }

    return stepGroup;
}

function setupControls() {
    const controlIds = [
        'stepWidth', 'stepDepth', 'stepRise',
        'grooveCount', 'grooveWidth', 'grooveDepth', 'overhang',
        'stepCount', 'incline'
    ];

    controlIds.forEach(id => {
        const input = document.getElementById(id);
        const valueSpan = document.getElementById(id + 'Val');

        input.addEventListener('input', () => {
            const val = parseFloat(input.value);
            valueSpan.textContent = val;
            params[id] = val;
            buildEscalator();
        });
    });
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// Keyboard movement state
const moveState = { w: false, a: false, s: false, d: false, q: false, e: false };
const moveSpeed = 0.15;

function setupKeyboardControls() {
    document.addEventListener('keydown', (event) => {
        const key = event.key.toLowerCase();
        if (key in moveState) {
            moveState[key] = true;
            event.preventDefault();
        }
    });

    document.addEventListener('keyup', (event) => {
        const key = event.key.toLowerCase();
        if (key in moveState) {
            moveState[key] = false;
        }
    });
}

function updateCameraPosition() {
    // Get camera's forward and right vectors (ignoring y for horizontal movement)
    const forward = new THREE.Vector3();
    camera.getWorldDirection(forward);

    const right = new THREE.Vector3();
    right.crossVectors(forward, camera.up).normalize();

    // Horizontal forward (for W/S movement along ground plane relative to view)
    const forwardHorizontal = new THREE.Vector3(forward.x, 0, forward.z).normalize();
    const rightHorizontal = new THREE.Vector3(right.x, 0, right.z).normalize();

    const movement = new THREE.Vector3();

    // W/S - forward/backward relative to camera orientation
    if (moveState.w) movement.add(forwardHorizontal);
    if (moveState.s) movement.sub(forwardHorizontal);

    // A/D - left/right strafe relative to camera orientation
    if (moveState.a) movement.sub(rightHorizontal);
    if (moveState.d) movement.add(rightHorizontal);

    // Q/E - down/up (world space)
    if (moveState.q) movement.y -= 1;
    if (moveState.e) movement.y += 1;

    if (movement.length() > 0) {
        movement.normalize().multiplyScalar(moveSpeed);
        camera.position.add(movement);
        controls.target.add(movement);
    }
}

function animate() {
    requestAnimationFrame(animate);
    updateCameraPosition();
    controls.update();
    renderer.render(scene, camera);
}

// Initialize
init();
setupKeyboardControls();


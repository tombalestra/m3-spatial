class GaussianViewer {
    constructor() {
        this.canvas = document.getElementById('renderCanvas');
        this.gl = this.canvas.getContext('webgl2');
        if (!this.gl) {
            alert('WebGL2 not supported');
            return;
        }

        // Default dimensions (will be updated from server)
        this.width = 256;
        this.height = 256;
        
        // Initialize display elements early
        this.positionDisplay = document.getElementById('position');
        this.rotationDisplay = document.getElementById('rotation');
        
        // Fetch configuration and initial camera from server, then initialize
        Promise.all([
            this.fetchInitialConfig(),
            this.fetchInitialCamera()
        ]).then(() => {
            this.initialize();
        });
    }
    
    async fetchInitialConfig() {
        try {
            const response = await fetch('/config');
            if (!response.ok) throw new Error('Failed to fetch config');
            
            const config = await response.json();
            this.width = config.width;
            this.height = config.height;
            console.log(`Initialized with dimensions: ${this.width}x${this.height}`);
        } catch (error) {
            console.error('Error fetching config:', error);
            // Keep default dimensions on error
        }
    }
    
    async fetchInitialCamera() {
        try {
            const response = await fetch('/initial_camera');
            if (!response.ok) throw new Error('Failed to fetch initial camera');
            
            const cameraData = await response.json();
            this.initialPosition = cameraData.position;
            this.initialRotation = cameraData.rotation;
            console.log('Initialized with camera:', cameraData);
        } catch (error) {
            console.error('Error fetching initial camera:', error);
            // Set default values on error
            this.initialPosition = [0, 0, 5];
            this.initialRotation = [0, -90, 0]; // pitch, yaw, roll
        }
    }
    
    initialize() {
        this.updateCanvasAspectRatio();

        // Use initial camera position and rotation from server
        this.camera = {
            position: vec3.fromValues(...this.initialPosition),
            rotation: vec3.fromValues(0, 0, 0),
            front: vec3.fromValues(0, 0, -1),
            up: vec3.fromValues(0, 1, 0),
            right: vec3.fromValues(1, 0, 0),
            viewMatrix: mat4.create(),
            projMatrix: mat4.create(),
            pitch: this.initialRotation[0],
            yaw: this.initialRotation[1],
            roll: this.initialRotation[2]
        };

        this.movement = {
            speed: 0.1,
            mouseSpeed: 0.1,
            zoomSpeed: 0.5
        };

        this.isRGBMode = true;
        this.isDragging = false;
        this.lastMousePos = { x: 0, y: 0 };

        this.initShaders();
        this.initBuffers();
        this.setupEventListeners();
        this.resize();
        this.updateCameraVectors();
        
        // Initial camera setup
        this.updateCamera();
        this.render();
        this.requestNewFrame();

        // Update displays with initial camera values
        this.updateAllDisplays();

        console.log("GaussianViewer initialized");
    }

    // New method to update all displays at once
    updateAllDisplays() {
        this.updatePositionDisplay();
        this.updateRotationDisplay();
    }

    initShaders() {
        const vsSource = `#version 300 es
            in vec4 aPosition;
            in vec2 aTexCoord;
            out vec2 vTexCoord;
            void main() {
                gl_Position = aPosition;
                vTexCoord = aTexCoord;
            }`;

        const fsSource = `#version 300 es
            precision highp float;
            in vec2 vTexCoord;
            uniform sampler2D uTexture;
            out vec4 fragColor;
            void main() {
                fragColor = texture(uTexture, vTexCoord);
            }`;

        const vertexShader = this.compileShader(vsSource, this.gl.VERTEX_SHADER);
        const fragmentShader = this.compileShader(fsSource, this.gl.FRAGMENT_SHADER);

        this.program = this.gl.createProgram();
        this.gl.attachShader(this.program, vertexShader);
        this.gl.attachShader(this.program, fragmentShader);
        this.gl.linkProgram(this.program);

        if (!this.gl.getProgramParameter(this.program, this.gl.LINK_STATUS)) {
            console.error('Shader program failed to link');
            return;
        }
    }

    compileShader(source, type) {
        const shader = this.gl.createShader(type);
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);

        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            console.error('Shader compile error:', this.gl.getShaderInfoLog(shader));
            this.gl.deleteShader(shader);
            return null;
        }
        return shader;
    }

    initBuffers() {
        const positions = new Float32Array([
            -1, -1,  1, -1,  -1, 1,   // First triangle
             1, -1,  1,  1,  -1, 1    // Second triangle
        ]);
    
        // Adjusted texture coordinates (flipped vertically)
        const texCoords = new Float32Array([
            0, 1,  1, 1,  0, 0,    // First triangle flipped vertically
            1, 1,  1, 0,  0, 0     // Second triangle flipped vertically
        ]);
    
        this.positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);
    
        this.texCoordBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.texCoordBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, texCoords, this.gl.STATIC_DRAW);
    }
    

    // New method that updates camera direction without changing position
    updateCamera() {
        this.updateCameraVectors();
        
        // Calculate target point by adding front vector to position
        const target = vec3.create();
        vec3.add(target, this.camera.position, this.camera.front);
        
        // Create view matrix with lookAt
        mat4.lookAt(
            this.camera.viewMatrix,
            this.camera.position,
            target,
            this.camera.up
        );
    }

    lookAtCenter() {
        const target = vec3.fromValues(0, 0, 0);
        
        // Calculate direction vector from camera to target
        const direction = vec3.create();
        vec3.subtract(direction, target, this.camera.position);
        vec3.normalize(direction, direction);
        
        // Calculate yaw and pitch from direction vector
        this.camera.yaw = Math.atan2(direction[2], direction[0]) * 180 / Math.PI;
        const horizontalLength = Math.sqrt(direction[0] * direction[0] + direction[2] * direction[2]);
        this.camera.pitch = Math.atan2(direction[1], horizontalLength) * 180 / Math.PI;
        
        // Update the camera vectors with new yaw/pitch
        this.updateCameraVectors();
        
        // Now create view matrix
        mat4.lookAt(
            this.camera.viewMatrix,
            this.camera.position,
            target,
            this.camera.up
        );
        
        // Make sure to update displays after changing camera orientation
        this.updateAllDisplays();
    }

    updateCameraVectors() {
        // Calculate new front vector
        const front = vec3.create();
        front[0] = Math.cos(this.camera.yaw * Math.PI / 180) * Math.cos(this.camera.pitch * Math.PI / 180);
        // Flip the Y component to match the Gaussian renderer's coordinate system
        front[1] = -Math.sin(this.camera.pitch * Math.PI / 180);
        front[2] = Math.sin(this.camera.yaw * Math.PI / 180) * Math.cos(this.camera.pitch * Math.PI / 180);
        vec3.normalize(this.camera.front, front);

        // Calculate right and up vectors
        vec3.cross(this.camera.right, this.camera.front, [0, 1, 0]);
        vec3.normalize(this.camera.right, this.camera.right);
        
        // Apply roll rotation to the up vector
        if (this.camera.roll !== 0) {
            // Create rotation matrix for roll around front vector
            const rollMatrix = mat4.create();
            mat4.fromRotation(rollMatrix, this.camera.roll * Math.PI / 180, this.camera.front);
            
            // Apply rotation to right vector
            vec3.transformMat4(this.camera.right, this.camera.right, rollMatrix);
        }
        
        vec3.cross(this.camera.up, this.camera.right, this.camera.front);
        vec3.normalize(this.camera.up, this.camera.up);
    }

    setupEventListeners() {
        window.addEventListener('resize', () => this.resize());
        
        document.addEventListener('keydown', (event) => {
            const velocity = this.movement.speed;
            const rotationStep = 5; // Degrees to rotate per keypress
            let moved = true;
            
            switch(event.key) {
                case 'w':
                case 'W':
                    // Move along positive world Y axis (up)
                    vec3.scaleAndAdd(this.camera.position, this.camera.position, [0, 1, 0], velocity);
                    break;
                case 's':
                case 'S':
                    // Move along negative world Y axis (down)
                    vec3.scaleAndAdd(this.camera.position, this.camera.position, [0, 1, 0], -velocity);
                    break;
                case 'a':
                case 'A':
                    // Move along negative world X axis (left)
                    vec3.scaleAndAdd(this.camera.position, this.camera.position, [1, 0, 0], velocity);
                    break;
                case 'd':
                case 'D':
                    // Move along positive world X axis (right)
                    vec3.scaleAndAdd(this.camera.position, this.camera.position, [1, 0, 0], -velocity);
                    break;
                case 'q':
                case 'Q':
                    // Move along positive world Z axis (forward)
                    vec3.scaleAndAdd(this.camera.position, this.camera.position, [0, 0, 1], velocity);
                    break;
                case 'e':
                case 'E':
                    // Move along negative world Z axis (backward)
                    vec3.scaleAndAdd(this.camera.position, this.camera.position, [0, 0, 1], -velocity);
                    break;
                case 'z':
                case 'Z':
                    // Control pitch
                    if (event.shiftKey) {
                        // Decrease pitch (look down)
                        this.camera.pitch -= rotationStep;
                    } else {
                        // Increase pitch (look up)
                        this.camera.pitch += rotationStep;
                    }
                    // Limit pitch to avoid gimbal lock
                    if (this.camera.pitch > 89.0) this.camera.pitch = 89.0;
                    if (this.camera.pitch < -89.0) this.camera.pitch = -89.0;
                    break;
                case 'x':
                case 'X':
                    // Control yaw
                    if (event.shiftKey) {
                        // Decrease yaw (rotate left)
                        this.camera.yaw -= rotationStep;
                    } else {
                        // Increase yaw (rotate right)
                        this.camera.yaw += rotationStep;
                    }
                    break;
                case 'c':
                case 'C':
                    // Control roll
                    if (event.shiftKey) {
                        // Decrease roll (rotate counterclockwise)
                        this.camera.roll -= rotationStep;
                    } else {
                        // Increase roll (rotate clockwise)
                        this.camera.roll += rotationStep;
                    }
                    break;
                case '0':
                    // RGB mode
                    this.setRenderMode('rgb');
                    moved = false;
                    break;
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                    // Feature modes 1-6
                    this.setRenderMode('feature', parseInt(event.key));
                    moved = false;
                    break;
                default:
                    moved = false;
            }
            
            if (moved) {
                // Update camera without reorienting to center
                this.updateCamera();
                this.updateAllDisplays();
                this.requestNewFrame();
            }
        });

        document.getElementById('resetView').addEventListener('click', () => {
            this.resetView();
        });

        // Check if width/height inputs exist before adding event listeners
        const widthInput = document.getElementById('widthInput');
        const heightInput = document.getElementById('heightInput');
        
        if (widthInput) {
            widthInput.addEventListener('change', (event) => {
                const newWidth = parseInt(event.target.value);
                if (newWidth > 0) {
                    this.setDimensions(newWidth, this.height);
                }
            });
        }
        
        if (heightInput) {
            heightInput.addEventListener('change', (event) => {
                const newHeight = parseInt(event.target.value);
                if (newHeight > 0) {
                    this.setDimensions(this.width, newHeight);
                }
            });
        }
    }

    resetView() {
        // Reset camera position and orientation to initial values
        vec3.set(this.camera.position, ...this.initialPosition);
        this.camera.pitch = this.initialRotation[0];
        this.camera.yaw = this.initialRotation[1];
        this.camera.roll = this.initialRotation[2];
        
        // Use updateCamera instead of lookAtCenter
        this.updateCamera();
        
        // Update displays
        this.updateAllDisplays();
        
        // Request new frame
        this.requestNewFrame();
    }

    updatePositionDisplay() {
        if (this.positionDisplay) {
            const pos = this.camera.position;
            this.positionDisplay.textContent = `X: ${pos[0].toFixed(2)}, Y: ${pos[1].toFixed(2)}, Z: ${pos[2].toFixed(2)}`;
        } else {
            console.warn('Position display element not found');
        }
    }

    updateRotationDisplay() {
        if (this.rotationDisplay) {
            this.rotationDisplay.textContent = `Pitch: ${this.camera.pitch.toFixed(2)}°, Yaw: ${this.camera.yaw.toFixed(2)}°, Roll: ${this.camera.roll.toFixed(2)}°`;
        } else {
            console.warn('Rotation display element not found');
        }
    }

    resize() {
        const displayWidth = this.canvas.clientWidth;
        const displayHeight = this.canvas.clientHeight;

        if (this.canvas.width !== displayWidth || this.canvas.height !== displayHeight) {
            this.canvas.width = displayWidth;
            this.canvas.height = displayHeight;
            this.gl.viewport(0, 0, this.gl.canvas.width, this.gl.canvas.height);
            
            mat4.perspective(this.camera.projMatrix, 
                           45 * Math.PI / 180,
                           this.gl.canvas.width / this.gl.canvas.height,
                           0.1, 100.0);
        }
    }

    async requestNewFrame() {
        const cameraData = {
            position: Array.from(this.camera.position),
            rotation: [this.camera.pitch, this.camera.yaw, this.camera.roll], // Include roll in rotation
            mode: this.isRGBMode ? 'rgb' : 'feature',
            featureIndex: this.featureIndex || 0,
            width: this.width,
            height: this.height
        };

        try {
            const response = await fetch('/render', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(cameraData)
            });

            if (!response.ok) throw new Error('Network response was not ok');
            
            const blob = await response.blob();
            const img = new Image();
            img.src = URL.createObjectURL(blob);
            
            img.onload = () => {
                this.updateTexture(img);
                this.render();
                // Ensure displays are updated after rendering
                this.updateAllDisplays();
            };
        } catch (error) {
            console.error('Error fetching new frame:', error);
        }
    }

    updateTexture(image) {
        const texture = this.gl.createTexture();
        this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
        
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, 
                          this.gl.RGBA, this.gl.UNSIGNED_BYTE, image);
        
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
    }

    render() {
        this.gl.clearColor(0.0, 0.0, 0.0, 1.0);
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);

        this.gl.useProgram(this.program);

        const positionAttributeLocation = this.gl.getAttribLocation(this.program, 'aPosition');
        const texCoordAttributeLocation = this.gl.getAttribLocation(this.program, 'aTexCoord');

        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);
        this.gl.enableVertexAttribArray(positionAttributeLocation);
        this.gl.vertexAttribPointer(positionAttributeLocation, 2, this.gl.FLOAT, false, 0, 0);

        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.texCoordBuffer);
        this.gl.enableVertexAttribArray(texCoordAttributeLocation);
        this.gl.vertexAttribPointer(texCoordAttributeLocation, 2, this.gl.FLOAT, false, 0, 0);

        this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
    }

    // Add method to update canvas aspect ratio
    updateCanvasAspectRatio() {
        const container = document.querySelector('.viewer-container');
        const aspectRatio = this.width / this.height;
        container.style.aspectRatio = `${aspectRatio}/1`;
        this.canvas.style.aspectRatio = `${aspectRatio}/1`;
    }

    // Method to set width and height
    setDimensions(width, height) {
        if (this.width !== width || this.height !== height) {
            this.width = width;
            this.height = height;
            this.updateCanvasAspectRatio();
            this.requestNewFrame();
        }
    }

    // New method to set render mode
    setRenderMode(mode, featureIndex = 0) {
        const oldMode = this.isRGBMode ? 'rgb' : 'feature';
        const oldFeatureIndex = this.featureIndex || 0;
        
        this.isRGBMode = (mode === 'rgb');
        this.featureIndex = featureIndex;
        
        // Update the active mode in the UI
        this.updateModeDisplay();
        
        // Only request a new frame if the mode or feature index changed
        if (oldMode !== mode || (!this.isRGBMode && oldFeatureIndex !== featureIndex)) {
            this.requestNewFrame();
        }
    }

    // New method to update the mode display
    updateModeDisplay() {
        // Update the active class on mode buttons
        document.querySelectorAll('.mode-button').forEach(btn => {
            btn.classList.remove('active');
        });
        
        if (this.isRGBMode) {
            document.getElementById('mode-0').classList.add('active');
        } else {
            document.getElementById(`mode-${this.featureIndex}`).classList.add('active');
        }
    }
}

// Initialize the viewer when the page loads
window.addEventListener('load', () => {
    const viewer = new GaussianViewer();
});
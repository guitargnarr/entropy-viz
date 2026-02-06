/**
 * WebGL2 fallback renderer.
 * 512 particles, CPU physics, instanced point sprites with additive blending.
 * Two-pass Gaussian blur for metaball-ish glow.
 */

import { mat4Perspective, mat4LookAt } from './math-utils.js';

const PARTICLE_COUNT = 512;
const HIST_BINS = 64;

export class WebGLFallback {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = null;
    this.particles = null; // Float32Array: [x, y, z, vx, vy, vz, hx, hy, hz, speed] * N
    this.program = null;
    this.blurProgram = null;
    this.fbo = null;
    this.fboTexture = null;
    this.pendingHistogram = null;
  }

  async init() {
    this.gl = this.canvas.getContext('webgl2', {
      antialias: false,
      alpha: false,
      premultipliedAlpha: false,
      powerPreference: 'high-performance',
    });
    if (!this.gl) throw new Error('No WebGL2');

    const gl = this.gl;
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // additive

    // Detect iOS Safari point size limit
    const range = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE);
    this.maxPointSize = range ? range[1] : 64;
    console.log('Max point size:', this.maxPointSize);

    this._initParticles();
    this._createPrograms();
    this._createFBO();
  }

  _initParticles() {
    // BCC lattice for 512 particles
    this.particles = new Float32Array(PARTICLE_COUNT * 10);
    const side = Math.ceil(Math.cbrt(PARTICLE_COUNT / 2));
    const spacing = 5.0 / side;
    let idx = 0;

    for (let z = 0; z < side && idx < PARTICLE_COUNT; z++) {
      for (let y = 0; y < side && idx < PARTICLE_COUNT; y++) {
        for (let x = 0; x < side && idx < PARTICLE_COUNT; x++) {
          if (idx < PARTICLE_COUNT) {
            const px = (x - side / 2 + 0.5) * spacing;
            const py = (y - side / 2 + 0.5) * spacing;
            const pz = (z - side / 2 + 0.5) * spacing;
            const base = idx * 10;
            this.particles[base] = px; this.particles[base+1] = py; this.particles[base+2] = pz;
            this.particles[base+3] = 0; this.particles[base+4] = 0; this.particles[base+5] = 0;
            this.particles[base+6] = px; this.particles[base+7] = py; this.particles[base+8] = pz;
            this.particles[base+9] = 0;
            idx++;
          }
          if (idx < PARTICLE_COUNT) {
            const px = (x - side / 2 + 1.0) * spacing;
            const py = (y - side / 2 + 1.0) * spacing;
            const pz = (z - side / 2 + 1.0) * spacing;
            const base = idx * 10;
            this.particles[base] = px; this.particles[base+1] = py; this.particles[base+2] = pz;
            this.particles[base+3] = 0; this.particles[base+4] = 0; this.particles[base+5] = 0;
            this.particles[base+6] = px; this.particles[base+7] = py; this.particles[base+8] = pz;
            this.particles[base+9] = 0;
            idx++;
          }
        }
      }
    }
  }

  _createPrograms() {
    const gl = this.gl;

    // Main particle program
    const vsrc = `#version 300 es
      uniform mat4 u_viewProj;
      uniform float u_pointSize;
      in vec3 a_position;
      in float a_speed;
      out float v_speed;
      void main() {
        v_speed = a_speed;
        vec4 pos = u_viewProj * vec4(a_position, 1.0);
        gl_Position = pos;
        gl_PointSize = u_pointSize / max(pos.w, 0.1);
      }`;

    const fsrc = `#version 300 es
      precision highp float;
      in float v_speed;
      out vec4 fragColor;
      void main() {
        vec2 c = gl_PointCoord - 0.5;
        float d = length(c);
        if (d > 0.5) discard;
        float alpha = smoothstep(0.5, 0.0, d);
        // Color from speed: teal -> amber -> white
        float t = clamp(v_speed / 6.0, 0.0, 1.0);
        vec3 col = mix(vec3(0.08, 0.72, 0.65), vec3(0.98, 0.45, 0.09), t);
        col = mix(col, vec3(1.0, 0.96, 0.9), max(t - 0.8, 0.0) * 5.0);
        fragColor = vec4(col * alpha * 1.2, alpha * 0.8);
      }`;

    this.program = this._compileProgram(vsrc, fsrc);

    // Simple fullscreen blur
    const blurVS = `#version 300 es
      in vec2 a_pos;
      out vec2 v_uv;
      void main() {
        v_uv = a_pos * 0.5 + 0.5;
        gl_Position = vec4(a_pos, 0.0, 1.0);
      }`;

    const blurFS = `#version 300 es
      precision highp float;
      uniform sampler2D u_tex;
      uniform vec2 u_dir;
      uniform vec2 u_resolution;
      in vec2 v_uv;
      out vec4 fragColor;
      void main() {
        vec2 pixel = u_dir / u_resolution;
        vec4 sum = vec4(0.0);
        sum += texture(u_tex, v_uv - 4.0 * pixel) * 0.05;
        sum += texture(u_tex, v_uv - 3.0 * pixel) * 0.09;
        sum += texture(u_tex, v_uv - 2.0 * pixel) * 0.12;
        sum += texture(u_tex, v_uv - 1.0 * pixel) * 0.15;
        sum += texture(u_tex, v_uv) * 0.18;
        sum += texture(u_tex, v_uv + 1.0 * pixel) * 0.15;
        sum += texture(u_tex, v_uv + 2.0 * pixel) * 0.12;
        sum += texture(u_tex, v_uv + 3.0 * pixel) * 0.09;
        sum += texture(u_tex, v_uv + 4.0 * pixel) * 0.05;
        fragColor = sum * 1.8;
      }`;

    this.blurProgram = this._compileProgram(blurVS, blurFS);

    // Create buffers
    this.posBuffer = gl.createBuffer();
    this.speedBuffer = gl.createBuffer();

    // Fullscreen quad
    this.quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
  }

  _compileProgram(vsrc, fsrc) {
    const gl = this.gl;
    const vs = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vs, vsrc);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      console.error('VS:', gl.getShaderInfoLog(vs));
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fs, fsrc);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      console.error('FS:', gl.getShaderInfoLog(fs));
    }

    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    return prog;
  }

  _createFBO() {
    const gl = this.gl;
    const w = this.canvas.width || 1;
    const h = this.canvas.height || 1;

    // Cap FBO size to prevent iOS memory issues (max 2048 on any dimension)
    const maxDim = Math.min(gl.getParameter(gl.MAX_TEXTURE_SIZE), 2048);
    const fboW = Math.min(w, maxDim);
    const fboH = Math.min(h, maxDim);
    this.fboWidth = fboW;
    this.fboHeight = fboH;

    // Clean up old FBO
    if (this.fboTexture) gl.deleteTexture(this.fboTexture);
    if (this.fbo) gl.deleteFramebuffer(this.fbo);

    this.fboTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.fboTexture);

    // Use RGBA8 directly — RGBA16F is unsupported on iOS Safari
    // and the visual difference is negligible for additive blending
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, fboW, fboH, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.fboTexture, 0);

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      console.error('FBO incomplete:', status);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  updatePhysics(dt, tOrder, time) {
    const springK = 12.0;
    const noiseStr = 4.0;
    const dampOrdered = 0.97;
    const dampChaos = 0.999;
    const damping = dampChaos + (dampOrdered - dampChaos) * tOrder;

    // Simple seeded pseudo-random
    let seed = (time * 1000) | 0;
    const rand = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return (seed / 0x7fffffff) * 2 - 1;
    };

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const b = i * 10;
      // Spring toward home
      const dx = this.particles[b+6] - this.particles[b];
      const dy = this.particles[b+7] - this.particles[b+1];
      const dz = this.particles[b+8] - this.particles[b+2];

      const fx = dx * springK * tOrder + rand() * noiseStr * (1 - tOrder);
      const fy = dy * springK * tOrder + rand() * noiseStr * (1 - tOrder);
      const fz = dz * springK * tOrder + rand() * noiseStr * (1 - tOrder);

      this.particles[b+3] = this.particles[b+3] * damping + fx * dt;
      this.particles[b+4] = this.particles[b+4] * damping + fy * dt;
      this.particles[b+5] = this.particles[b+5] * damping + fz * dt;

      // Clamp
      const spd = Math.sqrt(this.particles[b+3]**2 + this.particles[b+4]**2 + this.particles[b+5]**2);
      if (spd > 8) {
        const s = 8 / spd;
        this.particles[b+3] *= s; this.particles[b+4] *= s; this.particles[b+5] *= s;
      }

      this.particles[b] += this.particles[b+3] * dt;
      this.particles[b+1] += this.particles[b+4] * dt;
      this.particles[b+2] += this.particles[b+5] * dt;
      this.particles[b+9] = Math.sqrt(this.particles[b+3]**2 + this.particles[b+4]**2 + this.particles[b+5]**2);

      // Soft boundary
      const distC = Math.sqrt(this.particles[b]**2 + this.particles[b+1]**2 + this.particles[b+2]**2);
      if (distC > 4.5) {
        const push = (distC - 4.5) * 2 * dt / distC;
        this.particles[b+3] -= this.particles[b] * push;
        this.particles[b+4] -= this.particles[b+1] * push;
        this.particles[b+5] -= this.particles[b+2] * push;
      }
    }

    // Build histogram
    const hist = new Uint32Array(HIST_BINS);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const spd = this.particles[i * 10 + 9];
      const bin = Math.min(Math.floor((spd / 8.0) * HIST_BINS), HIST_BINS - 1);
      hist[bin]++;
    }
    this.pendingHistogram = hist;
  }

  frame(dt, tOrder, time, camera) {
    const gl = this.gl;
    this.updatePhysics(dt, tOrder, time);

    const w = this.canvas.width;
    const h = this.canvas.height;
    const aspect = w / h;

    // Extract position and speed arrays
    const positions = new Float32Array(PARTICLE_COUNT * 3);
    const speeds = new Float32Array(PARTICLE_COUNT);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      positions[i*3] = this.particles[i*10];
      positions[i*3+1] = this.particles[i*10+1];
      positions[i*3+2] = this.particles[i*10+2];
      speeds[i] = this.particles[i*10+9];
    }

    const proj = mat4Perspective(45 * Math.PI / 180, aspect, 0.1, 100);
    const view = camera.viewMatrix;

    // viewProj = proj * view
    const vp = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        vp[j*4+i] = 0;
        for (let k = 0; k < 4; k++) {
          vp[j*4+i] += proj[k*4+i] * view[j*4+k];
        }
      }
    }

    // Pass 1: Render particles to FBO (may be smaller than canvas on iOS)
    const fboW = this.fboWidth || w;
    const fboH = this.fboHeight || h;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, fboW, fboH);
    gl.clearColor(0.008, 0.008, 0.02, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

    gl.useProgram(this.program);

    const vpLoc = gl.getUniformLocation(this.program, 'u_viewProj');
    gl.uniformMatrix4fv(vpLoc, false, vp);
    const psLoc = gl.getUniformLocation(this.program, 'u_pointSize');
    // Scale point size with canvas height, clamped to device max (iOS caps at ~63px)
    const desiredSize = Math.min(h, 1440) * 0.1;
    gl.uniform1f(psLoc, Math.min(desiredSize, this.maxPointSize * 0.9));

    const posLoc = gl.getAttribLocation(this.program, 'a_position');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

    const spdLoc = gl.getAttribLocation(this.program, 'a_speed');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.speedBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, speeds, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(spdLoc);
    gl.vertexAttribPointer(spdLoc, 1, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.POINTS, 0, PARTICLE_COUNT);

    // Pass 2: Blur and composite to screen
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);
    gl.clearColor(0.008, 0.008, 0.02, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.BLEND);

    gl.useProgram(this.blurProgram);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.fboTexture);
    gl.uniform1i(gl.getUniformLocation(this.blurProgram, 'u_tex'), 0);
    gl.uniform2f(gl.getUniformLocation(this.blurProgram, 'u_dir'), 2.0, 2.0);
    gl.uniform2f(gl.getUniformLocation(this.blurProgram, 'u_resolution'), fboW, fboH);

    const qLoc = gl.getAttribLocation(this.blurProgram, 'a_pos');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.enableVertexAttribArray(qLoc);
    gl.vertexAttribPointer(qLoc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  getHistogram() {
    return this.pendingHistogram;
  }

  resize(w, h) {
    this.canvas.width = w;
    this.canvas.height = h;
    if (this.gl) {
      this._createFBO();
    }
  }

  destroy() {}
}
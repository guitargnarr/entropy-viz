/**
 * WebGPU renderer: manages device, pipelines, buffers, and per-frame dispatch.
 *
 * Pipeline per frame:
 * 1. Update uniforms
 * 2. Compute: density-clear
 * 3. Compute: physics
 * 4. Compute: density-splat
 * 5. Compute: histogram
 * 6. Render: fullscreen quad + raymarch
 * 7. Copy histogram to readback buffer
 */

import { generateColorLUT } from './color-map.js';
import { mat4Perspective, mat4Inverse } from './math-utils.js';

const PARTICLE_COUNT = 2048;
const PARTICLE_STRIDE = 48; // bytes per particle
const GRID_RES = 64;
const GRID_VOXELS = GRID_RES * GRID_RES * GRID_RES; // 262144
const HIST_BINS = 64;
const GRID_MIN = [-3.5, -3.5, -3.5];
const GRID_MAX = [3.5, 3.5, 3.5];

export class WebGPURenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.device = null;
    this.context = null;
    this.format = null;

    // Buffers
    this.particleBuffer = null;
    this.densityBuffer = null;
    this.histogramBuffer = null;
    this.histReadbackBuffers = [null, null]; // double-buffered
    this.currentReadback = 0;
    this.pendingHistogram = null;

    // Pipelines
    this.physicsPipeline = null;
    this.densityClearPipeline = null;
    this.densitySplatPipeline = null;
    this.histogramClearPipeline = null;
    this.histogramPipeline = null;
    this.renderPipeline = null;

    // Bind groups
    this.physicsBindGroup = null;
    this.densityClearBindGroup = null;
    this.densitySplatBindGroup = null;
    this.histogramClearBindGroup = null;
    this.histogramBindGroup = null;
    this.renderBindGroup = null;

    // Uniform buffers
    this.physicsUniformBuffer = null;
    this.densityUniformBuffer = null;
    this.histUniformBuffer = null;
    this.raymarchUniformBuffer = null;

    // Textures
    this.colorLUTTexture = null;
    this.colorLUTSampler = null;

    // For reading density as non-atomic in render pass
    this.densityReadBuffer = null;
  }

  async init() {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance'
    });
    if (!adapter) throw new Error('No WebGPU adapter');

    // Request device with minimal limits (actual max buffer ~1MB)
    // Requesting 256MB caused failures on iOS Safari Metal backend
    this.device = await adapter.requestDevice();

    this.context = this.canvas.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'opaque',
    });

    await this._createBuffers();
    await this._createPipelines();
    this._createBindGroups();
  }

  async _loadShader(path) {
    const resp = await fetch(path);
    return await resp.text();
  }

  async _createBuffers() {
    const d = this.device;

    // Particle buffer
    const particleData = this._generateBCCLattice();
    this.particleBuffer = d.createBuffer({
      size: PARTICLE_COUNT * PARTICLE_STRIDE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    new Float32Array(this.particleBuffer.getMappedRange()).set(particleData);
    this.particleBuffer.unmap();

    // Density volume (atomic u32) — used in compute
    this.densityBuffer = d.createBuffer({
      size: GRID_VOXELS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Density read buffer (non-atomic u32) — read by render pass
    this.densityReadBuffer = d.createBuffer({
      size: GRID_VOXELS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Histogram buffer (atomic u32)
    this.histogramBuffer = d.createBuffer({
      size: HIST_BINS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Double-buffered readback
    for (let i = 0; i < 2; i++) {
      this.histReadbackBuffers[i] = d.createBuffer({
        size: HIST_BINS * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
    }

    // Uniform buffers
    this.physicsUniformBuffer = d.createBuffer({
      size: 32, // 8 x f32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.densityUniformBuffer = d.createBuffer({
      size: 48, // aligned
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.histUniformBuffer = d.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.raymarchUniformBuffer = d.createBuffer({
      size: 256, // 2 mat4 + extras
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Color LUT 1D texture
    const lutData = generateColorLUT(256);
    this.colorLUTTexture = d.createTexture({
      size: [256],
      format: 'rgba8unorm',
      dimension: '1d',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    d.queue.writeTexture(
      { texture: this.colorLUTTexture },
      lutData,
      { bytesPerRow: 256 * 4 },
      [256]
    );

    this.colorLUTSampler = d.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
    });
  }

  _generateBCCLattice() {
    // Body-centered cubic lattice: 2048 particles
    // BCC has 2 atoms per unit cell: (0,0,0) and (0.5,0.5,0.5)
    const data = new Float32Array(PARTICLE_COUNT * 12); // 12 floats per particle (48 bytes)
    const side = Math.ceil(Math.cbrt(PARTICLE_COUNT / 2)); // ~10
    const spacing = 5.0 / side; // fit within [-2.5, 2.5]
    let idx = 0;

    for (let z = 0; z < side && idx < PARTICLE_COUNT; z++) {
      for (let y = 0; y < side && idx < PARTICLE_COUNT; y++) {
        for (let x = 0; x < side && idx < PARTICLE_COUNT; x++) {
          // Corner atom
          if (idx < PARTICLE_COUNT) {
            const px = (x - side / 2 + 0.5) * spacing;
            const py = (y - side / 2 + 0.5) * spacing;
            const pz = (z - side / 2 + 0.5) * spacing;
            const base = idx * 12;
            data[base + 0] = px;     // position.x
            data[base + 1] = py;     // position.y
            data[base + 2] = pz;     // position.z
            data[base + 3] = 0;      // _pad0
            data[base + 4] = 0;      // velocity.x
            data[base + 5] = 0;      // velocity.y
            data[base + 6] = 0;      // velocity.z
            data[base + 7] = 0;      // speed
            data[base + 8] = px;     // home_position.x
            data[base + 9] = py;     // home_position.y
            data[base + 10] = pz;    // home_position.z
            data[base + 11] = 0;     // _pad1
            idx++;
          }

          // Body-center atom
          if (idx < PARTICLE_COUNT) {
            const px = (x - side / 2 + 1.0) * spacing;
            const py = (y - side / 2 + 1.0) * spacing;
            const pz = (z - side / 2 + 1.0) * spacing;
            const base = idx * 12;
            data[base + 0] = px;
            data[base + 1] = py;
            data[base + 2] = pz;
            data[base + 3] = 0;
            data[base + 4] = 0;
            data[base + 5] = 0;
            data[base + 6] = 0;
            data[base + 7] = 0;
            data[base + 8] = px;
            data[base + 9] = py;
            data[base + 10] = pz;
            data[base + 11] = 0;
            idx++;
          }
        }
      }
    }
    return data;
  }

  async _createPipelines() {
    const d = this.device;

    // Load all shaders in parallel
    const [physicsSrc, densityClearSrc, densitySplatSrc, histClearSrc, histogramSrc, quadSrc, raymarchSrc] =
      await Promise.all([
        this._loadShader('shaders/physics.wgsl'),
        this._loadShader('shaders/density-clear.wgsl'),
        this._loadShader('shaders/density-splat.wgsl'),
        this._loadShader('shaders/histogram-clear.wgsl'),
        this._loadShader('shaders/histogram.wgsl'),
        this._loadShader('shaders/fullscreen-quad.wgsl'),
        this._loadShader('shaders/raymarch.wgsl'),
      ]);

    // ── Physics pipeline ──
    this.physicsPipeline = d.createComputePipeline({
      layout: 'auto',
      compute: {
        module: d.createShaderModule({ code: physicsSrc }),
        entryPoint: 'main',
      },
    });

    // ── Density clear pipeline ──
    this.densityClearPipeline = d.createComputePipeline({
      layout: 'auto',
      compute: {
        module: d.createShaderModule({ code: densityClearSrc }),
        entryPoint: 'main',
      },
    });

    // ── Density splat pipeline ──
    this.densitySplatPipeline = d.createComputePipeline({
      layout: 'auto',
      compute: {
        module: d.createShaderModule({ code: densitySplatSrc }),
        entryPoint: 'main',
      },
    });

    // ── Histogram clear pipeline ──
    this.histogramClearPipeline = d.createComputePipeline({
      layout: 'auto',
      compute: {
        module: d.createShaderModule({ code: histClearSrc }),
        entryPoint: 'main',
      },
    });

    // ── Histogram pipeline ──
    this.histogramPipeline = d.createComputePipeline({
      layout: 'auto',
      compute: {
        module: d.createShaderModule({ code: histogramSrc }),
        entryPoint: 'main',
      },
    });

    // ── Render pipeline (fullscreen quad + raymarch) ──
    const quadModule = d.createShaderModule({ code: quadSrc });
    const raymarchModule = d.createShaderModule({ code: raymarchSrc });

    this.renderPipeline = d.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: quadModule,
        entryPoint: 'vs_main',
      },
      fragment: {
        module: raymarchModule,
        entryPoint: 'fs_main',
        targets: [{ format: this.format }],
      },
      primitive: { topology: 'triangle-list' },
    });
  }

  _createBindGroups() {
    const d = this.device;

    // Physics bind group
    this.physicsBindGroup = d.createBindGroup({
      layout: this.physicsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.particleBuffer } },
        { binding: 1, resource: { buffer: this.physicsUniformBuffer } },
      ],
    });

    // Density clear bind group
    this.densityClearBindGroup = d.createBindGroup({
      layout: this.densityClearPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.densityBuffer } },
      ],
    });

    // Density splat bind group
    this.densitySplatBindGroup = d.createBindGroup({
      layout: this.densitySplatPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.particleBuffer } },
        { binding: 1, resource: { buffer: this.densityBuffer } },
        { binding: 2, resource: { buffer: this.densityUniformBuffer } },
      ],
    });

    // Histogram clear bind group
    this.histogramClearBindGroup = d.createBindGroup({
      layout: this.histogramClearPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.histogramBuffer } },
      ],
    });

    // Histogram bind group
    this.histogramBindGroup = d.createBindGroup({
      layout: this.histogramPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.particleBuffer } },
        { binding: 1, resource: { buffer: this.histogramBuffer } },
        { binding: 2, resource: { buffer: this.histUniformBuffer } },
      ],
    });

    // Render bind group
    this.renderBindGroup = d.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.densityReadBuffer } },
        { binding: 1, resource: { buffer: this.raymarchUniformBuffer } },
        { binding: 2, resource: this.colorLUTTexture.createView() },
        { binding: 3, resource: this.colorLUTSampler },
      ],
    });
  }

  updateUniforms(dt, tOrder, time, camera) {
    const d = this.device;

    // Physics uniforms
    const physData = new Float32Array([
      dt,           // dt
      tOrder,       // t_order
      0.97,         // damping_ordered
      0.999,        // damping_chaos
      4.0,          // noise_strength
      12.0,         // spring_k
      time,         // time
    ]);
    const physBuf = new Float32Array(8);
    physBuf.set(physData);
    physBuf[7] = PARTICLE_COUNT; // particle_count as float bits reinterpreted — actually need u32
    // Write particle_count as u32
    const physMixed = new ArrayBuffer(32);
    const physF = new Float32Array(physMixed);
    const physU = new Uint32Array(physMixed);
    physF[0] = dt;
    physF[1] = tOrder;
    physF[2] = 0.97;
    physF[3] = 0.999;
    physF[4] = 4.0;
    physF[5] = 12.0;
    physF[6] = time;
    physU[7] = PARTICLE_COUNT;
    d.queue.writeBuffer(this.physicsUniformBuffer, 0, physMixed);

    // Density uniforms (48 bytes)
    const densMixed = new ArrayBuffer(48);
    const densF = new Float32Array(densMixed);
    const densU = new Uint32Array(densMixed);
    densF[0] = GRID_MIN[0]; densF[1] = GRID_MIN[1]; densF[2] = GRID_MIN[2]; densF[3] = 0;
    densF[4] = GRID_MAX[0]; densF[5] = GRID_MAX[1]; densF[6] = GRID_MAX[2]; densF[7] = 0;
    densU[8] = GRID_RES;
    densU[9] = PARTICLE_COUNT;
    densF[10] = 2.0;  // splat_radius
    densF[11] = 1.5;  // splat_strength
    d.queue.writeBuffer(this.densityUniformBuffer, 0, densMixed);

    // Histogram uniforms
    const histMixed = new ArrayBuffer(16);
    const histU = new Uint32Array(histMixed);
    const histF = new Float32Array(histMixed);
    histU[0] = PARTICLE_COUNT;
    histU[1] = HIST_BINS;
    histF[2] = 8.0;  // max_speed
    histF[3] = 0;
    d.queue.writeBuffer(this.histUniformBuffer, 0, histMixed);

    // Raymarch uniforms (256 bytes)
    const aspect = this.canvas.width / this.canvas.height;
    const proj = mat4Perspective(45 * Math.PI / 180, aspect, 0.1, 100);
    const invView = mat4Inverse(camera.viewMatrix);
    const invProj = mat4Inverse(proj);

    const raymarchData = new ArrayBuffer(256);
    const rmF = new Float32Array(raymarchData);
    // inv_view (mat4) — offset 0
    rmF.set(invView, 0);
    // inv_proj (mat4) — offset 16
    rmF.set(invProj, 16);
    // camera_pos (vec3) — offset 32
    rmF[32] = camera.eye[0]; rmF[33] = camera.eye[1]; rmF[34] = camera.eye[2];
    rmF[35] = time; // time
    // grid_min (vec3) — offset 36
    rmF[36] = GRID_MIN[0]; rmF[37] = GRID_MIN[1]; rmF[38] = GRID_MIN[2]; rmF[39] = 0;
    // grid_max (vec3) — offset 40
    rmF[40] = GRID_MAX[0]; rmF[41] = GRID_MAX[1]; rmF[42] = GRID_MAX[2]; rmF[43] = 0;
    // scalar uniforms — offset 44
    rmF[44] = GRID_RES;       // grid_res
    rmF[45] = this.canvas.width;  // screen_width
    rmF[46] = this.canvas.height; // screen_height
    rmF[47] = 3.0;           // absorption coefficient
    d.queue.writeBuffer(this.raymarchUniformBuffer, 0, raymarchData);
  }

  frame(dt, tOrder, time, camera) {
    const d = this.device;
    this.updateUniforms(dt, tOrder, time, camera);

    const encoder = d.createCommandEncoder();

    // 1. Clear density volume
    const clearPass = encoder.beginComputePass();
    clearPass.setPipeline(this.densityClearPipeline);
    clearPass.setBindGroup(0, this.densityClearBindGroup);
    clearPass.dispatchWorkgroups(Math.ceil(GRID_VOXELS / 64));
    clearPass.end();

    // 2. Physics
    const physPass = encoder.beginComputePass();
    physPass.setPipeline(this.physicsPipeline);
    physPass.setBindGroup(0, this.physicsBindGroup);
    physPass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / 256));
    physPass.end();

    // 3. Density splat
    const splatPass = encoder.beginComputePass();
    splatPass.setPipeline(this.densitySplatPipeline);
    splatPass.setBindGroup(0, this.densitySplatBindGroup);
    splatPass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / 256));
    splatPass.end();

    // 4. Copy density buffer to read buffer (atomic -> non-atomic)
    encoder.copyBufferToBuffer(this.densityBuffer, 0, this.densityReadBuffer, 0, GRID_VOXELS * 4);

    // 5. Clear histogram
    const histClearPass = encoder.beginComputePass();
    histClearPass.setPipeline(this.histogramClearPipeline);
    histClearPass.setBindGroup(0, this.histogramClearBindGroup);
    histClearPass.dispatchWorkgroups(1); // 64 threads in 1 workgroup
    histClearPass.end();

    // 6. Histogram binning
    const histPass = encoder.beginComputePass();
    histPass.setPipeline(this.histogramPipeline);
    histPass.setBindGroup(0, this.histogramBindGroup);
    histPass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / 256));
    histPass.end();

    // 6. Render pass (fullscreen quad + raymarch)
    const textureView = this.context.getCurrentTexture().createView();
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: textureView,
        clearValue: { r: 0.008, g: 0.008, b: 0.02, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    renderPass.setPipeline(this.renderPipeline);
    renderPass.setBindGroup(0, this.renderBindGroup);
    renderPass.draw(3); // fullscreen triangle
    renderPass.end();

    d.queue.submit([encoder.finish()]);

    // 7. Copy histogram to readback in SEPARATE submit (so render isn't affected)
    const rbIdx = this.currentReadback;
    const rbBuf = this.histReadbackBuffers[rbIdx];
    if (rbBuf.mapState === 'unmapped') {
      const copyEncoder = d.createCommandEncoder();
      copyEncoder.copyBufferToBuffer(this.histogramBuffer, 0, rbBuf, 0, HIST_BINS * 4);
      d.queue.submit([copyEncoder.finish()]);

      // 8. Async readback of this buffer
      rbBuf.mapAsync(GPUMapMode.READ).then(() => {
        const data = new Uint32Array(rbBuf.getMappedRange().slice(0));
        rbBuf.unmap();
        this.pendingHistogram = data;
      }).catch(() => {
        // Buffer busy, skip
      });

      this.currentReadback = 1 - this.currentReadback;
    }
  }

  getHistogram() {
    return this.pendingHistogram;
  }

  resize(w, h) {
    this.canvas.width = w;
    this.canvas.height = h;
  }

  destroy() {
    if (this.device) {
      this.device.destroy();
    }
  }
}
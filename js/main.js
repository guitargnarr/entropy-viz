/**
 * Entry point: WebGPU detection, init, RAF loop, event handling.
 */

import { WebGPURenderer } from './webgpu-renderer.js';
import { WebGLFallback } from './webgl-fallback.js';
import { StateMachine } from './state-machine.js';
import { EntropyCalculator } from './entropy-calculator.js';
import { Camera } from './camera.js';
import { UI } from './ui.js';

async function main() {
  const canvas = document.getElementById('canvas');
  const ui = new UI();
  const stateMachine = new StateMachine();
  const entropy = new EntropyCalculator(64);
  const camera = new Camera();

  let renderer = null;
  let useWebGPU = false;

  // Resize canvas to device pixels
  function resize() {
    const dpr = Math.min(window.devicePixelRatio, 2);
    const w = Math.floor(window.innerWidth * dpr);
    const h = Math.floor(window.innerHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      if (renderer) renderer.resize(w, h);
    }
  }

  // Detect WebGPU
  try {
    if (navigator.gpu) {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        useWebGPU = true;
      }
    }
  } catch (e) {
    console.log('WebGPU not available, falling back to WebGL2');
  }

  resize();

  // Init renderer
  try {
    if (useWebGPU) {
      renderer = new WebGPURenderer(canvas);
      await renderer.init();
      ui.setRenderer('WebGPU');
      console.log('Using WebGPU');
    } else {
      throw new Error('Use WebGL2');
    }
  } catch (e) {
    console.log('WebGPU init failed, using WebGL2:', e.message);
    useWebGPU = false;
    renderer = new WebGLFallback(canvas);
    await renderer.init();
    ui.setRenderer('WebGL2');
    console.log('Using WebGL2');
  }

  ui.hideLoading();

  // Events
  window.addEventListener('resize', resize);

  canvas.addEventListener('click', () => {
    if (stateMachine.canClick) {
      stateMachine.click();
    }
  });

  // Touch support (passive: false required for iOS preventDefault)
  canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    if (stateMachine.canClick) {
      stateMachine.click();
    }
  }, { passive: false });

  // Prevent iOS elastic scroll / pull-to-refresh on canvas
  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
  }, { passive: false });

  // Mouse parallax
  window.addEventListener('mousemove', (e) => {
    const nx = (e.clientX / window.innerWidth) * 2 - 1;
    const ny = (e.clientY / window.innerHeight) * 2 - 1;
    camera.onMouseMove(nx, ny);
  });

  // RAF loop
  let lastTime = performance.now();
  let totalTime = 0;

  function frame(now) {
    requestAnimationFrame(frame);

    const dtRaw = (now - lastTime) / 1000;
    lastTime = now;
    // Clamp dt to prevent explosion after tab switch
    const dt = Math.min(dtRaw, 1 / 20);
    totalTime += dt;

    // Update state machine
    stateMachine.update(dt);

    // Update camera
    camera.update(dt);

    // Render frame
    renderer.frame(dt, stateMachine.tOrder, totalTime, camera);

    // Read histogram and compute entropy
    const hist = renderer.getHistogram();
    if (hist) {
      entropy.compute(hist);
    }
    entropy.updateDisplay(dt);

    // Update UI
    ui.update(entropy, stateMachine);
  }

  requestAnimationFrame(frame);
}

main().catch(err => {
  console.error('Fatal:', err);
  const loading = document.getElementById('loading-text');
  if (loading) {
    loading.textContent = 'Error: ' + err.message;
    loading.style.color = '#f97316';
  }
});
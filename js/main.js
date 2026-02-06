/**
 * Entry point: WebGPU detection, init, RAF loop, event handling.
 */

import { WebGPURenderer } from './webgpu-renderer.js';
import { WebGLFallback } from './webgl-fallback.js';
import { StateMachine } from './state-machine.js';
import { EntropyCalculator } from './entropy-calculator.js';
import { Camera } from './camera.js';
import { UI } from './ui.js';

// Debug overlay for mobile (shows errors on-screen)
const debugLines = [];
function dbg(msg) {
  console.log(msg);
  debugLines.push(msg);
  const el = document.getElementById('debug-overlay');
  if (el) el.textContent = debugLines.slice(-15).join('\n');
}

// Create debug overlay element
function createDebugOverlay() {
  const el = document.createElement('div');
  el.id = 'debug-overlay';
  el.style.cssText = 'position:fixed;top:80px;right:8px;z-index:9999;font:10px/1.4 monospace;color:#f97316;background:rgba(0,0,0,0.85);padding:8px;border-radius:4px;max-width:55%;word-break:break-all;pointer-events:none;white-space:pre-wrap;';
  document.body.appendChild(el);
}

// Capture all errors
window.onerror = (msg, src, line, col, err) => {
  dbg(`ERR: ${msg} @${line}:${col}`);
};
window.addEventListener('unhandledrejection', (e) => {
  dbg(`REJECT: ${e.reason}`);
});

async function main() {
  createDebugOverlay();
  dbg(`UA: ${navigator.userAgent.slice(0, 80)}`);
  dbg(`DPR: ${window.devicePixelRatio} | ${window.innerWidth}x${window.innerHeight}`);
  dbg(`WebGPU: ${!!navigator.gpu}`);

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

  // Detect WebGPU — but skip on iOS Safari (Metal backend has compatibility issues
  // with atomic storage buffers, 1D textures, and density-splat shader patterns)
  const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) ||
    (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
  dbg(`iOS: ${isIOS} | Safari: ${isSafari}`);

  if (isIOS) {
    dbg('iOS detected — forcing WebGL2 (WebGPU Metal compat issues)');
  }

  try {
    if (navigator.gpu && !isIOS) {
      dbg('Requesting GPU adapter...');
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        useWebGPU = true;
        dbg('GPU adapter OK');
      } else {
        dbg('GPU adapter null');
      }
    } else if (navigator.gpu && isIOS) {
      dbg('WebGPU available but skipped on iOS');
    }
  } catch (e) {
    dbg('WebGPU fail: ' + e.message);
  }

  resize();
  dbg(`Canvas: ${canvas.width}x${canvas.height}`);

  // Init renderer
  try {
    if (useWebGPU) {
      dbg('Init WebGPU renderer...');
      renderer = new WebGPURenderer(canvas);
      await renderer.init();
      ui.setRenderer('WebGPU');
      dbg('WebGPU OK');
    } else {
      throw new Error('Use WebGL2');
    }
  } catch (e) {
    dbg('Trying WebGL2: ' + e.message);
    useWebGPU = false;
    try {
      renderer = new WebGLFallback(canvas);
      await renderer.init();
      dbg('GL ctx: ' + (renderer.gl ? 'OK' : 'NULL'));
      dbg('MaxPtSize: ' + renderer.maxPointSize);
      dbg('FBO: ' + renderer.fboWidth + 'x' + renderer.fboHeight);
      ui.setRenderer('WebGL2');
      dbg('WebGL2 init OK');
    } catch (e2) {
      dbg('WebGL2 FAIL: ' + e2.message + ' | ' + e2.stack?.split('\n')[1]);
      throw e2;
    }
  }

  ui.hideLoading();
  dbg('Loading hidden, starting RAF');

  // Handle WebGL context loss (common on iOS Safari)
  canvas.addEventListener('webglcontextlost', (e) => {
    e.preventDefault();
    dbg('CONTEXT LOST');
  });
  canvas.addEventListener('webglcontextrestored', () => {
    dbg('CONTEXT RESTORED');
    if (!useWebGPU && renderer) {
      renderer.init().catch(err => dbg('Reinit fail: ' + err.message));
    }
  });

  // Events
  window.addEventListener('resize', resize);

  canvas.addEventListener('click', () => {
    if (stateMachine.canClick) {
      stateMachine.click();
      dbg('Click -> ' + stateMachine.state);
    }
  });

  // Touch support (passive: false required for iOS preventDefault)
  canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    if (stateMachine.canClick) {
      stateMachine.click();
      dbg('Tap -> ' + stateMachine.state);
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
  let frameCount = 0;

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
    try {
      renderer.frame(dt, stateMachine.tOrder, totalTime, camera);
    } catch (e) {
      if (frameCount < 3) dbg('Frame err: ' + e.message);
    }

    frameCount++;
    if (frameCount === 5) dbg('Rendering (5 frames OK)');

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
  dbg('FATAL: ' + err.message);
  const loading = document.getElementById('loading-text');
  if (loading) {
    loading.textContent = 'Error: ' + err.message;
    loading.style.color = '#f97316';
  }
});

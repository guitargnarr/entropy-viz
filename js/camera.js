/**
 * Orbital camera with slow drift + mouse parallax.
 * Outputs eye position + view matrix.
 */

import { mat4LookAt } from './math-utils.js';

export class Camera {
  constructor() {
    this.baseRadius = 8.0;
    this.radius = 8.0;
    this.theta = 0;           // horizontal angle
    this.phi = 0.3;           // vertical angle (slight elevation)
    this.driftSpeed = 0.02;   // rad/s orbital drift
    this.center = [0, 0, 0];
    this.up = [0, 1, 0];

    // Mouse parallax
    this.mouseX = 0;          // -1..1 normalized
    this.mouseY = 0;
    this.maxParallax = 0.08;  // ~5 degrees max deflection in radians

    this.eye = [0, 0, this.radius];
    this.viewMatrix = new Float32Array(16);

    // Adapt for portrait aspect ratios (phones)
    this.updateAspect();
    window.addEventListener('resize', () => this.updateAspect());
  }

  updateAspect() {
    const aspect = window.innerWidth / window.innerHeight;
    // Portrait (< 1.0): pull camera back so lattice fits
    // The narrower the viewport, the farther back we go
    this.radius = aspect < 1.0
      ? this.baseRadius * (1.0 + (1.0 - aspect) * 0.8)
      : this.baseRadius;
  }

  onMouseMove(nx, ny) {
    this.mouseX = nx;  // -1..1
    this.mouseY = ny;
  }

  update(dt) {
    this.theta += this.driftSpeed * dt;

    // Mouse parallax offsets
    const thetaOffset = this.mouseX * this.maxParallax;
    const phiOffset = this.mouseY * this.maxParallax;

    const t = this.theta + thetaOffset;
    const p = this.phi + phiOffset;

    this.eye[0] = this.radius * Math.sin(t) * Math.cos(p);
    this.eye[1] = this.radius * Math.sin(p);
    this.eye[2] = this.radius * Math.cos(t) * Math.cos(p);

    this.viewMatrix = mat4LookAt(this.eye, this.center, this.up);
  }
}
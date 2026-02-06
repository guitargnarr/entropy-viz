// Particle physics compute shader.
// Spring forces pull toward crystal lattice positions.
// Brownian noise kicks particles into chaos.
// t_order blends between the two regimes.

struct Particle {
  position: vec3<f32>,
  _pad0: f32,
  velocity: vec3<f32>,
  speed: f32,
  home_position: vec3<f32>,
  _pad1: f32,
};

struct Uniforms {
  dt: f32,
  t_order: f32,
  damping_ordered: f32,
  damping_chaos: f32,
  noise_strength: f32,
  spring_k: f32,
  time: f32,
  particle_count: u32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

// PCG hash for deterministic randomness per particle per frame
fn pcg(v: u32) -> u32 {
  var state = v * 747796405u + 2891336453u;
  var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand01(seed: u32) -> f32 {
  return f32(pcg(seed)) / 4294967295.0;
}

fn randNormal(seed1: u32, seed2: u32) -> f32 {
  // Box-Muller approximation via uniform pair
  let u1 = max(rand01(seed1), 0.0001);
  let u2 = rand01(seed2);
  return sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= uniforms.particle_count) { return; }

  var p = particles[idx];
  let dt = uniforms.dt;
  let t = uniforms.t_order;

  // ── Spring force toward home (scales with t_order) ──
  let displacement = p.home_position - p.position;
  let spring_force = displacement * uniforms.spring_k * t;

  // ── Brownian noise (scales with 1 - t_order) ──
  let frame_seed = bitcast<u32>(uniforms.time * 1000.0) + idx * 3u;
  let nx = randNormal(frame_seed, frame_seed + 1u) * uniforms.noise_strength * (1.0 - t);
  let ny = randNormal(frame_seed + 2u, frame_seed + 3u) * uniforms.noise_strength * (1.0 - t);
  let nz = randNormal(frame_seed + 4u, frame_seed + 5u) * uniforms.noise_strength * (1.0 - t);
  let noise_force = vec3<f32>(nx, ny, nz);

  // ── Damping (more damping when ordered) ──
  let damping = mix(uniforms.damping_chaos, uniforms.damping_ordered, t);

  // ── Integrate ──
  p.velocity = p.velocity * damping + (spring_force + noise_force) * dt;

  // Clamp velocity to prevent explosion
  let max_speed = 8.0;
  let spd = length(p.velocity);
  if (spd > max_speed) {
    p.velocity = p.velocity * (max_speed / spd);
  }

  p.position = p.position + p.velocity * dt;

  // Soft boundary: push back if too far from origin
  let dist_from_center = length(p.position);
  if (dist_from_center > 4.5) {
    let push = normalize(p.position) * (dist_from_center - 4.5) * 2.0;
    p.velocity = p.velocity - push * dt;
  }

  // Cache speed for histogram
  p.speed = length(p.velocity);

  particles[idx] = p;
}
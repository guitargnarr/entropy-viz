// Bin particle speeds into 64 histogram buckets for Shannon entropy.
// Histogram must be cleared before this dispatch (done by histogram-clear pass).

struct Particle {
  position: vec3<f32>,
  _pad0: f32,
  velocity: vec3<f32>,
  speed: f32,
  home_position: vec3<f32>,
  _pad1: f32,
};

struct HistUniforms {
  particle_count: u32,
  num_bins: u32,
  max_speed: f32,
  _pad: f32,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> uniforms: HistUniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= uniforms.particle_count) { return; }

  let speed = particles[idx].speed;
  let normalized = clamp(speed / uniforms.max_speed, 0.0, 0.9999);
  let bin = u32(normalized * f32(uniforms.num_bins));
  atomicAdd(&histogram[bin], 1u);
}
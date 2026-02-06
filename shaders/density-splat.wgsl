// Each particle writes a Gaussian contribution to the 64^3 density volume.
// Uses atomicAdd on u32 (fixed-point: value * 1000).

struct Particle {
  position: vec3<f32>,
  _pad0: f32,
  velocity: vec3<f32>,
  speed: f32,
  home_position: vec3<f32>,
  _pad1: f32,
};

struct DensityUniforms {
  grid_min: vec3<f32>,
  _pad0: f32,
  grid_max: vec3<f32>,
  _pad1: f32,
  grid_res: u32,        // 64
  particle_count: u32,
  splat_radius: f32,    // in grid cells (e.g., 2.0)
  splat_strength: f32,  // amplitude
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> density: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> uniforms: DensityUniforms;

fn worldToGrid(pos: vec3<f32>) -> vec3<f32> {
  let normalized = (pos - uniforms.grid_min) / (uniforms.grid_max - uniforms.grid_min);
  return normalized * f32(uniforms.grid_res);
}

fn gridIndex(ix: u32, iy: u32, iz: u32) -> u32 {
  return ix + iy * uniforms.grid_res + iz * uniforms.grid_res * uniforms.grid_res;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= uniforms.particle_count) { return; }

  let p = particles[idx];
  let gridPos = worldToGrid(p.position);

  // Splat radius in grid cells
  let r = i32(ceil(uniforms.splat_radius));
  let center = vec3<i32>(vec3<f32>(floor(gridPos.x), floor(gridPos.y), floor(gridPos.z)));
  let res = i32(uniforms.grid_res);

  // Velocity-dependent intensity: faster particles glow brighter
  let speed_factor = 1.0 + p.speed * 0.3;

  for (var dz = -r; dz <= r; dz++) {
    for (var dy = -r; dy <= r; dy++) {
      for (var dx = -r; dx <= r; dx++) {
        let cell = center + vec3<i32>(dx, dy, dz);

        // Bounds check
        if (cell.x < 0 || cell.x >= res ||
            cell.y < 0 || cell.y >= res ||
            cell.z < 0 || cell.z >= res) {
          continue;
        }

        // Gaussian falloff
        let cellCenter = vec3<f32>(f32(cell.x) + 0.5, f32(cell.y) + 0.5, f32(cell.z) + 0.5);
        let dist = length(gridPos - cellCenter);
        let sigma = uniforms.splat_radius * 0.5;
        let weight = exp(-dist * dist / (2.0 * sigma * sigma));

        // Fixed-point: multiply by 1000 for u32 precision
        let contribution = u32(weight * uniforms.splat_strength * speed_factor * 1000.0);
        if (contribution > 0u) {
          let gi = gridIndex(u32(cell.x), u32(cell.y), u32(cell.z));
          atomicAdd(&density[gi], contribution);
        }
      }
    }
  }
}
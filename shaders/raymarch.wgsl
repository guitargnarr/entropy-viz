// Volumetric raymarcher: marches through 64^3 density field.
// Beer-Lambert absorption, blackbody color from LUT, front-to-back compositing.

struct RaymarchUniforms {
  inv_view: mat4x4<f32>,
  inv_proj: mat4x4<f32>,
  camera_pos: vec3<f32>,
  time: f32,
  grid_min: vec3<f32>,
  _pad0: f32,
  grid_max: vec3<f32>,
  _pad1: f32,
  grid_res: f32,
  screen_width: f32,
  screen_height: f32,
  absorption: f32,
};

@group(0) @binding(0) var<storage, read> density: array<u32>;
@group(0) @binding(1) var<uniform> uniforms: RaymarchUniforms;
@group(0) @binding(2) var color_lut: texture_1d<f32>;
@group(0) @binding(3) var lut_sampler: sampler;

fn sampleDensity(pos: vec3<f32>) -> f32 {
  let norm = (pos - uniforms.grid_min) / (uniforms.grid_max - uniforms.grid_min);
  let gf = norm * uniforms.grid_res;
  let gi = vec3<i32>(floor(gf));
  let frac = gf - floor(gf);

  let res = i32(uniforms.grid_res);

  // Trilinear interpolation
  var result = 0.0;
  for (var dz = 0; dz <= 1; dz++) {
    for (var dy = 0; dy <= 1; dy++) {
      for (var dx = 0; dx <= 1; dx++) {
        let cx = clamp(gi.x + dx, 0, res - 1);
        let cy = clamp(gi.y + dy, 0, res - 1);
        let cz = clamp(gi.z + dz, 0, res - 1);
        let idx = u32(cx + cy * res + cz * res * res);

        // Convert from fixed-point u32 (value * 1000) back to float
        let val = f32(density[idx]) / 1000.0;

        let wx = select(1.0 - frac.x, frac.x, dx == 1);
        let wy = select(1.0 - frac.y, frac.y, dy == 1);
        let wz = select(1.0 - frac.z, frac.z, dz == 1);
        result += val * wx * wy * wz;
      }
    }
  }
  return result;
}

// Ray-AABB intersection
fn intersectBox(ro: vec3<f32>, rd: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
  let invRd = 1.0 / rd;
  let t0 = (bmin - ro) * invRd;
  let t1 = (bmax - ro) * invRd;
  let tmin = min(t0, t1);
  let tmax = max(t0, t1);
  let tNear = max(max(tmin.x, tmin.y), tmin.z);
  let tFar = min(min(tmax.x, tmax.y), tmax.z);
  return vec2<f32>(tNear, tFar);
}

struct FSIn {
  @location(0) uv: vec2<f32>,
};

@fragment
fn fs_main(in: FSIn) -> @location(0) vec4<f32> {
  // Reconstruct ray from screen UV
  let ndc = vec2<f32>(in.uv.x * 2.0 - 1.0, (1.0 - in.uv.y) * 2.0 - 1.0);
  let clip = vec4<f32>(ndc, -1.0, 1.0);
  var eye = uniforms.inv_proj * clip;
  eye = vec4<f32>(eye.xy, -1.0, 0.0);
  let world_dir = normalize((uniforms.inv_view * eye).xyz);

  let ro = uniforms.camera_pos;
  let rd = world_dir;

  // Intersect with density volume AABB (slightly expanded for margin)
  let margin = vec3<f32>(0.1);
  let hit = intersectBox(ro, rd, uniforms.grid_min - margin, uniforms.grid_max + margin);

  if (hit.x > hit.y || hit.y < 0.0) {
    // Background: subtle radial gradient
    let dist = length(in.uv - vec2<f32>(0.5));
    let bg = mix(vec3<f32>(0.012, 0.012, 0.025), vec3<f32>(0.008, 0.008, 0.015), dist * 1.5);
    return vec4<f32>(bg, 1.0);
  }

  let tStart = max(hit.x, 0.0);
  let tEnd = hit.y;
  let stepSize = (tEnd - tStart) / 96.0;  // Adaptive: 96 steps through volume

  // Front-to-back compositing
  var accumulated_color = vec3<f32>(0.0);
  var accumulated_alpha = 0.0;

  for (var i = 0u; i < 96u; i++) {
    if (accumulated_alpha > 0.97) { break; }

    let t = tStart + (f32(i) + 0.5) * stepSize;
    let pos = ro + rd * t;

    // Sample density field
    let d = sampleDensity(pos);

    if (d > 0.01) {
      // Beer-Lambert absorption
      let alpha = 1.0 - exp(-d * uniforms.absorption * stepSize);

      // Sample color from LUT based on density (normalized)
      let color_t = clamp(d * 0.15, 0.0, 1.0);
      let lut_color = textureSampleLevel(color_lut, lut_sampler, color_t, 0.0).rgb;

      // Emission: denser regions glow brighter
      let emission = lut_color * (1.0 + d * 0.1);

      // Front-to-back composite
      accumulated_color += emission * alpha * (1.0 - accumulated_alpha);
      accumulated_alpha += alpha * (1.0 - accumulated_alpha);
    }
  }

  // Background blend
  let dist = length(in.uv - vec2<f32>(0.5));
  let bg = mix(vec3<f32>(0.012, 0.012, 0.025), vec3<f32>(0.008, 0.008, 0.015), dist * 1.5);
  let final_color = accumulated_color + bg * (1.0 - accumulated_alpha);

  // Subtle vignette
  let vignette = 1.0 - dist * 0.4;

  return vec4<f32>(final_color * vignette, 1.0);
}
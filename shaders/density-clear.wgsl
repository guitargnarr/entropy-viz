// Zero the 64^3 density volume before splatting.

@group(0) @binding(0) var<storage, read_write> density: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= 262144u) { return; } // 64^3
  atomicStore(&density[idx], 0u);
}
// Zero the 64-bin histogram buffer before binning particles.

@group(0) @binding(0) var<storage, read_write> histogram: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= 64u) { return; }
  atomicStore(&histogram[idx], 0u);
}
/**
 * Generate a blackbody-inspired color ramp for the density field.
 * Maps speed/temperature to: deep blue -> teal -> amber -> white-hot
 * Returns a 256x1 RGBA8 Uint8Array for a 1D texture.
 */

export function generateColorLUT(size = 256) {
  const data = new Uint8Array(size * 4);

  // Control points: [position, r, g, b]
  const stops = [
    [0.00, 0.02, 0.04, 0.12],   // near-black blue
    [0.15, 0.05, 0.15, 0.35],   // deep blue-teal
    [0.30, 0.08, 0.72, 0.65],   // teal (#14b8a6 range)
    [0.50, 0.20, 0.85, 0.75],   // bright teal
    [0.65, 0.85, 0.60, 0.20],   // amber transition
    [0.80, 0.98, 0.45, 0.09],   // orange (#f97316 range)
    [0.92, 1.00, 0.80, 0.50],   // warm white
    [1.00, 1.00, 0.96, 0.90],   // white-hot
  ];

  for (let i = 0; i < size; i++) {
    const t = i / (size - 1);

    // Find surrounding stops
    let lower = stops[0];
    let upper = stops[stops.length - 1];
    for (let s = 0; s < stops.length - 1; s++) {
      if (t >= stops[s][0] && t <= stops[s + 1][0]) {
        lower = stops[s];
        upper = stops[s + 1];
        break;
      }
    }

    // Interpolate
    const range = upper[0] - lower[0];
    const f = range > 0 ? (t - lower[0]) / range : 0;
    // Smooth step for nicer gradients
    const sf = f * f * (3 - 2 * f);

    const r = lower[1] + (upper[1] - lower[1]) * sf;
    const g = lower[2] + (upper[2] - lower[2]) * sf;
    const b = lower[3] + (upper[3] - lower[3]) * sf;

    data[i * 4 + 0] = Math.round(r * 255);
    data[i * 4 + 1] = Math.round(g * 255);
    data[i * 4 + 2] = Math.round(b * 255);
    data[i * 4 + 3] = 255;
  }

  return data;
}
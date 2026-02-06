/**
 * Shannon entropy from a histogram of particle speeds.
 * H = -sum(p_k * log2(p_k)) for non-zero bins.
 * Normalized to [0, 1] by dividing by log2(numBins).
 */

export class EntropyCalculator {
  constructor(numBins = 64) {
    this.numBins = numBins;
    this.maxEntropy = Math.log2(numBins);
    this.currentEntropy = 0;
    this.normalizedEntropy = 0;
    // Smoothed value for display (avoids jitter)
    this.displayEntropy = 0;
    this.smoothing = 0.12;
  }

  /**
   * Compute Shannon entropy from a Uint32Array histogram.
   * @param {Uint32Array} histogram - bin counts
   * @returns {{ entropy: number, normalized: number }}
   */
  compute(histogram) {
    let total = 0;
    for (let i = 0; i < histogram.length; i++) {
      total += histogram[i];
    }
    if (total === 0) {
      this.currentEntropy = 0;
      this.normalizedEntropy = 0;
      return { entropy: 0, normalized: 0 };
    }

    let H = 0;
    for (let i = 0; i < histogram.length; i++) {
      if (histogram[i] > 0) {
        const p = histogram[i] / total;
        H -= p * Math.log2(p);
      }
    }

    this.currentEntropy = H;
    this.normalizedEntropy = H / this.maxEntropy;
    return { entropy: H, normalized: this.normalizedEntropy };
  }

  /**
   * Smooth the display value to avoid jitter.
   */
  updateDisplay(dt) {
    this.displayEntropy += (this.normalizedEntropy - this.displayEntropy) * this.smoothing;
  }
}
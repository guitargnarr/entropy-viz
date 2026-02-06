/**
 * UI overlay manager — updates DOM elements from simulation state.
 */

export class UI {
  constructor() {
    this.entropyValue = document.getElementById('entropy-value');
    this.entropyBarFill = document.getElementById('entropy-bar-fill');
    this.stateLabel = document.getElementById('state-label');
    this.clickInstruction = document.getElementById('click-instruction');
    this.rendererBadge = document.getElementById('renderer-badge');
    this.loading = document.getElementById('loading');
  }

  setRenderer(name) {
    this.rendererBadge.textContent = name;
  }

  hideLoading() {
    this.loading.classList.add('fade-out');
    setTimeout(() => {
      this.loading.style.display = 'none';
    }, 600);
  }

  update(entropy, stateMachine) {
    // Entropy counter
    const displayBits = entropy.currentEntropy;
    this.entropyValue.textContent = displayBits.toFixed(4);

    // Color: hot when entropy is high
    if (entropy.normalizedEntropy > 0.5) {
      this.entropyValue.classList.add('hot');
    } else {
      this.entropyValue.classList.remove('hot');
    }

    // Entropy bar
    this.entropyBarFill.style.width = `${entropy.displayEntropy * 100}%`;

    // State label — visible during transitions
    this.stateLabel.textContent = stateMachine.stateLabel;
    if (stateMachine.isTransitioning) {
      this.stateLabel.classList.add('visible');
    } else {
      this.stateLabel.classList.remove('visible');
    }

    // Click instruction
    const instruction = stateMachine.clickInstruction;
    if (instruction) {
      this.clickInstruction.textContent = instruction;
      this.clickInstruction.classList.remove('hidden');
    } else {
      this.clickInstruction.classList.add('hidden');
    }
  }
}
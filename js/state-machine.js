/**
 * State machine: ORDERED <-> CHAOS with transition states.
 * t_order smoothly interpolates [0,1] to blend physics parameters.
 */

export const State = {
  ORDERED:       'ORDERED',
  SHATTERING:    'SHATTERING',
  CHAOS:         'CHAOS',
  REASSEMBLING:  'REASSEMBLING',
};

// Easing functions
function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }
function easeInOutQuad(t) { return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; }

export class StateMachine {
  constructor() {
    this.state = State.ORDERED;
    this.tOrder = 1.0;           // 1.0 = fully ordered, 0.0 = full chaos
    this.transitionProgress = 0; // 0..1 within current transition
    this.transitionDuration = 0; // seconds
    this.elapsed = 0;
  }

  get canClick() {
    return this.state === State.ORDERED || this.state === State.CHAOS;
  }

  click() {
    if (this.state === State.ORDERED) {
      this.state = State.SHATTERING;
      this.transitionDuration = 1.5;
      this.transitionProgress = 0;
      this.elapsed = 0;
    } else if (this.state === State.CHAOS) {
      this.state = State.REASSEMBLING;
      this.transitionDuration = 2.0;
      this.transitionProgress = 0;
      this.elapsed = 0;
    }
  }

  update(dt) {
    if (this.state === State.SHATTERING) {
      this.elapsed += dt;
      this.transitionProgress = Math.min(this.elapsed / this.transitionDuration, 1.0);
      // Ease-out cubic: explosive start, gentle finish
      this.tOrder = 1.0 - easeOutCubic(this.transitionProgress);
      if (this.transitionProgress >= 1.0) {
        this.state = State.CHAOS;
        this.tOrder = 0.0;
      }
    } else if (this.state === State.REASSEMBLING) {
      this.elapsed += dt;
      this.transitionProgress = Math.min(this.elapsed / this.transitionDuration, 1.0);
      // Ease-in-out quad: graceful reassembly
      this.tOrder = easeInOutQuad(this.transitionProgress);
      if (this.transitionProgress >= 1.0) {
        this.state = State.ORDERED;
        this.tOrder = 1.0;
      }
    }
  }

  get stateLabel() {
    switch (this.state) {
      case State.ORDERED:      return 'ORDER';
      case State.SHATTERING:   return 'SHATTERING';
      case State.CHAOS:        return 'CHAOS';
      case State.REASSEMBLING: return 'REASSEMBLING';
    }
  }

  get clickInstruction() {
    if (this.state === State.ORDERED)  return 'click to shatter';
    if (this.state === State.CHAOS)    return 'click to reassemble';
    return '';
  }

  get isTransitioning() {
    return this.state === State.SHATTERING || this.state === State.REASSEMBLING;
  }
}
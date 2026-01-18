/**
 * Spring Animation Configurations
 * Based on realfood.gov technical analysis
 */

export const springs = {
  // Spring A - Overshoot spring for dramatic entrances (from reference)
  // Used for: Food title transitions, pyramid item entrances
  // Duration: ~700ms, Peak overshoot: 6.64%
  springA: {
    type: 'spring' as const,
    stiffness: 150,
    damping: 16,
    mass: 1,
  },

  // Spring B - Smooth deceleration for UI elements (from reference)
  // Used for: Navigation dots, UI transitions
  // Duration: ~650ms, No overshoot
  springB: {
    type: 'spring' as const,
    stiffness: 120,
    damping: 20,
    mass: 1,
  },

  // Card hover spring (from reference)
  hover: {
    type: 'spring' as const,
    stiffness: 400,
    damping: 25,
    mass: 1,
  },

  // Pyramid items rearrange (from reference)
  pyramidItem: {
    type: 'spring' as const,
    stiffness: 300,
    damping: 30,
    mass: 0.5,
  },

  // Gentle spring for smooth, natural motion
  gentle: {
    type: 'spring' as const,
    stiffness: 100,
    damping: 20,
    mass: 1,
  },

  // Bouncy spring for playful interactions
  bouncy: {
    type: 'spring' as const,
    stiffness: 300,
    damping: 20,
    mass: 1,
  },

  // Snappy spring for quick, responsive feedback
  snappy: {
    type: 'spring' as const,
    stiffness: 400,
    damping: 30,
    mass: 1,
  },

  // Smooth spring for elegant transitions
  smooth: {
    type: 'spring' as const,
    stiffness: 200,
    damping: 25,
    mass: 1,
  },

  // Slow spring for dramatic reveals
  slow: {
    type: 'spring' as const,
    stiffness: 80,
    damping: 20,
    mass: 1.5,
  },
} as const;

/**
 * Easing functions for non-spring animations
 */
export const easings = {
  easeIn: [0.4, 0, 1, 1] as const,
  easeOut: [0, 0, 0.2, 1] as const,
  easeInOut: [0.4, 0, 0.2, 1] as const,
  linear: [0, 0, 1, 1] as const,
} as const;

/**
 * Duration presets (in seconds)
 */
export const durations = {
  fast: 0.15,
  base: 0.2,
  slow: 0.3,
  slower: 0.5,
  slowest: 0.7,
} as const;


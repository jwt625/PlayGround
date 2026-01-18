'use client';

import { motion, useTransform, MotionValue } from 'framer-motion';
import styles from './PyramidSVG.module.css';

interface PyramidSVGProps {
  scrollYProgress: MotionValue<number>;
}

export function PyramidSVG({ scrollYProgress }: PyramidSVGProps) {
  // Stroke animation: draws from 0.05 to 0.95 of scroll
  const pathLength = useTransform(scrollYProgress, [0.05, 0.95], [0, 1]);
  const opacity = useTransform(scrollYProgress, [0.05, 0.1], [0, 0.5]);

  return (
    <svg
      className={styles.svg}
      viewBox="0 0 100 100"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* Inverted pyramid outline - correct path from spec */}
      <motion.path
        d="M 5 5 L 95 5 L 50 95 Z"
        stroke="#E5E0D6"
        strokeWidth="2"
        fill="none"
        style={{
          pathLength,
          opacity,
        }}
      />
    </svg>
  );
}


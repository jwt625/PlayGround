'use client';

import { useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import styles from './DisintegratingText.module.css';

interface DisintegratingTextProps {
  text: string;
  className?: string;
}

export function DisintegratingText({ text, className = '' }: DisintegratingTextProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start 0.8', 'start 0.3'],
  });

  const chars = text.split('');

  return (
    <div ref={containerRef} className={`${styles.container} ${className}`}>
      {chars.map((char, index) => {
        const totalChars = chars.length;
        // Reference formula: scrollProgress * (1 + charIndex / totalChars * 0.12)
        const charFactor = 1 + (index / totalChars) * 0.12;
        const delay = (index / totalChars) * 0.3;

        const opacity = useTransform(
          scrollYProgress,
          [delay, delay + 0.3 * charFactor],
          [0, 1]
        );

        const y = useTransform(
          scrollYProgress,
          [delay, delay + 0.3 * charFactor],
          [30, 0]
        );

        const blur = useTransform(
          scrollYProgress,
          [delay, delay + 0.3 * charFactor],
          [10, 0]
        );

        const brightness = useTransform(
          scrollYProgress,
          [delay, delay + 0.3 * charFactor],
          [0.5, 1]
        );

        return (
          <motion.span
            key={index}
            className={styles.char}
            style={{
              opacity,
              y,
              filter: `blur(${blur}px) brightness(${brightness})`,
              display: char === ' ' ? 'inline' : 'inline-block',
            }}
          >
            {char === ' ' ? '\u00A0' : char}
          </motion.span>
        );
      })}
    </div>
  );
}


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
        const progress = index / chars.length;
        const delay = progress * 0.5;
        
        const opacity = useTransform(
          scrollYProgress,
          [delay, delay + 0.2],
          [0, 1]
        );
        
        const y = useTransform(
          scrollYProgress,
          [delay, delay + 0.2],
          [20, 0]
        );
        
        const rotateX = useTransform(
          scrollYProgress,
          [delay, delay + 0.2],
          [90, 0]
        );

        return (
          <motion.span
            key={index}
            className={styles.char}
            style={{
              opacity,
              y,
              rotateX,
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


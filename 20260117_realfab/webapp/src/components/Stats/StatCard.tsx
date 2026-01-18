'use client';

import { useRef, useEffect, useState } from 'react';
import { motion, useTransform, MotionValue, useSpring } from 'framer-motion';
import { useInView } from '@/hooks/useInView';
import styles from './StatCard.module.css';

interface StatCardProps {
  value: string;
  description: string;
  color?: string;
  index: number;
  scrollYProgress: MotionValue<number>;
}

// Extract numeric value for counter animation
function extractNumber(value: string): number | null {
  const match = value.match(/[\d,]+/);
  if (!match) return null;
  return parseInt(match[0].replace(/,/g, ''), 10);
}

export function StatCard({ value, description, color = '#E53935', index, scrollYProgress }: StatCardProps) {
  const cardRef = useRef<HTMLDivElement>(null);
  const isInView = useInView(cardRef, { threshold: 0.5, triggerOnce: true });
  const [displayValue, setDisplayValue] = useState(value);

  // Scroll-based card position
  const cardStart = index * 0.15;
  const cardEnd = cardStart + 0.3;
  
  const y = useTransform(
    scrollYProgress,
    [cardStart, cardEnd],
    [100, 0]
  );

  const opacity = useTransform(
    scrollYProgress,
    [cardStart, cardStart + 0.1],
    [0, 1]
  );

  // Counter animation
  const numericValue = extractNumber(value);
  const springValue = useSpring(0, { stiffness: 50, damping: 30 });

  useEffect(() => {
    if (isInView && numericValue !== null) {
      springValue.set(numericValue);
    }
  }, [isInView, numericValue, springValue]);

  useEffect(() => {
    if (numericValue === null) return;

    const unsubscribe = springValue.on('change', (latest) => {
      const rounded = Math.round(latest);
      const prefix = value.match(/^\$/)?.[0] || '';
      const suffix = value.match(/[+%]$/)?.[0] || '';
      
      setDisplayValue(`${prefix}${rounded.toLocaleString()}${suffix}`);
    });

    return unsubscribe;
  }, [springValue, value, numericValue]);

  return (
    <motion.div
      ref={cardRef}
      className={styles.card}
      style={{ 
        y,
        opacity,
        backgroundColor: color,
      }}
      whileHover={{ 
        scale: 1.02,
        filter: 'saturate(1.2)',
      }}
      transition={{ type: 'spring', stiffness: 300, damping: 25 }}
    >
      <div className={styles.value}>{displayValue}</div>
      <div className={styles.description}>{description}</div>
    </motion.div>
  );
}


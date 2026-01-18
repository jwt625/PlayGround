'use client';

import { useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { useInView } from '@/hooks/useInView';
import { StatCard } from './StatCard';
import styles from './Stats.module.css';

interface Stat {
  value: string;
  description: string;
  color?: string;
}

const stats: Stat[] = [
  {
    value: '3',
    description: 'companies control 90%+ of advanced chip manufacturing',
    color: '#E53935',
  },
  {
    value: '500+',
    description: 'process steps in modern SOTA semiconductor fabrication',
    color: '#FB8C00',
  },
  {
    value: '99%',
    description: 'of high-purity materials wasted in subtractive processes',
    color: '#FDD835',
  },
  {
    value: '$20B+',
    description: 'cost to build a single leading-edge fab',
    color: '#43A047',
  },
];

export function Stats() {
  const containerRef = useRef<HTMLDivElement>(null);
  const textRef = useRef<HTMLDivElement>(null);
  const isInView = useInView(textRef, { threshold: 0.3 });

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start end', 'end start'],
  });

  return (
    <div ref={containerRef} className={styles.container}>
      <div className={styles.content}>
        {/* Sticky text column */}
        <div className={styles.textColumn}>
          <div ref={textRef} className={styles.stickyText}>
            <motion.h2
              className={styles.heading}
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
              transition={{ duration: 0.6 }}
            >
              Our industry is broken.
            </motion.h2>
            <motion.p
              className={styles.subheading}
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
              transition={{ duration: 0.6, delay: 0.1 }}
            >
              The data is clear.
            </motion.p>
            <motion.p
              className={styles.description}
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              For decades, we've been told that bigger is better - that only
              trillion-dollar fabs can produce modern electronics. This has led to
              unprecedented concentration of manufacturing power and fragile
              global supply chains.
            </motion.p>
          </div>
        </div>

        {/* Scrolling cards column */}
        <div className={styles.cardsColumn}>
          {stats.map((stat, index) => (
            <StatCard
              key={index}
              value={stat.value}
              description={stat.description}
              color={stat.color}
              index={index}
              scrollYProgress={scrollYProgress}
            />
          ))}
        </div>
      </div>
    </div>
  );
}


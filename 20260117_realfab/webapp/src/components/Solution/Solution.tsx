'use client';

import { useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { DisintegratingText } from '@/components/Problem/DisintegratingText';
import styles from './Solution.module.css';

export function Solution() {
  const containerRef = useRef<HTMLDivElement>(null);

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start end', 'end start'],
  });

  const blur = useTransform(scrollYProgress, [0, 0.3, 0.7, 1], [10, 0, 0, 10]);
  const opacity = useTransform(scrollYProgress, [0, 0.3, 0.7, 1], [0, 1, 1, 0]);

  return (
    <div ref={containerRef} className={styles.container}>
      <div className={styles.content}>
        <motion.div
          className={styles.textContent}
          style={{
            filter: blur.get() > 0 ? `blur(${blur.get()}px)` : 'none',
            opacity,
          }}
        >
          <DisintegratingText
            text="We can change this paradigm."
            className={styles.heading}
          />
          
          <motion.p
            className={styles.description}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-100px' }}
            transition={{ duration: 0.8, delay: 0.3 }}
          >
            For the first time, additive manufacturing technologies are enabling
            distributed, local fabrication. We're rebuilding from the ground up
            with minimal waste, maximum accessibility, and true resilience.
          </motion.p>

          <motion.p
            className={styles.description}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-100px' }}
            transition={{ duration: 0.8, delay: 0.5 }}
          >
            Desktop tools replace billion-dollar facilities. Open-source hardware
            replaces proprietary black boxes. Community makerspaces replace
            centralized mega-fabs.
          </motion.p>

          <motion.p
            className={styles.description}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-100px' }}
            transition={{ duration: 0.8, delay: 0.7 }}
          >
            This is the future of fabrication.
          </motion.p>
        </motion.div>
      </div>
    </div>
  );
}


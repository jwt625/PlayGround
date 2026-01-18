'use client';

import { useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { DisintegratingText } from './DisintegratingText';
import styles from './Problem.module.css';

export function Problem() {
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
            text="The old system is failing."
            className={styles.heading}
          />
          
          <motion.p
            className={styles.description}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-100px' }}
            transition={{ duration: 0.8, delay: 0.3 }}
          >
            Centralized mega-fabs have created a fragile, monopolistic system.
            Three companies control the future of computing. A single fab costs
            more than most countries' GDP. Supply chains span the globe, vulnerable
            to disruption.
          </motion.p>

          <motion.p
            className={styles.description}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-100px' }}
            transition={{ duration: 0.8, delay: 0.5 }}
          >
            This isn't innovation. It's consolidation. And it's holding us back.
          </motion.p>
        </motion.div>
      </div>
    </div>
  );
}


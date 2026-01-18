'use client';

import { useRef } from 'react';
import { motion, useScroll, useTransform, useSpring } from 'framer-motion';
import styles from './Problem.module.css';

// Statistics data for the 5 states
const STATS = [
  {
    id: 1,
    heading: 'Our industry is broken.',
    subheading: 'The data is clear.',
    highlight: null,
    number: null,
    blockSize: null,
  },
  {
    id: 2,
    heading: '3 companies control 90%+ of ',
    highlight: 'advanced chip manufacturing',
    number: 3,
    suffix: '',
    blockSize: { width: 400, height: 400 },
    color: '#D62718',
  },
  {
    id: 3,
    heading: '500+ process steps in modern SOTA ',
    highlight: 'semiconductor fabrication',
    number: 500,
    suffix: '+',
    blockSize: { width: 400, height: 600 },
    color: '#E33224',
  },
  {
    id: 4,
    heading: '99% of materials wasted in ',
    highlight: 'subtractive processes',
    number: 99,
    suffix: '%',
    blockSize: { width: 500, height: 700 },
    color: '#C41E1A',
  },
  {
    id: 5,
    heading: '$20B+ cost to build a ',
    highlight: 'single leading-edge fab',
    number: 20,
    suffix: 'B+',
    blockSize: { width: 600, height: 800 },
    color: '#B01810',
  },
];

export function Problem() {
  const containerRef = useRef<HTMLDivElement>(null);

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start start', 'end end'],
  });

  // State 1: 0.0 - 0.2
  const state1Opacity = useTransform(scrollYProgress, [0, 0.15, 0.2], [1, 1, 0]);

  // State 2: 0.2 - 0.4
  const state2Opacity = useTransform(scrollYProgress, [0.15, 0.2, 0.35, 0.4], [0, 1, 1, 0]);
  const block1Y = useTransform(scrollYProgress, [0.2, 0.3], [100, 0]);
  const block1Opacity = useTransform(scrollYProgress, [0.2, 0.28], [0, 1]);

  // State 3: 0.4 - 0.6
  const state3Opacity = useTransform(scrollYProgress, [0.35, 0.4, 0.55, 0.6], [0, 1, 1, 0]);
  const block2Y = useTransform(scrollYProgress, [0.4, 0.5], [100, 0]);
  const block2Opacity = useTransform(scrollYProgress, [0.4, 0.48], [0, 1]);

  // State 4: 0.6 - 0.8
  const state4Opacity = useTransform(scrollYProgress, [0.55, 0.6, 0.75, 0.8], [0, 1, 1, 0]);
  const block3Y = useTransform(scrollYProgress, [0.6, 0.7], [100, 0]);
  const block3Opacity = useTransform(scrollYProgress, [0.6, 0.68], [0, 1]);

  // State 5: 0.8 - 1.0
  const state5Opacity = useTransform(scrollYProgress, [0.75, 0.8, 1], [0, 1, 1]);
  const block4Y = useTransform(scrollYProgress, [0.8, 0.9], [100, 0]);
  const block4Opacity = useTransform(scrollYProgress, [0.8, 0.88], [0, 1]);

  // Number counters with spring physics
  const number1Progress = useTransform(scrollYProgress, [0.2, 0.3], [0, 1]);
  const number1 = useSpring(useTransform(number1Progress, [0, 1], [0, STATS[1].number!]), {
    stiffness: 50,
    damping: 30,
  });

  const number2Progress = useTransform(scrollYProgress, [0.4, 0.5], [0, 1]);
  const number2 = useSpring(useTransform(number2Progress, [0, 1], [0, STATS[2].number!]), {
    stiffness: 50,
    damping: 30,
  });

  const number3Progress = useTransform(scrollYProgress, [0.6, 0.7], [0, 1]);
  const number3 = useSpring(useTransform(number3Progress, [0, 1], [0, STATS[3].number!]), {
    stiffness: 50,
    damping: 30,
  });

  const number4Progress = useTransform(scrollYProgress, [0.8, 0.9], [0, 1]);
  const number4 = useSpring(useTransform(number4Progress, [0, 1], [0, STATS[4].number!]), {
    stiffness: 50,
    damping: 30,
  });

  return (
    <div ref={containerRef} className={styles.scrollContainer}>
      <div className={styles.stickyContainer}>
        {/* Persistent section title */}
        <h2 className={styles.sectionTitle}>The State of Our Fabs</h2>

        {/* Left column - Text content */}
        <div className={styles.leftColumn}>
          {/* State 1 */}
          <motion.div className={styles.stateText} style={{ opacity: state1Opacity }}>
            <h2 className={styles.headingLarge}>
              {STATS[0].heading}
              <br />
              {STATS[0].subheading}
            </h2>
          </motion.div>

          {/* State 2 */}
          <motion.div className={styles.stateText} style={{ opacity: state2Opacity }}>
            <h2 className={styles.headingMedium}>
              {STATS[1].heading}
              <span className={styles.redHighlight}>{STATS[1].highlight}</span>
            </h2>
          </motion.div>

          {/* State 3 */}
          <motion.div className={styles.stateText} style={{ opacity: state3Opacity }}>
            <h2 className={styles.headingMedium}>
              {STATS[2].heading}
              <span className={styles.redHighlight}>{STATS[2].highlight}</span>
            </h2>
          </motion.div>

          {/* State 4 */}
          <motion.div className={styles.stateText} style={{ opacity: state4Opacity }}>
            <h2 className={styles.headingMedium}>
              {STATS[3].heading}
              <span className={styles.redHighlight}>{STATS[3].highlight}</span>
            </h2>
          </motion.div>

          {/* State 5 */}
          <motion.div className={styles.stateText} style={{ opacity: state5Opacity }}>
            <h2 className={styles.headingMedium}>
              {STATS[4].heading}
              <span className={styles.redHighlight}>{STATS[4].highlight}</span>
            </h2>
          </motion.div>
        </div>

        {/* Right column - Graphics */}
        <div className={styles.rightColumn}>
          {/* Block 1 - State 2 (3 companies) */}
          <motion.div
            className={styles.redBlock}
            style={{
              y: block1Y,
              opacity: block1Opacity,
              width: STATS[1].blockSize!.width,
              height: STATS[1].blockSize!.height,
              backgroundColor: STATS[1].color,
              zIndex: 30,
            }}
          >
            <motion.div className={styles.blockNumber}>
              {number1.get().toFixed(0)}
            </motion.div>
          </motion.div>

          {/* Block 2 - State 3 (500+ steps) */}
          <motion.div
            className={styles.redBlock}
            style={{
              y: block2Y,
              opacity: block2Opacity,
              width: STATS[2].blockSize!.width,
              height: STATS[2].blockSize!.height,
              backgroundColor: STATS[2].color,
              zIndex: 20,
            }}
          >
            <motion.div className={styles.blockNumber}>
              {Math.round(number2.get())}{STATS[2].suffix}
            </motion.div>
          </motion.div>

          {/* Block 3 - State 4 (99% waste) */}
          <motion.div
            className={styles.redBlock}
            style={{
              y: block3Y,
              opacity: block3Opacity,
              width: STATS[3].blockSize!.width,
              height: STATS[3].blockSize!.height,
              backgroundColor: STATS[3].color,
              zIndex: 10,
            }}
          >
            <motion.div className={styles.blockNumber}>
              {Math.round(number3.get())}{STATS[3].suffix}
            </motion.div>
          </motion.div>

          {/* Block 4 - State 5 ($20B+ cost) */}
          <motion.div
            className={styles.redBlock}
            style={{
              y: block4Y,
              opacity: block4Opacity,
              width: STATS[4].blockSize!.width,
              height: STATS[4].blockSize!.height,
              backgroundColor: STATS[4].color,
              zIndex: 5,
            }}
          >
            <motion.div className={styles.blockNumber}>
              ${Math.round(number4.get())}{STATS[4].suffix}
            </motion.div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}


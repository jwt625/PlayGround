'use client';

import { useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
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
    heading: '1,000+ process steps in modern SOTA ',
    highlight: 'semiconductor fabrication',
    number: 1000,
    suffix: '+',
    blockSize: { width: 500, height: 570 },
    color: '#E33224',
  },
  {
    id: 4,
    heading: '99% of materials wasted in ',
    highlight: 'subtractive processes',
    number: 99,
    suffix: '%',
    blockSize: { width: 600, height: 740 },
    color: '#C41E1A',
  },
  {
    id: 5,
    heading: '$20B+ cost to build a ',
    highlight: 'single leading-edge fab',
    number: 20,
    suffix: 'B+',
    blockSize: { width: 700, height: 910 },
    color: '#B01810',
  },
];

export function Problem() {
  const containerRef = useRef<HTMLDivElement>(null);

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start start', 'end end'],
  });

  // State 1: 0.0 - 0.2 (fade out at end)
  const state1Opacity = useTransform(scrollYProgress, [0, 0.15, 0.18, 0.2], [1, 1, 0, 0]);
  const state1Y = useTransform(scrollYProgress, [0, 0.15, 0.18, 0.2], [0, 0, -20, -40]);

  // State 2: 0.2 - 0.4 (fade in after state 1 fades out, fade out at end)
  const state2Opacity = useTransform(scrollYProgress, [0.2, 0.22, 0.38, 0.4], [0, 1, 1, 0]);
  const state2Y = useTransform(scrollYProgress, [0.2, 0.22, 0.38, 0.4], [20, 0, 0, -20]);
  const block1Opacity = useTransform(scrollYProgress, [0.2, 0.28], [0, 1]);

  // State 3: 0.4 - 0.6 (fade in after state 2 fades out, fade out at end)
  const state3Opacity = useTransform(scrollYProgress, [0.4, 0.42, 0.58, 0.6], [0, 1, 1, 0]);
  const state3Y = useTransform(scrollYProgress, [0.4, 0.42, 0.58, 0.6], [20, 0, 0, -20]);
  const block2Opacity = useTransform(scrollYProgress, [0.4, 0.48], [0, 1]);

  // State 4: 0.6 - 0.8 (fade in after state 3 fades out, fade out at end)
  const state4Opacity = useTransform(scrollYProgress, [0.6, 0.62, 0.78, 0.8], [0, 1, 1, 0]);
  const state4Y = useTransform(scrollYProgress, [0.6, 0.62, 0.78, 0.8], [20, 0, 0, -20]);
  const block3Opacity = useTransform(scrollYProgress, [0.6, 0.68], [0, 1]);

  // State 5: 0.8 - 0.88 (fade in after state 4 fades out, stays visible with buffer)
  const state5Opacity = useTransform(scrollYProgress, [0.8, 0.82, 0.88, 1], [0, 1, 1, 1]);
  const state5Y = useTransform(scrollYProgress, [0.8, 0.82, 0.88, 1], [20, 0, 0, 0]);
  const block4Opacity = useTransform(scrollYProgress, [0.8, 0.88], [0, 1]);

  // Exit animation: 0.94 - 1.0 (with buffer from 0.88 - 0.94)
  // Text fades out first: 0.94 - 0.95
  const exitTextOpacity = useTransform(scrollYProgress, [0.94, 0.95], [1, 0]);

  // Numbers fade out: 0.95 - 0.96
  const numberOpacity = useTransform(scrollYProgress, [0.95, 0.96], [1, 0]);

  // Blocks expand to fill canvas in sequence (back to front with stagger): 0.96 - 1.0
  // Expansion: width/height grow while bottom-right stays anchored (CSS position: absolute; bottom: 0; right: 0)

  // Block 1 (top layer, z-index 30) - entrance slide, then expand last
  const block1X = useTransform(scrollYProgress, [0.2, 0.3], [100, 0]);
  const block1Y = useTransform(scrollYProgress, [0.2, 0.3], [100, 0]);
  const block1Width = useTransform(scrollYProgress, [0.99, 1.0], [STATS[1].blockSize!.width, 3000]);
  const block1Height = useTransform(scrollYProgress, [0.99, 1.0], [STATS[1].blockSize!.height, 3000]);
  const block1ZIndex = useTransform(scrollYProgress, [0.96, 0.961], [30, 230]);

  // Block 2 (z-index 20) - expand third
  const block2X = useTransform(scrollYProgress, [0.4, 0.5], [100, 0]);
  const block2Y = useTransform(scrollYProgress, [0.4, 0.5], [100, 0]);
  const block2Width = useTransform(scrollYProgress, [0.98, 1.0], [STATS[2].blockSize!.width, 3000]);
  const block2Height = useTransform(scrollYProgress, [0.98, 1.0], [STATS[2].blockSize!.height, 3000]);
  const block2ZIndex = useTransform(scrollYProgress, [0.96, 0.961], [20, 220]);

  // Block 3 (z-index 10) - expand second
  const block3X = useTransform(scrollYProgress, [0.6, 0.7], [100, 0]);
  const block3Y = useTransform(scrollYProgress, [0.6, 0.7], [100, 0]);
  const block3Width = useTransform(scrollYProgress, [0.97, 1.0], [STATS[3].blockSize!.width, 3000]);
  const block3Height = useTransform(scrollYProgress, [0.97, 1.0], [STATS[3].blockSize!.height, 3000]);
  const block3ZIndex = useTransform(scrollYProgress, [0.96, 0.961], [10, 210]);

  // Block 4 (bottom layer, z-index 5) - expand first
  const block4X = useTransform(scrollYProgress, [0.8, 0.9], [100, 0]);
  const block4Y = useTransform(scrollYProgress, [0.8, 0.9], [100, 0]);
  const block4Width = useTransform(scrollYProgress, [0.96, 1.0], [STATS[4].blockSize!.width, 3000]);
  const block4Height = useTransform(scrollYProgress, [0.96, 1.0], [STATS[4].blockSize!.height, 3000]);
  const block4ZIndex = useTransform(scrollYProgress, [0.96, 0.961], [5, 205]);

  // Number counters - linear interpolation based on scroll progress
  const number1 = useTransform(scrollYProgress, [0.2, 0.3], [0, STATS[1].number!], {
    clamp: true,
  });
  const number1Rounded = useTransform(number1, (v) => Math.round(v));

  const number2 = useTransform(scrollYProgress, [0.4, 0.5], [0, STATS[2].number!], {
    clamp: true,
  });
  const number2Rounded = useTransform(number2, (v) => Math.round(v));

  const number3 = useTransform(scrollYProgress, [0.6, 0.7], [0, STATS[3].number!], {
    clamp: true,
  });
  const number3Rounded = useTransform(number3, (v) => Math.round(v));

  const number4 = useTransform(scrollYProgress, [0.8, 0.9], [0, STATS[4].number!], {
    clamp: true,
  });
  const number4Rounded = useTransform(number4, (v) => Math.round(v));

  return (
    <div ref={containerRef} className={styles.scrollContainer}>
      <div className={styles.stickyContainer}>
        {/* Persistent section title */}
        <h2 className={styles.sectionTitle}>The State of Our Fabs</h2>

        {/* Left column - Text content */}
        <div className={styles.leftColumn}>
          {/* State 1 */}
          <motion.div className={styles.stateText} style={{ opacity: state1Opacity, y: state1Y }}>
            <h2 className={styles.headingLarge}>
              {STATS[0].heading}
              <br />
              {STATS[0].subheading}
            </h2>
          </motion.div>

          {/* State 2 */}
          <motion.div className={styles.stateText} style={{ opacity: state2Opacity, y: state2Y }}>
            <h2 className={styles.headingMedium}>
              {STATS[1].heading}
              <span className={styles.redHighlight}>{STATS[1].highlight}</span>
            </h2>
          </motion.div>

          {/* State 3 */}
          <motion.div className={styles.stateText} style={{ opacity: state3Opacity, y: state3Y }}>
            <h2 className={styles.headingMedium}>
              {STATS[2].heading}
              <span className={styles.redHighlight}>{STATS[2].highlight}</span>
            </h2>
          </motion.div>

          {/* State 4 */}
          <motion.div className={styles.stateText} style={{ opacity: state4Opacity, y: state4Y }}>
            <h2 className={styles.headingMedium}>
              <span className={styles.tooltipWrapper}>
                <span className={styles.numberHighlight}>99%</span>
                <span className={styles.tooltip}>
                  Ratio of total chemical/gas input mass to final chip mass. Different processing steps can vary.
                </span>
              </span>
              {' of materials wasted in '}
              <span className={styles.redHighlight}>{STATS[3].highlight}</span>
            </h2>
          </motion.div>

          {/* State 5 */}
          <motion.div
            className={styles.stateText}
            style={{
              opacity: useTransform([state5Opacity, exitTextOpacity], ([s5, exit]) =>
                Math.min(s5 as number, exit as number)
              ),
              y: state5Y
            }}
          >
            <h2 className={styles.headingMedium}>
              {STATS[4].heading}
              <span className={styles.redHighlight}>{STATS[4].highlight}</span>
            </h2>
          </motion.div>
        </div>

        {/* Block 1 - State 2 (3 companies) */}
        <motion.div
          className={styles.redBlock}
          style={{
            x: block1X,
            y: block1Y,
            opacity: block1Opacity,
            width: block1Width,
            height: block1Height,
            backgroundColor: STATS[1].color,
            zIndex: block1ZIndex,
          }}
        >
          <motion.div className={styles.blockNumber} style={{ opacity: numberOpacity }}>
            {number1Rounded}
          </motion.div>
        </motion.div>

        {/* Block 2 - State 3 (1000+ steps) */}
        <motion.div
          className={styles.redBlock}
          style={{
            x: block2X,
            y: block2Y,
            opacity: block2Opacity,
            width: block2Width,
            height: block2Height,
            backgroundColor: STATS[2].color,
            zIndex: block2ZIndex,
          }}
        >
          <motion.div className={styles.blockNumber} style={{ opacity: numberOpacity }}>
            <motion.span>{number2Rounded}</motion.span>
            {STATS[2].suffix}
          </motion.div>
        </motion.div>

        {/* Block 3 - State 4 (99% waste) */}
        <motion.div
          className={styles.redBlock}
          style={{
            x: block3X,
            y: block3Y,
            opacity: block3Opacity,
            width: block3Width,
            height: block3Height,
            backgroundColor: STATS[3].color,
            zIndex: block3ZIndex,
          }}
        >
          <motion.div className={styles.blockNumber} style={{ opacity: numberOpacity }}>
            <motion.span>{number3Rounded}</motion.span>
            {STATS[3].suffix}
          </motion.div>
        </motion.div>

        {/* Block 4 - State 5 ($20B+ cost) */}
        <motion.div
          className={styles.redBlock}
          style={{
            x: block4X,
            y: block4Y,
            opacity: block4Opacity,
            width: block4Width,
            height: block4Height,
            backgroundColor: STATS[4].color,
            zIndex: block4ZIndex,
          }}
        >
          <motion.div className={styles.blockNumber} style={{ opacity: numberOpacity }}>
            $<motion.span>{number4Rounded}</motion.span>
            {STATS[4].suffix}
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
}


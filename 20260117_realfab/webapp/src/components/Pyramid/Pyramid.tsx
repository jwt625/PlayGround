'use client';

import { useRef, useState } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { pyramidItems } from './pyramidData';
import { PyramidItem } from './PyramidItem';
import { PyramidSVG } from './PyramidSVG';
import styles from './Pyramid.module.css';

export function Pyramid() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredItem, setHoveredItem] = useState<string | null>(null);

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start start', 'end end'],
  });

  // Sort items by entry order for animation
  const sortedItems = [...pyramidItems].sort((a, b) => a.entryOrder - b.entryOrder);

  // Title scale animation (0.0-0.1: scales down)
  const titleScale = useTransform(scrollYProgress, [0, 0.1], [1, 0.6]);
  const titleY = useTransform(scrollYProgress, [0, 0.1], [0, -40]);

  // Tier label animations
  const proteinLabelOpacity = useTransform(scrollYProgress, [0.1, 0.12, 0.33, 0.35], [0, 1, 1, 0]);
  const vegLabelOpacity = useTransform(scrollYProgress, [0.35, 0.37, 0.58, 0.6], [0, 1, 1, 0]);
  const grainLabelOpacity = useTransform(scrollYProgress, [0.6, 0.62, 0.83, 0.85], [0, 1, 1, 0]);

  // Final title animation
  const finalTitleOpacity = useTransform(scrollYProgress, [0.85, 0.9], [0, 1]);

  return (
    <div ref={containerRef} className={styles.container}>
      <div className={styles.content}>
        {/* Animated Title */}
        <motion.div
          className={styles.titleContainer}
          style={{ scale: titleScale, y: titleY }}
        >
          <motion.div className={styles.kicker}>
            Introducing
          </motion.div>
          <motion.h2 className={styles.heading}>
            The New Fab Pyramid
          </motion.h2>
          <motion.p className={styles.description}>
            An inverted pyramid: build on additive core technologies, use local & accessible tools, minimize reliance on big fab.
          </motion.p>
        </motion.div>

        {/* Pyramid Visualization Container */}
        <div className={styles.pyramidContainer}>
          {/* SVG Outline */}
          <PyramidSVG scrollYProgress={scrollYProgress} />

          {/* Tier Labels */}
          <motion.div
            className={styles.tierLabel}
            data-tier="additive"
            style={{ opacity: proteinLabelOpacity }}
          >
            Additive Core
          </motion.div>
          <motion.div
            className={styles.tierLabel}
            data-tier="local"
            style={{ opacity: vegLabelOpacity }}
          >
            Local & Accessible
          </motion.div>
          <motion.div
            className={styles.tierLabel}
            data-tier="bigfab"
            style={{ opacity: grainLabelOpacity }}
          >
            Big Fab (Minimize)
          </motion.div>

          {/* All Items with Absolute Positioning */}
          <div className={styles.itemsContainer}>
            {sortedItems.map((item) => (
              <PyramidItem
                key={item.id}
                item={item}
                scrollYProgress={scrollYProgress}
                onHover={setHoveredItem}
                isHovered={hoveredItem === item.id}
              />
            ))}
          </div>
        </div>

        {/* Final Title */}
        <motion.div
          className={styles.finalTitle}
          style={{ opacity: finalTitleOpacity }}
        >
          <h3>Build the Future, Locally</h3>
          <p>Democratize fabrication through accessible additive technologies</p>
        </motion.div>
      </div>
    </div>
  );
}


'use client';

import { motion, useTransform, MotionValue } from 'framer-motion';
import Image from 'next/image';
import { PyramidItem as PyramidItemType } from './pyramidData';
import { springs } from '@/lib/springs';
import styles from './PyramidItem.module.css';

interface PyramidItemProps {
  item: PyramidItemType;
  scrollYProgress: MotionValue<number>;
  onHover: (id: string | null) => void;
  isHovered: boolean;
}

export function PyramidItem({ item, scrollYProgress, onHover, isHovered }: PyramidItemProps) {
  // Calculate entrance timing based on entryOrder
  // Items 1-13: enter during 0.1-0.35 (protein tier)
  // Items 14-26: enter during 0.35-0.6 (vegetables tier)
  // Items 27-38: enter during 0.6-0.85 (grains tier)

  let phaseStart: number, phaseEnd: number, itemsInPhase: number, indexInPhase: number;

  if (item.entryOrder <= 13) {
    // Protein tier
    phaseStart = 0.1;
    phaseEnd = 0.35;
    itemsInPhase = 13;
    indexInPhase = item.entryOrder - 1;
  } else if (item.entryOrder <= 26) {
    // Vegetables tier
    phaseStart = 0.35;
    phaseEnd = 0.6;
    itemsInPhase = 13;
    indexInPhase = item.entryOrder - 14;
  } else {
    // Grains tier
    phaseStart = 0.6;
    phaseEnd = 0.85;
    itemsInPhase = 12;
    indexInPhase = item.entryOrder - 27;
  }

  const phaseDuration = phaseEnd - phaseStart;
  const staggerDelay = (phaseDuration * indexInPhase) / itemsInPhase;
  const itemStart = phaseStart + staggerDelay;
  const itemEnd = Math.min(itemStart + 0.15, phaseEnd);

  // Entrance animations
  const opacity = useTransform(
    scrollYProgress,
    [itemStart, itemStart + 0.05],
    [0, 1]
  );

  const scale = useTransform(
    scrollYProgress,
    [itemStart, itemEnd],
    [0.6, 1]
  );

  const y = useTransform(
    scrollYProgress,
    [itemStart, itemEnd],
    [40, 0]
  );

  return (
    <motion.div
      className={styles.item}
      style={{
        top: `${item.top}%`,
        left: `${item.left}%`,
        width: `${item.width}%`,
        height: `${item.height}%`,
        zIndex: item.zIndex,
        opacity,
        scale,
        y,
      }}
    >
      <div className={styles.imageContainer}>
        <Image
          src={item.image}
          alt={item.name}
          fill
          sizes={`${item.width}vw`}
          className={styles.image}
        />
      </div>
    </motion.div>
  );
}


'use client';

import { motion, AnimatePresence, useAnimate } from 'framer-motion';
import { useState, useEffect } from 'react';
import Image from 'next/image';
import styles from './IntroCurtain.module.css';

interface IntroCurtainProps {
  onComplete?: () => void;
}

export function IntroCurtain({ onComplete }: IntroCurtainProps) {
  const [isVisible, setIsVisible] = useState(true);
  const [phase, setPhase] = useState<'converge' | 'exit'>('converge');

  useEffect(() => {
    // Phase 1: Items converge towards center (0-1.2s) - stop 20% earlier
    // Phase 2: Items separate and move up while curtain slides (1.2-2.2s)
    const exitTimer = setTimeout(() => {
      setPhase('exit');
    }, 1200);

    const curtainTimer = setTimeout(() => {
      setIsVisible(false);
      setTimeout(() => {
        onComplete?.();
      }, 1000); // Match curtain animation duration
    }, 1200);

    return () => {
      clearTimeout(exitTimer);
      clearTimeout(curtainTimer);
    };
  }, [onComplete]);

  // Smooth ease-in-out with flat middle segment
  const customEase = [0.45, 0, 0.55, 1]; // cubic-bezier for smooth accel/decel

  // Item 1: Starts just below top edge, moves down towards center (stops 20% earlier)
  const item1Variants = {
    initial: { x: 0, y: '-45vh' }, // Just below top edge (center is at 0,0)
    converge: {
      x: 0,
      y: '-9vh', // Stop 20% earlier (20% of 45vh = 9vh from center)
      transition: {
        duration: 0.6,
        ease: customEase,
      },
    },
    exit: {
      x: 0,
      y: '-200vh', // Move up and out (no separate phase, goes directly)
      transition: {
        duration: 1,
        ease: customEase,
      },
    },
  };

  // Item 2: Starts bottom left, moves up-right towards center (stops 20% earlier)
  const item2Variants = {
    initial: { x: '-25vw', y: '35vh' }, // Bottom left, half outside viewport
    converge: {
      x: '-8vw', // Stop 20% earlier (20% of 25vw = 5vw, so -3vw becomes -8vw)
      y: '9vh', // Stop 20% earlier (20% of 33vh = ~7vh, so 2vh becomes 9vh)
      transition: {
        duration: 0.6,
        ease: customEase,
        delay: 0.6, // Moves third (after item 3)
      },
    },
    exit: {
      x: '-25vw', // Separate halfway back while moving up (halfway between -8vw and -25vw)
      y: '-100vh', // Move up and out
      transition: {
        duration: 1,
        ease: customEase,
        delay: 0, // Exit immediately when phase changes
      },
    },
  };

  // Item 3: Starts bottom right, moves left towards center (stops 20% earlier)
  const item3Variants = {
    initial: { x: '25vw', y: '25vh' }, // Bottom right, near bottom within canvas
    converge: {
      x: '8vw', // Stop 20% earlier (20% of 25vw = 5vw, so 3vw becomes 8vw)
      y: '7vh', // Stop 20% earlier (20% of 25vh = 5vh, so 2vh becomes 7vh)
      transition: {
        duration: 0.6,
        ease: customEase,
        delay: 0.3, // Moves second (after item 1)
      },
    },
    exit: {
      x: '25vw', // Separate halfway back while moving up (halfway between 8vw and 25vw)
      y: '-100vh', // Move up and out
      transition: {
        duration: 1,
        ease: customEase,
        delay: 0, // Exit immediately when phase changes
      },
    },
  };

  // Curtain slide-up animation
  const curtainVariants = {
    initial: { y: 0 },
    exit: {
      y: '-100%',
      transition: {
        duration: 1,
        ease: [0.65, 0, 0.35, 1], // cubic-bezier(0.65, 0, 0.35, 1)
      },
    },
  };

  return (
    <>
      {/* Items - positioned independently */}
      <div className={styles.itemsContainer}>
        {/* Item 1: LPBF Metal Printer - moves down from top */}
        <motion.div
          className={styles.item1}
          variants={item1Variants}
          initial="initial"
          animate={phase}
        >
          <Image
            src="/images/pyramid/lpbf-metal-printer.webp"
            alt="LPBF Metal Printer"
            width={400}
            height={400}
            priority
            className={styles.image}
          />
        </motion.div>

        {/* Item 2: Silicon Boule - moves up-right from bottom left */}
        <motion.div
          className={styles.item2}
          variants={item2Variants}
          initial="initial"
          animate={phase}
        >
          <Image
            src="/images/pyramid/silicon-boule.webp"
            alt="Silicon Boule"
            width={400}
            height={400}
            priority
            className={styles.image}
          />
        </motion.div>

        {/* Item 3: TPP System - moves left from bottom right */}
        <motion.div
          className={styles.item3}
          variants={item3Variants}
          initial="initial"
          animate={phase}
        >
          <Image
            src="/images/pyramid/tpp-system.webp"
            alt="TPP System"
            width={400}
            height={400}
            priority
            className={styles.image}
          />
        </motion.div>
      </div>

      {/* Curtain - slides up independently */}
      <AnimatePresence>
        {isVisible && (
          <motion.div
            className={styles.overlay}
            variants={curtainVariants}
            initial="initial"
            exit="exit"
          />
        )}
      </AnimatePresence>
    </>
  );
}


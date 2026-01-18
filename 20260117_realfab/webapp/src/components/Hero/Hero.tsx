'use client';

import { motion, useScroll, useTransform } from 'framer-motion';
import { useState, useRef } from 'react';
import Image from 'next/image';
import { VideoModal } from '@/components/VideoModal';
import { springs } from '@/lib/springs';
import styles from './Hero.module.css';

const heroVariants = {
  initial: { opacity: 0, y: 30 },
  animate: { 
    opacity: 1, 
    y: 0,
    transition: springs.gentle,
  },
};

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.15,
      delayChildren: 0.5, // Delay after curtain reveals (1.5s intro + 1s curtain = 2.5s total, hero starts at ~0.5s after reveal)
    },
  },
};

interface HeroProps {
  title?: string;
  description?: string;
  videoUrl?: string;
  videoThumbnail?: string;
  showIntro?: boolean;
}

export function Hero({
  title = "Real Fab Starts Here",
  description = "Better technology begins with accessible fabrication—not gatekept mega-fabs. The new paradigm for semiconductor manufacturing defines real chips as locally-made, additive-manufactured, and democratically accessible, placing them back at the center of innovation.",
  videoUrl,
  videoThumbnail,
  showIntro = false,
}: HeroProps) {
  const [isVideoOpen, setIsVideoOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start start', 'end start'],
  });

  // Parallax transforms for floating elements
  const y1 = useTransform(scrollYProgress, [0, 1], [0, -150]);
  const y2 = useTransform(scrollYProgress, [0, 1], [0, -100]);
  const y3 = useTransform(scrollYProgress, [0, 1], [0, -200]);
  const rotate1 = useTransform(scrollYProgress, [0, 1], [0, 45]);
  const rotate2 = useTransform(scrollYProgress, [0, 1], [0, -30]);
  const rotate3 = useTransform(scrollYProgress, [0, 1], [0, 60]);

  return (
    <>
      <div ref={containerRef} className={styles.hero}>
        {/* Parallax floating elements */}
        <motion.div
          className={styles.floatingElement1}
          style={{ y: y1, rotate: rotate1 }}
        >
          <Image
            src="/images/pyramid/lpbf-metal-printer.webp"
            alt="LPBF Metal Printer"
            width={200}
            height={200}
            className={styles.floatingImage}
          />
        </motion.div>
        <motion.div
          className={styles.floatingElement2}
          style={{ y: y2, rotate: rotate2 }}
        >
          <Image
            src="/images/pyramid/silicon-boule.webp"
            alt="Silicon Boule"
            width={200}
            height={200}
            className={styles.floatingImage}
          />
        </motion.div>
        <motion.div
          className={styles.floatingElement3}
          style={{ y: y3, rotate: rotate3 }}
        >
          <Image
            src="/images/pyramid/tpp-system.webp"
            alt="TPP System"
            width={200}
            height={200}
            className={styles.floatingImage}
          />
        </motion.div>

        <motion.div
          className={styles.content}
          variants={staggerContainer}
          initial={showIntro ? "initial" : false}
          animate="animate"
        >
          <motion.h1 className={styles.title} variants={heroVariants}>
            {title}
          </motion.h1>

          <motion.p className={styles.description} variants={heroVariants}>
            {description}
          </motion.p>

          <motion.div className={styles.ctas} variants={heroVariants}>
            <button className={styles.ctaPrimary}>
              View the Guidelines
            </button>
          </motion.div>

          {videoUrl && (
            <motion.div className={styles.videoPreview} variants={heroVariants}>
              <button
                className={styles.videoButton}
                onClick={() => setIsVideoOpen(true)}
                aria-label="Play announcement video"
              >
                <div className={styles.playIcon}>
                  <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M8 5v14l11-7z" />
                  </svg>
                </div>
                <span>Watch the Announcement</span>
              </button>
            </motion.div>
          )}
        </motion.div>
      </div>

      {videoUrl && (
        <VideoModal
          isOpen={isVideoOpen}
          onClose={() => setIsVideoOpen(false)}
          videoUrl={videoUrl}
          thumbnail={videoThumbnail}
        />
      )}
    </>
  );
}


'use client';

import { motion } from 'framer-motion';
import { useState } from 'react';
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
      delayChildren: 0.2,
    },
  },
};

interface HeroProps {
  title?: string;
  subtitle?: string;
  description?: string;
  videoUrl?: string;
  videoThumbnail?: string;
}

export function Hero({
  title = "Build Real Chips",
  subtitle = "RealFab.org",
  description = "Decentralized semiconductor fabrication through additive manufacturing and distributed production.",
  videoUrl,
  videoThumbnail,
}: HeroProps) {
  const [isVideoOpen, setIsVideoOpen] = useState(false);

  return (
    <>
      <div className={styles.hero}>
        <motion.div
          className={styles.content}
          variants={staggerContainer}
          initial="initial"
          animate="animate"
        >
          <motion.h1 className={styles.title} variants={heroVariants}>
            {title}
          </motion.h1>
          
          <motion.p className={styles.subtitle} variants={heroVariants}>
            {subtitle}
          </motion.p>
          
          <motion.p className={styles.description} variants={heroVariants}>
            {description}
          </motion.p>

          <motion.div className={styles.ctas} variants={heroVariants}>
            <button className={styles.ctaPrimary}>
              View the Manifesto
            </button>
            <button className={styles.ctaSecondary}>
              See the Data
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


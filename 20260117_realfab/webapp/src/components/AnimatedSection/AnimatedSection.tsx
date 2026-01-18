'use client';

import { ReactNode, useRef } from 'react';
import { motion } from 'framer-motion';
import { useInView } from '@/hooks/useInView';
import { Section } from '@/components/Section';
import { fadeInUp, staggerContainer } from '@/lib/animations';

interface AnimatedSectionProps {
  id?: string;
  children: ReactNode;
  className?: string;
  dark?: boolean;
  fullHeight?: boolean;
  stagger?: boolean;
  delay?: number;
}

/**
 * Section component with scroll-triggered fade-in animation
 * 
 * @param stagger - Enable stagger animation for children
 * @param delay - Animation delay in seconds
 */
export function AnimatedSection({
  id,
  children,
  className,
  dark,
  fullHeight,
  stagger = false,
  delay = 0,
}: AnimatedSectionProps) {
  const ref = useRef<HTMLElement>(null);
  const isInView = useInView(ref, { threshold: 0.1, triggerOnce: true });

  const variants = stagger ? staggerContainer : fadeInUp;

  return (
    <Section
      id={id}
      ref={ref}
      className={className}
      dark={dark}
      fullHeight={fullHeight}
    >
      <motion.div
        initial="initial"
        animate={isInView ? 'animate' : 'initial'}
        variants={variants}
        transition={{ delay }}
      >
        {children}
      </motion.div>
    </Section>
  );
}


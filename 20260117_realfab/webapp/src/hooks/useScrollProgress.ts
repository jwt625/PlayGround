'use client';

import { useScroll, useTransform, MotionValue } from 'framer-motion';
import { RefObject } from 'react';

interface UseScrollProgressOptions {
  target?: RefObject<HTMLElement>;
  offset?: ["start end" | "end start" | "start start" | "end end", "start end" | "end start" | "start start" | "end end"];
}

interface ScrollProgressReturn {
  scrollYProgress: MotionValue<number>;
  scrollY: MotionValue<number>;
}

/**
 * Hook to track scroll progress of an element or the entire page
 * 
 * @param options - Configuration options
 * @param options.target - Optional ref to track specific element scroll
 * @param options.offset - Scroll offset range, e.g., ["start end", "end start"]
 * @returns Object containing scrollYProgress (0-1) and scrollY (pixels)
 * 
 * @example
 * const { scrollYProgress } = useScrollProgress();
 * const opacity = useTransform(scrollYProgress, [0, 1], [0, 1]);
 * 
 * @example
 * const ref = useRef(null);
 * const { scrollYProgress } = useScrollProgress({ 
 *   target: ref, 
 *   offset: ["start end", "end start"] 
 * });
 */
export function useScrollProgress(
  options: UseScrollProgressOptions = {}
): ScrollProgressReturn {
  const { target, offset = ['start end', 'end start'] as const } = options;

  const { scrollYProgress, scrollY } = useScroll({
    target,
    offset,
  });

  return { scrollYProgress, scrollY };
}

/**
 * Hook to create parallax effect based on scroll progress
 * 
 * @param speed - Parallax speed multiplier (negative for reverse direction)
 * @param options - Same options as useScrollProgress
 * @returns MotionValue for y transform
 * 
 * @example
 * const y = useParallax(-50);
 * <motion.div style={{ y }}>Parallax content</motion.div>
 * 
 * @example
 * const ref = useRef(null);
 * const y = useParallax(100, { target: ref });
 */
export function useParallax(
  speed: number = 50,
  options: UseScrollProgressOptions = {}
): MotionValue<number> {
  const { scrollYProgress } = useScrollProgress(options);
  
  return useTransform(scrollYProgress, [0, 1], [0, speed]);
}

/**
 * Hook to create scroll-based opacity fade
 * 
 * @param fadeRange - Range of scroll progress for fade [start, end]
 * @param options - Same options as useScrollProgress
 * @returns MotionValue for opacity
 * 
 * @example
 * const opacity = useScrollFade([0, 0.5]);
 * <motion.div style={{ opacity }}>Fading content</motion.div>
 */
export function useScrollFade(
  fadeRange: [number, number] = [0, 1],
  options: UseScrollProgressOptions = {}
): MotionValue<number> {
  const { scrollYProgress } = useScrollProgress(options);
  
  return useTransform(scrollYProgress, fadeRange, [0, 1]);
}

/**
 * Hook to create scroll-based scale effect
 * 
 * @param scaleRange - Range of scale values [start, end]
 * @param options - Same options as useScrollProgress
 * @returns MotionValue for scale
 * 
 * @example
 * const scale = useScrollScale([0.8, 1]);
 * <motion.div style={{ scale }}>Scaling content</motion.div>
 */
export function useScrollScale(
  scaleRange: [number, number] = [0.8, 1],
  options: UseScrollProgressOptions = {}
): MotionValue<number> {
  const { scrollYProgress } = useScrollProgress(options);
  
  return useTransform(scrollYProgress, [0, 1], scaleRange);
}


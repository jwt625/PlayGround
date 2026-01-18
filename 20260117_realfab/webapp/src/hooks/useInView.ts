'use client';

import { useEffect, useState, RefObject } from 'react';

interface UseInViewOptions {
  threshold?: number | number[];
  rootMargin?: string;
  triggerOnce?: boolean;
}

/**
 * Hook to detect when an element is in viewport
 * 
 * @param ref - React ref to the element to observe
 * @param options - IntersectionObserver options
 * @returns Boolean indicating if element is in view
 * 
 * @example
 * const ref = useRef(null);
 * const isInView = useInView(ref, { threshold: 0.5 });
 */
export function useInView(
  ref: RefObject<HTMLElement | null>,
  options: UseInViewOptions = {}
): boolean {
  const { threshold = 0, rootMargin = '0px', triggerOnce = false } = options;
  const [isInView, setIsInView] = useState(false);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        const inView = entry.isIntersecting;
        setIsInView(inView);

        // If triggerOnce is true, disconnect after first intersection
        if (inView && triggerOnce) {
          observer.disconnect();
        }
      },
      { threshold, rootMargin }
    );

    observer.observe(element);

    return () => {
      observer.disconnect();
    };
  }, [ref, threshold, rootMargin, triggerOnce]);

  return isInView;
}

/**
 * Hook to get intersection ratio of an element
 * 
 * @param ref - React ref to the element to observe
 * @param options - IntersectionObserver options
 * @returns Number between 0 and 1 indicating intersection ratio
 * 
 * @example
 * const ref = useRef(null);
 * const ratio = useIntersectionRatio(ref);
 */
export function useIntersectionRatio(
  ref: RefObject<HTMLElement | null>,
  options: Omit<UseInViewOptions, 'triggerOnce'> = {}
): number {
  const { threshold = 0, rootMargin = '0px' } = options;
  const [ratio, setRatio] = useState(0);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        setRatio(entry.intersectionRatio);
      },
      { threshold, rootMargin }
    );

    observer.observe(element);

    return () => {
      observer.disconnect();
    };
  }, [ref, threshold, rootMargin]);

  return ratio;
}


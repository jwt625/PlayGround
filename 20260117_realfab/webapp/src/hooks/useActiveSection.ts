'use client';

import { useEffect, useState } from 'react';

/**
 * Hook to track which section is currently active based on scroll position
 * 
 * @param sectionIds - Array of section IDs to track
 * @param offset - Offset from top of viewport (in pixels) to consider a section active
 * @returns ID of the currently active section
 * 
 * @example
 * const activeSection = useActiveSection(['hero', 'problem', 'solution', 'pyramid']);
 */
export function useActiveSection(
  sectionIds: string[],
  offset: number = 100
): string | null {
  const [activeSection, setActiveSection] = useState<string | null>(null);

  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.scrollY + offset;

      // Find the section that is currently in view
      for (let i = sectionIds.length - 1; i >= 0; i--) {
        const section = document.getElementById(sectionIds[i]);
        if (section) {
          const sectionTop = section.offsetTop;
          const sectionBottom = sectionTop + section.offsetHeight;

          if (scrollPosition >= sectionTop && scrollPosition < sectionBottom) {
            setActiveSection(sectionIds[i]);
            return;
          }
        }
      }

      // If we're at the very top, set first section as active
      if (scrollPosition < offset) {
        setActiveSection(sectionIds[0]);
      }
    };

    // Initial check
    handleScroll();

    // Listen to scroll events
    window.addEventListener('scroll', handleScroll, { passive: true });

    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, [sectionIds, offset]);

  return activeSection;
}

/**
 * Smooth scroll to a section by ID
 * 
 * @param sectionId - ID of the section to scroll to
 * @param offset - Offset from top (in pixels)
 * 
 * @example
 * scrollToSection('pyramid', 80);
 */
export function scrollToSection(sectionId: string, offset: number = 0): void {
  const section = document.getElementById(sectionId);
  if (!section) return;

  const targetPosition = section.offsetTop - offset;

  window.scrollTo({
    top: targetPosition,
    behavior: 'smooth',
  });
}


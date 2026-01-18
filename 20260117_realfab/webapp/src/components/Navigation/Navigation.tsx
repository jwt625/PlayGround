'use client';

import { motion, useSpring } from 'framer-motion';
import { useActiveSection, scrollToSection } from '@/hooks/useActiveSection';
import styles from './Navigation.module.css';

export interface NavSection {
  id: string;
  label: string;
}

interface NavigationProps {
  sections: NavSection[];
}

export function Navigation({ sections }: NavigationProps) {
  const activeSection = useActiveSection(
    sections.map((s) => s.id),
    100
  );

  // Spring animation for smooth dot position transitions
  const activeDotSpring = useSpring(0, {
    stiffness: 120,
    damping: 20,
  });

  // Update spring value when active section changes
  const activeIndex = sections.findIndex((s) => s.id === activeSection);
  if (activeIndex !== -1) {
    activeDotSpring.set(activeIndex);
  }

  const handleNavClick = (sectionId: string) => {
    scrollToSection(sectionId, 80);
  };

  return (
    <nav className={styles.navigation}>
      <ul className={styles.navList}>
        {sections.map((section, index) => {
          const isActive = section.id === activeSection;

          return (
            <li key={section.id} className={styles.navItem}>
              <button
                onClick={() => handleNavClick(section.id)}
                className={styles.navButton}
                aria-label={`Navigate to ${section.label}`}
                aria-current={isActive ? 'true' : 'false'}
              >
                {/* Dot indicator */}
                <motion.span
                  className={styles.navDot}
                  initial={{ opacity: 0 }}
                  animate={{
                    opacity: isActive ? 1 : 0.3,
                    scale: isActive ? 1.2 : 1,
                  }}
                  transition={{
                    type: 'spring',
                    stiffness: 120,
                    damping: 20,
                    duration: 0.65,
                  }}
                />

                {/* Label - fades in on hover or when active */}
                <motion.span
                  className={styles.navLabel}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{
                    opacity: isActive ? 1 : 0,
                    x: isActive ? 0 : -10,
                  }}
                  whileHover={{
                    opacity: 1,
                    x: 0,
                  }}
                  transition={{
                    duration: 0.3,
                    ease: 'easeOut',
                    delay: isActive ? 0.2 : 0,
                  }}
                >
                  {section.label}
                </motion.span>
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}


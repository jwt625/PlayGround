'use client';

import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
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
  const [isVisible, setIsVisible] = useState(false);

  const activeSection = useActiveSection(
    sections.map((s) => s.id),
    100
  );

  // Show navigation after scrolling past hero section
  useEffect(() => {
    const handleScroll = () => {
      const heroSection = document.getElementById('hero');
      if (heroSection) {
        const heroBottom = heroSection.offsetTop + heroSection.offsetHeight;
        setIsVisible(window.scrollY > heroBottom - 100);
      }
    };

    handleScroll(); // Check initial state
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleNavClick = (sectionId: string) => {
    scrollToSection(sectionId, 80);
  };

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.nav
          className={styles.navigation}
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.3, ease: 'easeOut' }}
        >
          <ul className={styles.navList}>
            {sections.map((section) => {
              const isActive = section.id === activeSection;

              return (
                <li key={section.id} className={styles.navItem}>
                  <button
                    onClick={() => handleNavClick(section.id)}
                    className={`${styles.navButton} ${isActive ? styles.active : ''}`}
                    aria-label={`Navigate to ${section.label}`}
                    aria-current={isActive ? 'true' : 'false'}
                  >
                    {/* Dot indicator - always visible for inactive, hidden for active */}
                    <motion.span
                      className={styles.navDot}
                      animate={{
                        opacity: isActive ? 0 : 0.5,
                        scale: 1,
                      }}
                      transition={{
                        type: 'spring',
                        stiffness: 120,
                        damping: 20,
                      }}
                    />

                    {/* Label - only visible when active */}
                    <motion.span
                      className={styles.navLabel}
                      animate={{
                        opacity: isActive ? 1 : 0,
                        width: isActive ? 'auto' : 0,
                      }}
                      transition={{
                        type: 'spring',
                        stiffness: 120,
                        damping: 20,
                      }}
                    >
                      {section.label}
                    </motion.span>
                  </button>
                </li>
              );
            })}
          </ul>
        </motion.nav>
      )}
    </AnimatePresence>
  );
}


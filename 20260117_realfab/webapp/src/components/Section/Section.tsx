import { ReactNode, forwardRef } from 'react';
import styles from './Section.module.css';

interface SectionProps {
  id?: string;
  children: ReactNode;
  className?: string;
  dark?: boolean;
  fullHeight?: boolean;
}

/**
 * Section wrapper component for page sections
 * 
 * @param id - Section ID for navigation
 * @param children - Section content
 * @param className - Additional CSS classes
 * @param dark - Apply dark theme
 * @param fullHeight - Make section full viewport height
 */
export const Section = forwardRef<HTMLElement, SectionProps>(
  ({ id, children, className = '', dark = false, fullHeight = false }, ref) => {
    const classes = [
      styles.section,
      dark && styles.dark,
      fullHeight && styles.fullHeight,
      className,
    ]
      .filter(Boolean)
      .join(' ');

    return (
      <section id={id} ref={ref} className={classes}>
        {children}
      </section>
    );
  }
);

Section.displayName = 'Section';


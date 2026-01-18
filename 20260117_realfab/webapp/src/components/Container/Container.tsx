import { ReactNode } from 'react';
import styles from './Container.module.css';

interface ContainerProps {
  children: ReactNode;
  className?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
}

/**
 * Container component for content width constraints
 * 
 * @param children - Container content
 * @param className - Additional CSS classes
 * @param size - Container max-width size
 */
export function Container({
  children,
  className = '',
  size = 'xl',
}: ContainerProps) {
  const classes = [
    styles.container,
    styles[`size-${size}`],
    className,
  ]
    .filter(Boolean)
    .join(' ');

  return <div className={classes}>{children}</div>;
}


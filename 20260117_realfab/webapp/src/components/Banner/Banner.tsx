'use client';

import styles from './Banner.module.css';

export function Banner() {
  return (
    <div className={styles.banner}>
      <div className={styles.content}>
        <svg
          className={styles.flag}
          viewBox="0 0 20 12"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <rect width="20" height="12" fill="#B22234" />
          <rect y="0.923" width="20" height="0.923" fill="white" />
          <rect y="2.769" width="20" height="0.923" fill="white" />
          <rect y="4.615" width="20" height="0.923" fill="white" />
          <rect y="6.462" width="20" height="0.923" fill="white" />
          <rect y="8.308" width="20" height="0.923" fill="white" />
          <rect y="10.154" width="20" height="0.923" fill="white" />
          <rect width="8" height="6.462" fill="#3C3B6E" />
        </svg>
        <span className={styles.text}>
          An official website of the <strong>Outside Five Sigma</strong>
        </span>
      </div>
    </div>
  );
}


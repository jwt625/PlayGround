'use client';

import Image from 'next/image';
import styles from './Banner.module.css';

export function Banner() {
  return (
    <div className={styles.banner}>
      <div className={styles.content}>
        <Image
          src="/images/twm.png"
          alt="TWM Logo"
          width={30}
          height={18}
          className={styles.logo}
        />
        <span className={styles.text}>
          An official website of the <strong>Outside Five Sigma</strong>
        </span>
      </div>
    </div>
  );
}


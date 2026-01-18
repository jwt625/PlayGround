'use client';

import { useState } from 'react';
import { IntroCurtain } from '@/components/IntroCurtain';
import { Banner } from '@/components/Banner';
import { Navigation } from '@/components/Navigation';
import { Hero } from '@/components/Hero';
import { Stats } from '@/components/Stats';
import { Problem } from '@/components/Problem';
import { Solution } from '@/components/Solution';
import { Pyramid } from '@/components/Pyramid';
import { Section } from '@/components/Section';
import { AnimatedSection } from '@/components/AnimatedSection';
import { Container } from '@/components/Container';
import styles from './page.module.css';

const sections = [
  { id: 'hero', label: 'Home' },
  { id: 'stats', label: 'Stats' },
  { id: 'problem', label: 'Problem' },
  { id: 'solution', label: 'Solution' },
  { id: 'pyramid', label: 'Pyramid' },
  { id: 'resources', label: 'Resources' },
];

export default function Home() {
  const [introComplete, setIntroComplete] = useState(false);

  return (
    <>
      <IntroCurtain onComplete={() => setIntroComplete(true)} />

      <Banner />
      <Navigation sections={sections} />

      <main>
        {/* Hero Section */}
        <Section id="hero" fullHeight>
          <Hero
            title="Real Fab Starts Here"
            description="Better technology begins with accessible fabrication—not gatekept mega-fabs. The new paradigm for semiconductor manufacturing defines real chips as locally-made, additive-manufactured, and democratically accessible, placing them back at the center of innovation."
            videoUrl="/videos/announcement-placeholder.mp4"
            showIntro={!introComplete}
          />
        </Section>

        {/* Statistics Section */}
        <Section id="stats">
          <Stats />
        </Section>

        {/* Problem Section */}
        <Section id="problem">
          <Problem />
        </Section>

        {/* Solution Section */}
        <Section id="solution">
          <Solution />
        </Section>

        {/* Pyramid Section */}
        <Section id="pyramid">
          <Pyramid />
        </Section>

        {/* Resources Section */}
        <AnimatedSection id="resources">
          <Container>
            <div className={styles.content}>
              <h2 className={styles.sectionTitle}>Resources</h2>
              <p className={styles.sectionText}>
                Learn more about decentralized fabrication and additive
                manufacturing for semiconductors.
              </p>
            </div>
          </Container>
        </AnimatedSection>
      </main>
    </>
  );
}

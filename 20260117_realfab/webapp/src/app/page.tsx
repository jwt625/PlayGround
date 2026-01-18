'use client';

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
  return (
    <>
      <Banner />
      <Navigation sections={sections} />

      <main>
        {/* Hero Section */}
        <Section id="hero" fullHeight>
          <Hero
            title="Build Real Chips"
            subtitle="RealFab.org"
            description="The semiconductor industry has centralized fabrication into mega-fabs with hundreds of process steps, massive material waste, and monopolistic control. It's time for a revolution in how we make things."
            videoUrl="/videos/announcement-placeholder.mp4"
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

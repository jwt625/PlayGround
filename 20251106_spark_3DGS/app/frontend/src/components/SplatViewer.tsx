import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { SplatMesh } from '@sparkjsdev/spark';

interface SplatViewerProps {
  splatUrl?: string;
}

export function SplatViewer({ splatUrl }: SplatViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const splatMeshRef = useRef<SplatMesh | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Initialize Three.js scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    sceneRef.current = scene;

    // Initialize camera
    const camera = new THREE.PerspectiveCamera(
      75,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 1, 3);
    cameraRef.current = camera;

    // Initialize renderer with performance optimizations
    const renderer = new THREE.WebGLRenderer({
      antialias: false,  // Disable antialiasing for better performance
      powerPreference: 'high-performance'
    });
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));  // Cap pixel ratio for performance
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Add orbit controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 0, 0);
    controls.update();
    controlsRef.current = controls;

    // Animation loop
    let animationId: number;
    const animate = () => {
      animationId = requestAnimationFrame(animate);

      // Update controls
      if (controlsRef.current) {
        controlsRef.current.update();
      }

      renderer.render(scene, camera);
    };
    animate();

    // Handle window resize
    const handleResize = () => {
      if (!containerRef.current || !camera || !renderer) return;
      
      const width = containerRef.current.clientWidth;
      const height = containerRef.current.clientHeight;
      
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener('resize', handleResize);

      if (controlsRef.current) {
        controlsRef.current.dispose();
      }

      if (splatMeshRef.current) {
        scene.remove(splatMeshRef.current);
      }

      if (renderer) {
        renderer.dispose();
        containerRef.current?.removeChild(renderer.domElement);
      }
    };
  }, []);

  // Load splat when URL changes
  useEffect(() => {
    if (!splatUrl || !sceneRef.current) return;

    console.log('Loading splat from:', splatUrl);
    setLoading(true);
    setError(null);

    // Remove existing splat
    if (splatMeshRef.current) {
      sceneRef.current.remove(splatMeshRef.current);
      splatMeshRef.current = null;
    }

    try {
      console.log('Creating SplatMesh...');
      // Create and add splat mesh - it loads asynchronously in the background
      const splatMesh = new SplatMesh({ url: splatUrl });
      splatMesh.position.set(0, 0, 0);
      sceneRef.current.add(splatMesh);
      splatMeshRef.current = splatMesh;

      console.log('SplatMesh added to scene, loading in background...');

      // Hide loading indicator after a short delay
      // The splat will continue loading and appear when ready
      setTimeout(() => setLoading(false), 3000);
    } catch (err) {
      console.error('Error creating splat:', err);
      setError(err instanceof Error ? err.message : 'Failed to create splat');
      setLoading(false);
    }
  }, [splatUrl]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
      
      {loading && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          background: 'rgba(0, 0, 0, 0.7)',
          color: 'white',
          padding: '20px',
          borderRadius: '8px',
        }}>
          Loading splat...
        </div>
      )}
      
      {error && (
        <div style={{
          position: 'absolute',
          top: '20px',
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'rgba(255, 0, 0, 0.8)',
          color: 'white',
          padding: '10px 20px',
          borderRadius: '8px',
        }}>
          Error: {error}
        </div>
      )}
    </div>
  );
}


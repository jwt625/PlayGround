import { useState, useEffect } from 'react';
import { SplatViewer } from './components/SplatViewer';
import './App.css';

function App() {
  const DEFAULT_PLY_URL = 'https://huggingface.co/datasets/jwt625/splat/resolve/main/intel100G_CWDM.ply';
  const [splatUrl, setSplatUrl] = useState<string>(DEFAULT_PLY_URL);
  const [uploadStatus, setUploadStatus] = useState<string>('');

  // Check for URL parameter on mount, otherwise use default
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const urlParam = params.get('url');

    if (urlParam) {
      console.log('Loading from URL parameter:', urlParam);
      setSplatUrl(urlParam);
      setUploadStatus('Loading from URL...');
      setTimeout(() => setUploadStatus(''), 3000);
    } else {
      console.log('Loading default PLY file:', DEFAULT_PLY_URL);
      setSplatUrl(DEFAULT_PLY_URL);
    }
  }, []);

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative' }}>
      {/* Fullscreen viewer with overlay */}
      <SplatViewer splatUrl={splatUrl} />

      {/* Top-left overlay with title and controls */}
      <div style={{
        position: 'absolute',
        top: '20px',
        left: '20px',
        background: 'rgba(0, 0, 0, 0.7)',
        color: 'white',
        padding: '15px 20px',
        borderRadius: '8px',
        backdropFilter: 'blur(10px)',
        maxWidth: '400px',
        zIndex: 1000,
      }}>
        <h1 style={{ margin: '0 0 10px 0', fontSize: '1.5em' }}>
          3D Gaussian Splatting Viewer
        </h1>
        <p style={{ margin: '0 0 10px 0', fontSize: '0.9em', color: '#ccc' }}>
          <strong>Controls:</strong> WASDQE to move, Mouse to rotate
        </p>
        <p style={{ margin: '0 0 10px 0', fontSize: '0.8em', color: '#aaa' }}>
          Load custom files: <code style={{ background: 'rgba(255,255,255,0.1)', padding: '2px 4px', borderRadius: '3px' }}>?url=YOUR_PLY_URL</code>
        </p>
        <p style={{ margin: '0', fontSize: '0.75em', color: '#999' }}>
          Built on top of <a href="https://github.com/sparkjsdev/spark" target="_blank" rel="noopener noreferrer" style={{ color: '#64B5F6', textDecoration: 'none' }}>spark</a> by <a href="https://outside5sigma.com/" target="_blank" rel="noopener noreferrer" style={{ color: '#64B5F6', textDecoration: 'none' }}>Wentao</a>
        </p>
        {uploadStatus && (
          <div style={{ marginTop: '10px', fontSize: '0.85em', color: '#4CAF50' }}>
            {uploadStatus}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

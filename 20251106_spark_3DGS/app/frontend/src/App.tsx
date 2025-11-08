import { useState, useEffect } from 'react';
import { FileUpload } from './components/FileUpload';
import { SplatViewer } from './components/SplatViewer';
import './App.css';

function App() {
  const [splatUrl, setSplatUrl] = useState<string | undefined>();
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string>('');

  // Check for URL parameter on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const urlParam = params.get('url');

    if (urlParam) {
      console.log('Loading from URL parameter:', urlParam);
      setSplatUrl(urlParam);
      setUploadStatus('Loading from URL...');
      setTimeout(() => setUploadStatus(''), 3000);
    }
  }, []);

  // Test with existing file
  const loadTestFile = () => {
    console.log('Loading test file...');
    setSplatUrl('http://localhost:3000/files/upload-1762455220300-383288333.sog');
  };

  const handleFileSelect = async (file: File) => {
    console.log('File selected:', file.name, file.size, 'bytes');
    setUploading(true);
    setUploadStatus('Uploading and converting...');

    try {
      const formData = new FormData();
      formData.append('file', file);

      console.log('Uploading to server...');
      const response = await fetch('http://localhost:3000/upload', {
        method: 'POST',
        body: formData,
      });

      console.log('Response status:', response.status);
      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Server response:', data);
      setSplatUrl(data.url);

      if (data.converted) {
        setUploadStatus(`Converted to ${data.format.toUpperCase()}! (${data.compressionRatio}x compression)`);
      } else {
        setUploadStatus(`Upload complete! Serving as ${data.format.toUpperCase()} (conversion not available)`);
      }
      setTimeout(() => setUploadStatus(''), 3000);
    } catch (error) {
      console.error('Upload error:', error);
      setUploadStatus(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setTimeout(() => setUploadStatus(''), 5000);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{ width: '100vw', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <header style={{
        padding: '20px',
        background: '#282c34',
        color: 'white',
        borderBottom: '2px solid #444',
      }}>
        <h1 style={{ margin: 0 }}>3D Gaussian Splatting Viewer</h1>
        <p style={{ margin: '8px 0 0 0', fontSize: '14px', color: '#aaa' }}>
          Upload a .ply file or use ?url=YOUR_PLY_URL to load from external sources
        </p>
      </header>

      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: '20px', gap: '20px' }}>
        {!splatUrl ? (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ maxWidth: '600px', width: '100%' }}>
              <button
                onClick={loadTestFile}
                style={{
                  marginBottom: '20px',
                  padding: '10px 20px',
                  background: '#2196F3',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  width: '100%',
                }}
              >
                Load Test File (Already Converted)
              </button>
              <FileUpload onFileSelect={handleFileSelect} />
              {uploading && (
                <div style={{ marginTop: '20px', textAlign: 'center', color: '#666' }}>
                  {uploadStatus}
                </div>
              )}
              {uploadStatus && !uploading && (
                <div style={{
                  marginTop: '20px',
                  textAlign: 'center',
                  color: uploadStatus.startsWith('Error') ? '#f44336' : '#4CAF50',
                }}>
                  {uploadStatus}
                </div>
              )}
            </div>
          </div>
        ) : (
          <>
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
              <button
                onClick={() => setSplatUrl(undefined)}
                style={{
                  padding: '10px 20px',
                  background: '#4CAF50',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                }}
              >
                Upload New File
              </button>
              {uploadStatus && (
                <span style={{ color: '#4CAF50' }}>{uploadStatus}</span>
              )}
            </div>
            <div style={{ flex: 1, border: '1px solid #ccc', borderRadius: '8px', overflow: 'hidden' }}>
              <SplatViewer splatUrl={splatUrl} />
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default App;

import React, { useEffect, useState } from 'react';

interface DeploymentStatusProps {
  deploymentId: string;
  onComplete?: (result: DeploymentResult) => void;
  githubUser?: {
    login: string;
    name?: string;
    avatar_url: string;
  };
}

interface DeploymentResult {
  success: boolean;
  repositoryUrl?: string;
  websiteUrl?: string;
  error?: string;
}

interface DeploymentStatusData {
  status: 'pending' | 'queued' | 'in_progress' | 'success' | 'failure';
  progress: number;
  message: string;
  repositoryUrl?: string;
  websiteUrl?: string;
  error?: string;
}

export function DeploymentStatus({ deploymentId, onComplete, githubUser }: DeploymentStatusProps) {
  const [status, setStatus] = useState<DeploymentStatusData>({
    status: 'pending',
    progress: 0,
    message: 'Initializing deployment...'
  });
  const [isPolling, setIsPolling] = useState(true);

  useEffect(() => {
    if (!deploymentId || !isPolling) return;

    const pollStatus = async () => {
      try {
        const token = localStorage.getItem('github_access_token');
        if (!token) {
          throw new Error('GitHub token not found');
        }

        const response = await fetch(`http://localhost:8000/api/github/deployment/${deploymentId}/status`, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });

        if (!response.ok) {
          throw new Error(`Failed to get deployment status: ${response.status}`);
        }

        const data = await response.json();
        
        setStatus({
          status: data.status,
          progress: data.progress_percentage,
          message: data.message,
          repositoryUrl: data.repository?.html_url,
          websiteUrl: data.pages_url,
          error: data.error_message
        });

        // Stop polling if deployment is complete or failed
        if (data.status === 'success' || data.status === 'failure') {
          setIsPolling(false);
          
          if (onComplete) {
            onComplete({
              success: data.status === 'success',
              repositoryUrl: data.repository?.html_url,
              websiteUrl: data.pages_url,
              error: data.error_message
            });
          }
        }
      } catch (error) {
        console.error('Failed to poll deployment status:', error);
        setStatus(prev => ({
          ...prev,
          status: 'failure',
          message: `Failed to get deployment status: ${error.message}`,
          error: error.message
        }));
        setIsPolling(false);
        
        if (onComplete) {
          onComplete({
            success: false,
            error: error.message
          });
        }
      }
    };

    // Poll immediately, then every 3 seconds
    pollStatus();
    const interval = setInterval(pollStatus, 3000);

    return () => clearInterval(interval);
  }, [deploymentId, isPolling, onComplete]);

  const getStatusIcon = () => {
    switch (status.status) {
      case 'pending':
      case 'queued':
        return '⏳';
      case 'in_progress':
        return '🚀';
      case 'success':
        return '✅';
      case 'failure':
        return '❌';
      default:
        return '⏳';
    }
  };

  const getStatusColor = () => {
    switch (status.status) {
      case 'pending':
      case 'queued':
        return '#6c757d';
      case 'in_progress':
        return '#007bff';
      case 'success':
        return '#28a745';
      case 'failure':
        return '#dc3545';
      default:
        return '#6c757d';
    }
  };

  return (
    <div className="deployment-status">
      <div className="status-header">
        <h3>
          <span className="status-icon">{getStatusIcon()}</span>
          Deployment Status
        </h3>
      </div>

      <div className="status-content">
        <div className="progress-section">
          <div className="progress-bar-container">
            <div 
              className="progress-bar"
              style={{ 
                width: `${status.progress}%`,
                backgroundColor: getStatusColor()
              }}
            />
          </div>
          <div className="progress-text">
            {status.progress}% - {status.message}
          </div>
        </div>

        {status.status === 'in_progress' && (
          <div className="deployment-steps">
            <div className={`step ${status.progress >= 20 ? 'completed' : 'active'}`}>
              <span className="step-icon">📁</span>
              Creating repository
            </div>
            <div className={`step ${status.progress >= 40 ? 'completed' : status.progress >= 20 ? 'active' : ''}`}>
              <span className="step-icon">📄</span>
              Uploading content
            </div>
            <div className={`step ${status.progress >= 60 ? 'completed' : status.progress >= 40 ? 'active' : ''}`}>
              <span className="step-icon">⚙️</span>
              Configuring GitHub Pages
            </div>
            <div className={`step ${status.progress >= 80 ? 'completed' : status.progress >= 60 ? 'active' : ''}`}>
              <span className="step-icon">🔨</span>
              Building website
            </div>
            <div className={`step ${status.progress >= 100 ? 'completed' : status.progress >= 80 ? 'active' : ''}`}>
              <span className="step-icon">🌐</span>
              Deploying to web
            </div>
          </div>
        )}

        {status.status === 'success' && (
          <div className="success-section">
            <h4>🎉 Deployment Successful!</h4>
            <p>Your paper website has been deployed successfully.</p>

            {githubUser && (
              <div className="success-user-info">
                <img
                  src={githubUser.avatar_url}
                  alt={githubUser.login}
                  className="success-avatar"
                />
                <div className="success-details">
                  <p className="success-name">{githubUser.name || githubUser.login}</p>
                  <p className="success-login">@{githubUser.login}</p>
                </div>
              </div>
            )}

            <div className="result-links">
              {status.repositoryUrl && (
                <a
                  href={status.repositoryUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn btn-secondary"
                >
                  📁 View Repository
                </a>
              )}

              {status.websiteUrl && (
                <a
                  href={status.websiteUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn btn-primary"
                >
                  🌐 View Website
                </a>
              )}
            </div>

            {status.websiteUrl && (
              <div className="website-info">
                <p><strong>Website URL:</strong></p>
                <code className="url-code">{status.websiteUrl}</code>
                <button
                  onClick={() => navigator.clipboard.writeText(status.websiteUrl!)}
                  className="copy-btn"
                  title="Copy URL"
                >
                  📋
                </button>
              </div>
            )}
          </div>
        )}

        {status.status === 'failure' && (
          <div className="error-section">
            <h4>❌ Deployment Failed</h4>
            <p>There was an error deploying your paper website.</p>
            
            {status.error && (
              <div className="error-details">
                <strong>Error details:</strong>
                <pre className="error-message">{status.error}</pre>
              </div>
            )}
            
            <div className="error-actions">
              <button 
                onClick={() => {
                  setIsPolling(true);
                  setStatus(prev => ({ ...prev, status: 'pending', progress: 0 }));
                }}
                className="btn btn-secondary"
              >
                🔄 Retry Deployment
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

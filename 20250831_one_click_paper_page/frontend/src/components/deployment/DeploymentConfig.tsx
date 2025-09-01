import React, { useState } from "react";
import type { PaperTemplate } from "../../types/github";

interface DeploymentConfigProps {
  templates: PaperTemplate[];
  onConfigChange: (config: DeploymentConfiguration) => void;
  initialConfig?: Partial<DeploymentConfiguration>;
  paperMetadata?: {
    title?: string;
    authors?: string[];
    abstract?: string;
  };
  githubUser?: {
    login: string;
    name?: string;
    avatar_url: string;
  };
  onBackToTemplate?: () => void;
  onDeploy?: () => void;
  canDeploy?: boolean;
}

export interface DeploymentConfiguration {
  repositoryName: string;
  template: string;
  paperTitle: string;
  paperAuthors: string[];
}

export function DeploymentConfig({
  templates,
  onConfigChange,
  initialConfig = {},

  githubUser,
  onBackToTemplate,
  onDeploy,
  canDeploy = false,
}: DeploymentConfigProps) {
  // Generate smart defaults from paper metadata
  const generateRepositoryName = (title: string) => {
    return title
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, "")
      .replace(/\s+/g, "-")
      .substring(0, 50)
      .replace(/^-+|-+$/g, "");
  };

  const [config, setConfig] = useState<DeploymentConfiguration>(() => {
    // Initialize with proper defaults from initialConfig
    return {
      repositoryName: initialConfig.repositoryName || "",
      template: initialConfig.template || "minimal-academic",
      paperTitle: initialConfig.paperTitle || "",
      paperAuthors: initialConfig.paperAuthors || [],
    };
  });

  // No longer needed - initialization happens in useState

  // Call onConfigChange with initial config on mount (only once)
  const hasCalledInitialConfig = React.useRef(false);
  React.useEffect(() => {
    if (!hasCalledInitialConfig.current) {
      onConfigChange(config);
      hasCalledInitialConfig.current = true;
    }
  }, [onConfigChange, config]);

  const [authorInput, setAuthorInput] = useState("");

  const updateConfig = (updates: Partial<DeploymentConfiguration>) => {
    const newConfig = { ...config, ...updates };
    setConfig(newConfig);
    onConfigChange(newConfig);
  };

  const addAuthor = () => {
    if (
      authorInput.trim() &&
      !config.paperAuthors.includes(authorInput.trim())
    ) {
      updateConfig({
        paperAuthors: [...config.paperAuthors, authorInput.trim()],
      });
      setAuthorInput("");
    }
  };

  const removeAuthor = (index: number) => {
    updateConfig({
      paperAuthors: config.paperAuthors.filter((_, i) => i !== index),
    });
  };

  const handleGenerateRepositoryName = () => {
    if (config.paperTitle) {
      const name = generateRepositoryName(config.paperTitle);
      updateConfig({ repositoryName: name });
    }
  };

  return (
    <div className="deployment-config">
      {/* Action Buttons at Top */}
      <div className="action-buttons-top">
        <button onClick={onBackToTemplate} className="btn btn-secondary">
          ← Back to Template
        </button>

        <button
          onClick={onDeploy}
          disabled={!canDeploy || !config.repositoryName}
          className="btn btn-primary"
        >
          🚀 Deploy to GitHub Pages
        </button>
      </div>

      <h3>🚀 Deployment Configuration</h3>

      {/* Deployment Information */}
      <div className="config-section">
        <div className="info-banner">
          <div className="info-icon">ℹ️</div>
          <div>
            <h4>Automatic GitHub Actions Deployment</h4>
            <p className="help-text">
              Your repository will be created with GitHub Actions workflows that
              automatically convert and deploy your paper when you push files.
            </p>
          </div>
        </div>
      </div>
      {/* Paper Information */}
      <div className="config-section">
        <h4>📄 Paper Information</h4>

        <div className="form-group">
          <label htmlFor="paperTitle">Paper Title</label>
          <input
            id="paperTitle"
            type="text"
            value={config.paperTitle}
            onChange={e => updateConfig({ paperTitle: e.target.value })}
            placeholder="Enter your paper title"
            className="form-input"
          />
        </div>

        <div className="form-group">
          <label htmlFor="authorInput">Authors</label>
          <div className="author-input-group">
            <input
              id="authorInput"
              type="text"
              value={authorInput}
              onChange={e => setAuthorInput(e.target.value)}
              onKeyDown={e => e.key === "Enter" && addAuthor()}
              placeholder="Enter author name and press Enter"
              className="form-input"
            />
            <button
              type="button"
              onClick={addAuthor}
              className="btn btn-secondary"
              disabled={!authorInput.trim()}
            >
              Add
            </button>
          </div>

          {config.paperAuthors.length > 0 && (
            <div className="author-tags">
              {config.paperAuthors.map((author, index) => (
                <span key={index} className="author-tag">
                  {author}
                  <button
                    type="button"
                    onClick={() => removeAuthor(index)}
                    className="remove-author"
                    aria-label={`Remove ${author}`}
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Repository Configuration */}
      <div className="config-section">
        <h4>📁 Repository Configuration</h4>

        <div className="form-group">
          <label htmlFor="repositoryName">Repository Name</label>
          <div className="repository-input-group">
            <input
              id="repositoryName"
              type="text"
              value={config.repositoryName}
              onChange={e => updateConfig({ repositoryName: e.target.value })}
              placeholder="my-paper-website"
              className="form-input"
              pattern="[a-zA-Z0-9._\-]+"
            />
            <button
              type="button"
              onClick={handleGenerateRepositoryName}
              className="btn btn-secondary"
              disabled={!config.paperTitle}
              title="Generate from paper title"
            >
              Generate
            </button>
          </div>
          <p className="help-text">
            Repository name must contain only letters, numbers, hyphens,
            underscores, and periods.
          </p>
        </div>
      </div>

      {/* Template Selection */}
      <div className="config-section">
        <h4>🎨 Website Template</h4>

        <div className="template-grid">
          {templates.map(template => (
            <div
              key={template.id}
              className={`template-card ${config.template === template.id ? "selected" : ""}`}
              onClick={() => updateConfig({ template: template.id })}
            >
              <div className="template-header">
                <h5>{template.name}</h5>
                {config.template === template.id && (
                  <span className="selected-badge">✓</span>
                )}
              </div>
              <p className="template-description">{template.description}</p>
              <div className="template-features">
                {template.features.slice(0, 3).map((feature, index) => (
                  <span key={index} className="feature-tag">
                    {feature}
                  </span>
                ))}
                {template.features.length > 3 && (
                  <span className="feature-tag more">
                    +{template.features.length - 3} more
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Deployment Preview */}
      <div className="config-section deployment-preview">
        <h4>🌐 Deployment Preview</h4>
        <div className="preview-info">
          {githubUser && (
            <div className="github-user-info">
              <img
                src={githubUser.avatar_url}
                alt={githubUser.login}
                className="user-avatar"
              />
              <div className="user-details">
                <p className="user-name">
                  {githubUser.name || githubUser.login}
                </p>
                <p className="user-login">@{githubUser.login}</p>
              </div>
            </div>
          )}

          <div className="deployment-urls">
            <p>
              <strong>Repository:</strong> github.com/
              {githubUser?.login || "[username]"}/
              {config.repositoryName || "repository-name"}
            </p>
            <p>
              <strong>Website URL:</strong> https://
              {githubUser?.login || "[username]"}.github.io/
              {config.repositoryName || "repository-name"}
            </p>
            <p>
              <strong>Template:</strong>{" "}
              {templates.find(t => t.id === config.template)?.name ||
                "Minimal Academic"}
            </p>
            <p>
              <strong>Visibility:</strong> Public
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

import React, { useState } from 'react';
import { GitHubAuth } from './components/auth/GitHubAuth';
import { FileUpload } from './components/upload/FileUpload';
import { TemplateSelector } from './components/templates/TemplateSelector';
import { ConversionProgress } from './components/conversion/ConversionProgress';
import { ConversionModeSelector } from './components/conversion/ConversionModeSelector';
import { DeploymentConfig } from './components/deployment/DeploymentConfig';
import { DeploymentStatus } from './components/deployment/DeploymentStatus';
import type { PaperTemplate, GitHubUser } from './types/github';
import type { ConversionResult, ConversionMode } from './lib/api/conversion';
import type { DeploymentConfiguration } from './components/deployment/DeploymentConfig';
import { useGitHubAuth } from './lib/github/auth';
import { useConversion } from './lib/api/conversion';
import { useDeployment } from './lib/api/deployment';
import './App.css';
import './components/deployment/deployment.css';

type Step = 'auth' | 'upload' | 'template' | 'convert' | 'configure' | 'deploy';

function App() {
  const [currentStep, setCurrentStep] = useState<Step>('auth');
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<PaperTemplate | null>(null);
  const [overleafUrl, setOverleafUrl] = useState<string>('');
  const [conversionMode, setConversionMode] = useState<ConversionMode>('auto');
  const [repositoryName] = useState<string>(''); // Keep for backward compatibility
  const [deploymentConfig, setDeploymentConfig] = useState<DeploymentConfiguration | null>(null);
  const [deploymentId, setDeploymentId] = useState<string | null>(null);
  const [templates, setTemplates] = useState<any[]>([]);
  const { isAuthenticated, user } = useGitHubAuth();
  const conversion = useConversion();
  const deployment = useDeployment();

  // Auto-advance to upload step if already authenticated
  React.useEffect(() => {
    if (isAuthenticated && currentStep === 'auth') {
      setCurrentStep('upload');
    }
  }, [isAuthenticated, currentStep]);

  const handleAuthSuccess = (user: GitHubUser, _token: string) => {
    console.log('Authentication successful:', user);
    setCurrentStep('upload');
  };

  const handleFilesSelected = (files: File[]) => {
    setSelectedFiles(files);
    if (files.length > 0) {
      setCurrentStep('template');
    }
  };

  const handleOverleafUrl = (url: string) => {
    setOverleafUrl(url);
    setCurrentStep('template');
  };

  const handleTemplateSelected = (template: PaperTemplate) => {
    setSelectedTemplate(template);
    setCurrentStep('convert');

    // Start conversion with the first selected file
    if (selectedFiles.length > 0) {
      conversion.startConversion(
        selectedFiles[0],
        template.id,
        conversionMode,
        repositoryName || undefined
      );
    }
  };

  const handleConversionComplete = (result: ConversionResult) => {
    console.log('Conversion completed:', result);
    setCurrentStep('configure');
  };

  const handleConversionCancel = () => {
    conversion.reset();
    setCurrentStep('template');
  };

  const handleConversionRetry = () => {
    if (selectedFiles.length > 0 && selectedTemplate) {
      conversion.startConversion(
        selectedFiles[0],
        selectedTemplate.id,
        conversionMode,
        repositoryName || undefined
      );
    }
  };

  // Load templates on mount
  React.useEffect(() => {
    const loadTemplates = async () => {
      try {
        const templateList = await deployment.getTemplates();
        setTemplates(templateList);
      } catch (error) {
        console.error('Failed to load templates:', error);
      }
    };

    loadTemplates();
  }, [deployment]);

  const handleDeploymentConfigChange = (config: DeploymentConfiguration) => {
    setDeploymentConfig(config);
  };

  const handleStartDeployment = async () => {
    if (!deploymentConfig || !conversion.result) {
      return;
    }

    try {
      setCurrentStep('deploy');

      // Create repository
      const repoResponse = await deployment.createRepository({
        name: deploymentConfig.repositoryName,
        description: `Academic paper website: ${deploymentConfig.paperTitle || 'Untitled'}`,
        private: false, // Always public for open science
        template: deploymentConfig.template,
        conversion_job_id: conversion.jobId || '',
      });

      setDeploymentId(repoResponse.deployment_id);

      // Deploy content
      await deployment.deployContent(repoResponse.deployment_id, {
        repository_name: deploymentConfig.repositoryName,
        template: deploymentConfig.template,
        paper_title: deploymentConfig.paperTitle,
        paper_authors: deploymentConfig.paperAuthors,
      });

    } catch (error) {
      console.error('Deployment failed:', error);
      // Handle error - could show error message to user
    }
  };

  const handleDeploymentComplete = (result: any) => {
    console.log('Deployment completed:', result);
    // Could show success message or redirect
  };

  const getStepNumber = (step: Step): number => {
    const steps: Step[] = ['auth', 'upload', 'template', 'convert', 'configure', 'deploy'];
    return steps.indexOf(step) + 1;
  };

  const renderStepIndicator = () => {
    const steps: { key: Step; label: string }[] = [
      { key: 'auth', label: 'Authenticate' },
      { key: 'upload', label: 'Upload' },
      { key: 'template', label: 'Template' },
      { key: 'configure', label: 'Configure' },
      { key: 'deploy', label: 'Deploy' },
    ];

    return (
      <div className="flex items-center justify-center mb-8">
        {steps.map((step, index) => {
          const isActive = currentStep === step.key;
          const isCompleted = getStepNumber(currentStep) > index + 1;
          const isAccessible = getStepNumber(currentStep) >= index + 1;

          return (
            <React.Fragment key={step.key}>
              <div className="flex flex-col items-center">
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium ${
                    isCompleted
                      ? 'bg-green-500 text-white'
                      : isActive
                      ? 'bg-blue-500 text-white'
                      : isAccessible
                      ? 'bg-gray-300 text-gray-700'
                      : 'bg-gray-200 text-gray-400'
                  }`}
                >
                  {isCompleted ? (
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                  ) : (
                    index + 1
                  )}
                </div>
                <span
                  className={`mt-2 text-xs font-medium ${
                    isActive ? 'text-blue-600' : 'text-gray-500'
                  }`}
                >
                  {step.label}
                </span>
              </div>
              {index < steps.length - 1 && (
                <div
                  className={`w-16 h-0.5 mx-4 ${
                    getStepNumber(currentStep) > index + 1
                      ? 'bg-green-500'
                      : 'bg-gray-300'
                  }`}
                />
              )}
            </React.Fragment>
          );
        })}
      </div>
    );
  };

  const renderCurrentStep = () => {
    switch (currentStep) {
      case 'auth':
        return <GitHubAuth onAuthSuccess={handleAuthSuccess} />;

      case 'upload':
        return (
          <FileUpload
            onFilesSelected={handleFilesSelected}
            onOverleafUrl={handleOverleafUrl}
          />
        );

      case 'template':
        return (
          <div className="w-full max-w-4xl mx-auto p-6">
            <ConversionModeSelector
              mode={conversionMode}
              onModeChange={setConversionMode}
            />
            <TemplateSelector
              onTemplateSelected={handleTemplateSelected}
              selectedTemplate={selectedTemplate || undefined}
            />
          </div>
        );

      case 'convert':
        return (
          <div className="text-center p-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">Converting Document</h2>
            <ConversionProgress
              isConverting={conversion.isConverting}
              phase={conversion.phase}
              stage={conversion.stage}
              error={conversion.error}
              result={conversion.result}
              onCancel={handleConversionCancel}
              onRetry={handleConversionRetry}
              onContinue={handleConversionComplete}
            />
          </div>
        );

      case 'configure':
        return (
          <div className="max-w-4xl mx-auto">
            <DeploymentConfig
              templates={templates}
              onConfigChange={handleDeploymentConfigChange}
              paperMetadata={conversion.result?.metadata ? {
                title: conversion.result.metadata.title,
                authors: conversion.result.metadata.authors,
                abstract: conversion.result.metadata.abstract,
              } : undefined}
              githubUser={user ? {
                login: user.login,
                name: user.name || undefined,
                avatar_url: user.avatar_url,
              } : undefined}
              initialConfig={{
                repositoryName: repositoryName || '',
                template: selectedTemplate?.id || 'minimal-academic',
                paperTitle: '',
                paperAuthors: [],
                autoDeployEnabled: true,
              }}
            />

            <div className="text-center mt-8">
              <button
                onClick={() => setCurrentStep('template')}
                className="bg-gray-500 text-white px-6 py-3 rounded-lg hover:bg-gray-600 mr-4"
              >
                ← Back to Template
              </button>

              {deploymentConfig?.autoDeployEnabled ? (
                <button
                  onClick={handleStartDeployment}
                  disabled={!deploymentConfig || !deploymentConfig.repositoryName}
                  className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  🚀 Deploy to GitHub Pages
                </button>
              ) : (
                <button
                  onClick={() => setCurrentStep('deploy')}
                  className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700"
                >
                  Continue →
                </button>
              )}
            </div>
          </div>
        );

      case 'deploy':
        return (
          <div className="max-w-4xl mx-auto">
            {deploymentId ? (
              <DeploymentStatus
                deploymentId={deploymentId}
                onComplete={handleDeploymentComplete}
                githubUser={user ? {
                  login: user.login,
                  name: user.name || undefined,
                  avatar_url: user.avatar_url,
                } : undefined}
              />
            ) : (
              <div className="text-center p-8">
                <h2 className="text-2xl font-bold text-gray-800 mb-4">Manual Deployment</h2>
                <p className="text-gray-600 mb-6">
                  Your conversion is complete! You can now manually create a repository and deploy your content.
                </p>

                <div className="bg-blue-50 p-6 rounded-lg mb-6">
                  <h3 className="font-semibold text-blue-800 mb-4">Conversion Summary:</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-left">
                    <div>
                      <p className="text-blue-700"><strong>Files:</strong> {selectedFiles.length} uploaded</p>
                      {overleafUrl && <p className="text-blue-700"><strong>Overleaf:</strong> {overleafUrl}</p>}
                      <p className="text-blue-700"><strong>Template:</strong> {selectedTemplate?.name}</p>
                    </div>
                    <div>
                      <p className="text-blue-700"><strong>User:</strong> {user?.login}</p>
                      <p className="text-blue-700"><strong>Mode:</strong> {conversionMode}</p>
                      {conversion.result && (
                        <p className="text-blue-700"><strong>Status:</strong> Conversion Complete</p>
                      )}
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <button
                    onClick={() => setCurrentStep('configure')}
                    className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700"
                  >
                    ← Back to Configure Deployment
                  </button>

                  {conversion.result && (
                    <div className="mt-4">
                      <p className="text-sm text-gray-600 mb-2">
                        Download your converted files to deploy manually:
                      </p>
                      <button className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700">
                        📥 Download Converted Files
                      </button>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-lg">📄</span>
              </div>
              <h1 className="text-xl font-bold text-gray-900">Paper to Website</h1>
            </div>
            {isAuthenticated && user && (
              <div className="flex items-center space-x-3">
                <img
                  src={user.avatar_url}
                  alt={user.login}
                  className="w-8 h-8 rounded-full"
                />
                <span className="text-sm text-gray-700">{user.login}</span>
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderStepIndicator()}
        {renderCurrentStep()}
      </main>

      <footer className="bg-white border-t mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-gray-500 text-sm">
            Convert your academic papers into beautiful websites with GitHub Pages
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;

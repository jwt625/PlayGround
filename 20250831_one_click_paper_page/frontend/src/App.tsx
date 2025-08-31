import React, { useState } from 'react';
import { GitHubAuth } from './components/auth/GitHubAuth';
import { FileUpload } from './components/upload/FileUpload';
import { TemplateSelector } from './components/templates/TemplateSelector';
import { PaperTemplate, GitHubUser } from './types/github';
import { useGitHubAuth } from './lib/github/auth';
import './App.css';

type Step = 'auth' | 'upload' | 'template' | 'configure' | 'deploy';

function App() {
  const [currentStep, setCurrentStep] = useState<Step>('auth');
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<PaperTemplate | null>(null);
  const [overleafUrl, setOverleafUrl] = useState<string>('');
  const { isAuthenticated, user } = useGitHubAuth();

  // Auto-advance to upload step if already authenticated
  React.useEffect(() => {
    if (isAuthenticated && currentStep === 'auth') {
      setCurrentStep('upload');
    }
  }, [isAuthenticated, currentStep]);

  const handleAuthSuccess = (user: GitHubUser, token: string) => {
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
    setCurrentStep('configure');
  };

  const getStepNumber = (step: Step): number => {
    const steps: Step[] = ['auth', 'upload', 'template', 'configure', 'deploy'];
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
          <TemplateSelector
            onTemplateSelected={handleTemplateSelected}
            selectedTemplate={selectedTemplate || undefined}
          />
        );

      case 'configure':
        return (
          <div className="text-center p-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-4">Configuration</h2>
            <p className="text-gray-600 mb-4">
              Configuration step coming soon! This will allow you to customize your paper metadata,
              theme settings, and deployment options.
            </p>
            <button
              onClick={() => setCurrentStep('deploy')}
              className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700"
            >
              Continue to Deploy
            </button>
          </div>
        );

      case 'deploy':
        return (
          <div className="text-center p-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-4">Deploy</h2>
            <p className="text-gray-600 mb-4">
              Deployment step coming soon! This will create your GitHub repository,
              upload your files, and trigger the conversion workflow.
            </p>
            <div className="bg-blue-50 p-4 rounded-lg">
              <h3 className="font-semibold text-blue-800 mb-2">Summary:</h3>
              <ul className="text-left text-blue-700 space-y-1">
                <li>• Files: {selectedFiles.length} uploaded</li>
                {overleafUrl && <li>• Overleaf: {overleafUrl}</li>}
                <li>• Template: {selectedTemplate?.name}</li>
                <li>• User: {user?.login}</li>
              </ul>
            </div>
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

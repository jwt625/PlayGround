import React, { useState } from "react";
import { GitHubAuth } from "./components/auth/GitHubAuth";
import { FileUpload } from "./components/upload/FileUpload";
import { TemplateSelector } from "./components/templates/TemplateSelector";
import { ConversionProgress } from "./components/conversion/ConversionProgress";
import { ConversionModeSelector } from "./components/conversion/ConversionModeSelector";
import { DeploymentConfig } from "./components/deployment/DeploymentConfig";
import { DeploymentStatus } from "./components/deployment/DeploymentStatus";
import type { PaperTemplate, GitHubUser } from "./types/github";
import type { ConversionResult, ConversionMode } from "./lib/api/conversion";
import type { DeploymentConfiguration } from "./components/deployment/DeploymentConfig";
import { useGitHubAuth } from "./lib/github/auth";
import { useConversion } from "./lib/api/conversion";
import { useDeployment } from "./lib/api/deployment";
import "./App.css";
import "./components/deployment/deployment.css";

type Step = "auth" | "upload" | "template" | "convert" | "configure" | "deploy";

function App() {
  const [currentStep, setCurrentStep] = useState<Step>("auth");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [selectedTemplate, setSelectedTemplate] =
    useState<PaperTemplate | null>(null);
  const [, setOverleafUrl] = useState<string>("");
  const [conversionMode, setConversionMode] = useState<ConversionMode>("auto");
  const [repositoryName] = useState<string>(""); // Keep for backward compatibility
  const [deploymentConfig, setDeploymentConfig] =
    useState<DeploymentConfiguration | null>(null);
  const [deploymentId, setDeploymentId] = useState<string | null>(null);
  const [templates, setTemplates] = useState<PaperTemplate[]>([]);
  const [, setTestDeploymentResult] = useState<unknown>(null);
  const [isTestingDeployment, setIsTestingDeployment] = useState(false);
  const [tokenScopes, setTokenScopes] = useState<string[] | null>(null);
  const [isCheckingScopes, setIsCheckingScopes] = useState(false);
  const { isAuthenticated, user, token } = useGitHubAuth();
  const conversion = useConversion();
  const deployment = useDeployment();

  // Auto-advance to upload step if already authenticated
  React.useEffect(() => {
    if (isAuthenticated && currentStep === "auth") {
      setCurrentStep("upload");
    }
  }, [isAuthenticated, currentStep]);

  const handleAuthSuccess = (user: GitHubUser) => {
    console.log("Authentication successful:", user);
    setCurrentStep("upload");
  };

  const handleTestDeployment = async () => {
    if (!token) {
      alert("Please authenticate with GitHub first");
      return;
    }

    setIsTestingDeployment(true);
    setTestDeploymentResult(null);

    try {
      // Use the optimized deployment approach
      const response = await fetch(
        "http://localhost:8000/api/github/test-deploy-optimized",
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
        }
      );

      if (!response.ok) {
        const error = await response.text();
        throw new Error(`Test deployment failed: ${error}`);
      }

      const result = await response.json();
      setTestDeploymentResult(result);

      // Show success message with optimized approach benefits
      alert(
        `🚀 Test deployment successful!\n\nRepository: ${result.repository.name}\nURL: ${result.repository.url}\nPages URL: ${result.repository.pages_url}\n\nThis uses our optimized approach with:\n✅ No fork security issues\n✅ Faster deployment\n✅ Automatic workflow setup\n\nCheck the repository - it should deploy automatically!`
      );
    } catch (error) {
      console.error("Test deployment failed:", error);
      alert(
        `Test deployment failed: ${error instanceof Error ? error.message : String(error)}`
      );
    } finally {
      setIsTestingDeployment(false);
    }
  };

  const handleTestDualDeployment = async () => {
    if (!token) {
      alert("Please authenticate with GitHub first");
      return;
    }

    setIsTestingDeployment(true);
    setTestDeploymentResult(null);

    try {
      const response = await fetch(
        "http://localhost:8000/api/github/test-dual-deploy",
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
        }
      );

      if (!response.ok) {
        const error = await response.text();
        throw new Error(`Dual deployment test failed: ${error}`);
      }

      const result = await response.json();
      setTestDeploymentResult(result);

      // Show success message with simplified dual deployment benefits
      alert(
        `🏠 Simplified dual deployment successful!\n\n` +
        `Repository: ${result.standalone_repo.name}\n` +
        `Website URL: ${result.standalone_url}\n\n` +
        `Benefits:\n` +
        `✅ Jekyll workflow included in initial commit\n` +
        `✅ GitHub Pages enabled automatically\n` +
        `✅ Single commit - no timing issues\n` +
        `✅ Automatic builds on every push\n\n` +
        `Your paper will be built and deployed automatically!`
      );
    } catch (error) {
      console.error("Dual deployment test failed:", error);
      alert(
        `Dual deployment test failed: ${error instanceof Error ? error.message : String(error)}`
      );
    } finally {
      setIsTestingDeployment(false);
    }
  };

  const handleCheckTokenScopes = async () => {
    if (!token) {
      alert("Please authenticate with GitHub first");
      return;
    }

    setIsCheckingScopes(true);
    try {
      const response = await fetch("http://localhost:8000/api/github/token/scopes", {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to check scopes: ${response.status}`);
      }

      const data = await response.json();
      setTokenScopes(data.scopes);
      console.log("Token scopes:", data.scopes);

      const hasWorkflowScope = data.scopes.includes('workflow');
      const scopesList = data.scopes.join(', ');

      alert(
        `Token Scopes: ${scopesList}\n\n` +
        `Workflow scope: ${hasWorkflowScope ? '✅ Present' : '❌ Missing'}\n\n` +
        `${hasWorkflowScope ? 'Ready for .github/workflows deployment!' : 'Need to re-authenticate to get workflow scope'}`
      );
    } catch (error) {
      console.error("Error checking token scopes:", error);
      alert(`Failed to check token scopes: ${error}`);
    } finally {
      setIsCheckingScopes(false);
    }
  };

  const handleFilesSelected = (files: File[]) => {
    setSelectedFiles(files);
    if (files.length > 0) {
      setCurrentStep("template");
    }
  };

  const handleOverleafUrl = (url: string) => {
    setOverleafUrl(url);
    setCurrentStep("template");
  };

  const handleTemplateSelected = (template: PaperTemplate) => {
    setSelectedTemplate(template);
    setCurrentStep("convert");

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
    console.log("Conversion completed:", result);
    setCurrentStep("configure");
  };

  const handleConversionCancel = () => {
    conversion.reset();
    setCurrentStep("template");
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
        console.error("Failed to load templates:", error);
      }
    };

    loadTemplates();
  }, [deployment]); // Include deployment dependency

  const handleDeploymentConfigChange = React.useCallback(
    (config: DeploymentConfiguration) => {
      setDeploymentConfig(config);
    },
    []
  );

  // Generate repository name from paper title
  const generateRepositoryName = (title: string) => {
    return title
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, "")
      .replace(/\s+/g, "-")
      .substring(0, 50)
      .replace(/^-+|-+$/g, "");
  };

  // Get default repository name
  const defaultRepositoryName = React.useMemo(() => {
    const title = conversion.result?.metadata?.title;
    return title ? generateRepositoryName(title) : "";
  }, [conversion.result?.metadata?.title]);

  const handleStartDeployment = async () => {
    if (!deploymentConfig || !conversion.result || !user) {
      return;
    }

    try {
      // Set deployment step immediately
      setCurrentStep("deploy");

      // Trigger full automated deployment (repository creation + content deployment)
      const token = localStorage.getItem("github_access_token");
      if (!token) {
        throw new Error("GitHub token not found. Please authenticate first.");
      }

      const response = await fetch("http://localhost:8000/api/github/deploy", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          conversion_job_id: conversion.jobId,
          repository_name: deploymentConfig.repositoryName,
          template: deploymentConfig.template,
          paper_title: deploymentConfig.paperTitle,
          paper_authors: deploymentConfig.paperAuthors,
        }),
      });

      if (!response.ok) {
        let errorMessage = "Deployment failed";
        try {
          const error = await response.json();
          errorMessage = error.detail || errorMessage;
        } catch {
          // If response is not JSON (e.g., HTML error page), use status text
          errorMessage = `${response.status}: ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const result = await response.json();
      setDeploymentId(result.deployment_id);
    } catch (error) {
      console.error("Deployment failed:", error);
      alert(`Deployment failed: ${error}`);
      setCurrentStep("configure"); // Go back to config on error
    }
  };

  const handleDeploymentComplete = (result: unknown) => {
    console.log("Deployment completed:", result);
    // Could show success message or redirect
  };

  const getStepNumber = (step: Step): number => {
    const steps: Step[] = [
      "auth",
      "upload",
      "template",
      "convert",
      "configure",
      "deploy",
    ];
    return steps.indexOf(step) + 1;
  };

  const renderStepIndicator = () => {
    const steps: { key: Step; label: string }[] = [
      { key: "auth", label: "Authenticate" },
      { key: "upload", label: "Upload" },
      { key: "template", label: "Template" },
      { key: "configure", label: "Configure" },
      { key: "deploy", label: "Deploy" },
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
                      ? "bg-green-500 text-white"
                      : isActive
                        ? "bg-blue-500 text-white"
                        : isAccessible
                          ? "bg-gray-300 text-gray-700"
                          : "bg-gray-200 text-gray-400"
                  }`}
                >
                  {isCompleted ? (
                    <svg
                      className="w-5 h-5"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
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
                    isActive ? "text-blue-600" : "text-gray-500"
                  }`}
                >
                  {step.label}
                </span>
              </div>
              {index < steps.length - 1 && (
                <div
                  className={`w-16 h-0.5 mx-4 ${
                    getStepNumber(currentStep) > index + 1
                      ? "bg-green-500"
                      : "bg-gray-300"
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
      case "auth":
        return <GitHubAuth onAuthSuccess={handleAuthSuccess} />;

      case "upload":
        return (
          <FileUpload
            onFilesSelected={handleFilesSelected}
            onOverleafUrl={handleOverleafUrl}
          />
        );

      case "template":
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

      case "convert":
        return (
          <div className="text-center p-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">
              Converting Document
            </h2>
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

      case "configure":
        return (
          <div className="max-w-4xl mx-auto">
            <DeploymentConfig
              templates={templates}
              onConfigChange={handleDeploymentConfigChange}
              paperMetadata={
                conversion.result?.metadata
                  ? {
                      title: conversion.result.metadata.title,
                      authors: conversion.result.metadata.authors,
                      abstract: conversion.result.metadata.abstract,
                    }
                  : undefined
              }
              githubUser={
                user
                  ? {
                      login: user.login,
                      name: user.name || undefined,
                      avatar_url: user.avatar_url,
                    }
                  : undefined
              }
              initialConfig={{
                repositoryName: defaultRepositoryName || "",
                template: selectedTemplate?.id || "minimal-academic",
                paperTitle: conversion.result?.metadata?.title || "",
                paperAuthors: conversion.result?.metadata?.authors || [],
              }}
              onBackToTemplate={() => setCurrentStep("template")}
              onDeploy={handleStartDeployment}
              canDeploy={
                !!(
                  deploymentConfig?.repositoryName && deploymentConfig?.template
                )
              }
            />
          </div>
        );

      case "deploy":
        return (
          <div className="max-w-4xl mx-auto">
            <DeploymentStatus
              deploymentId={deploymentId || "pending"}
              onComplete={handleDeploymentComplete}
              githubUser={
                user
                  ? {
                      login: user.login,
                      name: user.name || undefined,
                      avatar_url: user.avatar_url,
                    }
                  : undefined
              }
            />
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
              <h1 className="text-xl font-bold text-gray-900">
                Paper to Website
              </h1>
            </div>
            {isAuthenticated && user && (
              <div className="flex items-center space-x-3">
                <div className="flex gap-2">
                  <button
                    onClick={handleTestDeployment}
                    disabled={isTestingDeployment}
                    className="px-3 py-1 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isTestingDeployment ? "Testing..." : "🚀 Test Deploy"}
                  </button>
                  <button
                    onClick={handleTestDualDeployment}
                    disabled={isTestingDeployment}
                    className="px-3 py-1 text-sm bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isTestingDeployment ? "Testing..." : "🏠 Dual Deploy"}
                  </button>
                  <button
                    onClick={handleCheckTokenScopes}
                    disabled={isCheckingScopes}
                    className="px-3 py-1 text-sm bg-orange-600 text-white rounded-md hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isCheckingScopes ? "Checking..." : "🔑 Check Scopes"}
                  </button>
                </div>
                {tokenScopes && (
                  <div className="text-xs text-gray-600 flex items-center">
                    {tokenScopes.includes('workflow') ? (
                      <span className="text-green-600">✅ Workflow</span>
                    ) : (
                      <span className="text-red-600">❌ No Workflow</span>
                    )}
                  </div>
                )}
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
            Convert your academic papers into beautiful websites with GitHub
            Pages
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;

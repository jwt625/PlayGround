/**
 * TypeScript types for GitHub API integration
 */

export interface GitHubUser {
  id: number;
  login: string;
  name: string | null;
  email: string | null;
  avatar_url: string;
  html_url: string;
}

export interface GitHubRepository {
  id: number;
  name: string;
  full_name: string;
  description: string | null;
  html_url: string;
  clone_url: string;
  ssh_url: string;
  default_branch: string;
  private: boolean;
  owner: GitHubUser;
  created_at: string;
  updated_at: string;
}

export interface CreateRepositoryRequest {
  name: string;
  description?: string;
  private?: boolean;
  auto_init?: boolean;
  gitignore_template?: string;
  license_template?: string;
}

export interface FileUpload {
  path: string;
  content: string;
  encoding?: "utf-8" | "base64";
}

export interface CommitRequest {
  message: string;
  files: FileUpload[];
  branch?: string;
}

export interface GitHubCommit {
  sha: string;
  url: string;
  html_url: string;
  message: string;
  author: {
    name: string;
    email: string;
    date: string;
  };
}

export interface WorkflowRun {
  id: number;
  name: string;
  status: "queued" | "in_progress" | "completed";
  conclusion:
    | "success"
    | "failure"
    | "neutral"
    | "cancelled"
    | "skipped"
    | "timed_out"
    | "action_required"
    | null;
  html_url: string;
  created_at: string;
  updated_at: string;
}

export interface PaperTemplate {
  id: string;
  name: string;
  description: string;
  preview_url?: string;
  repository_url: string;
  features: string[];
}

export interface PaperConfig {
  title: string;
  authors: string[];
  date?: string;
  theme: string;
  template: string;
  overleaf_url?: string;
  overleaf_token?: string;
}

export interface ConversionJob {
  id: string;
  repository: GitHubRepository;
  status: "pending" | "processing" | "completed" | "failed";
  workflow_run?: WorkflowRun;
  created_at: string;
  completed_at?: string;
  error_message?: string;
  output_url?: string;
}

export interface GitHubAppConfig {
  app_id: string;
  client_id: string;
  client_secret: string;
  private_key: string;
  webhook_secret: string;
}

export interface OAuthTokenResponse {
  access_token: string;
  token_type: string;
  scope: string;
}

export interface GitHubError {
  message: string;
  documentation_url?: string;
  errors?: Array<{
    resource: string;
    field: string;
    code: string;
  }>;
}

// Simplified: Use only minimal-academic template
export const DEFAULT_TEMPLATE: PaperTemplate = {
  id: "minimal-academic",
  name: "Minimal Academic",
  description: "Lightweight, fast-loading academic template focused on content. Ideal for single papers.",
  repository_url: "https://github.com/pages-themes/minimal",
  features: [
    "Lightweight and fast",
    "Clean typography",
    "Math support",
    "Mobile responsive",
    "Easy customization",
    "GitHub Pages compatible"
  ],
};

// Keep for backward compatibility but only include the default template
export const AVAILABLE_TEMPLATES: PaperTemplate[] = [DEFAULT_TEMPLATE];

// File type detection
export type SupportedFileType = "pdf" | "docx" | "latex" | "zip" | "overleaf";

export interface FileTypeDetection {
  type: SupportedFileType;
  confidence: number;
  file?: File;
  overleaf_url?: string;
}

// Conversion pipeline status
export interface ConversionStep {
  name: string;
  status: "pending" | "running" | "completed" | "failed";
  message?: string;
  duration?: number;
}

export interface ConversionPipeline {
  steps: ConversionStep[];
  current_step: number;
  overall_status: "pending" | "running" | "completed" | "failed";
  started_at?: string;
  completed_at?: string;
}

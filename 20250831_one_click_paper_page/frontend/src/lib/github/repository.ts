/**
 * GitHub repository management module
 */

import { Octokit } from '@octokit/rest';
import {
  GitHubRepository,
  CreateRepositoryRequest,
  FileUpload,
  CommitRequest,
  GitHubCommit,
  WorkflowRun,
  PaperTemplate,
  AVAILABLE_TEMPLATES,
} from '../../types/github';

export class GitHubRepositoryManager {
  private octokit: Octokit;

  constructor(accessToken: string) {
    this.octokit = new Octokit({
      auth: accessToken,
    });
  }

  /**
   * Create a new repository from template
   */
  async createRepositoryFromTemplate(
    request: CreateRepositoryRequest,
    templateId: string
  ): Promise<GitHubRepository> {
    try {
      const template = AVAILABLE_TEMPLATES.find(t => t.id === templateId);
      if (!template) {
        throw new Error(`Template ${templateId} not found`);
      }

      // Create the repository
      const { data: repo } = await this.octokit.repos.createForAuthenticatedUser({
        name: request.name,
        description: request.description || `Academic paper website - ${request.name}`,
        private: request.private || false,
        auto_init: true,
        gitignore_template: 'Node',
        license_template: 'mit',
      });

      // Copy template files
      await this.copyTemplateFiles(repo.owner.login, repo.name, template);

      // Enable GitHub Pages
      await this.enableGitHubPages(repo.owner.login, repo.name);

      return repo as GitHubRepository;
    } catch (error) {
      console.error('Repository creation error:', error);
      throw error;
    }
  }

  /**
   * Copy template files to the new repository
   */
  private async copyTemplateFiles(
    owner: string,
    repo: string,
    template: PaperTemplate
  ): Promise<void> {
    try {
      // Get the workflow template from our local template directory
      const workflowContent = await this.getWorkflowTemplate();
      const configContent = this.getDefaultConfig(template.id);
      const readmeContent = this.getReadmeTemplate(template);

      // Create the files
      const files: FileUpload[] = [
        {
          path: '.github/workflows/convert-and-deploy.yml',
          content: workflowContent,
        },
        {
          path: 'paper-config.json',
          content: configContent,
        },
        {
          path: 'README.md',
          content: readmeContent,
        },
      ];

      await this.commitFiles(owner, repo, {
        message: `Initialize repository with ${template.name} template`,
        files,
      });
    } catch (error) {
      console.error('Template copying error:', error);
      throw error;
    }
  }

  /**
   * Get workflow template content
   */
  private async getWorkflowTemplate(): Promise<string> {
    // In a real implementation, this would fetch from our template directory
    // For now, return the workflow content we created earlier
    return `name: Convert Paper and Deploy to GitHub Pages

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  convert-and-deploy:
    environment:
      name: github-pages
      url: \${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout
      uses: actions/checkout@v4
      
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
        
    - name: Install uv
      uses: astral-sh/setup-uv@v4
      with:
        version: "latest"
        
    - name: Install dependencies
      run: |
        uv venv
        uv pip install docling marker-pdf pandoc-python-filter
        
    - name: Install system dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y pandoc texlive-latex-base texlive-latex-extra
        
    - name: Download conversion scripts
      run: |
        curl -o detect_and_convert.py https://raw.githubusercontent.com/your-username/paper-to-website/main/scripts/detect_and_convert.py
        curl -o apply_theme.py https://raw.githubusercontent.com/your-username/paper-to-website/main/scripts/apply_theme.py
        
    - name: Convert document
      run: python detect_and_convert.py
        
    - name: Apply theme
      run: python apply_theme.py
        
    - name: Setup Pages
      uses: actions/configure-pages@v5
      
    - name: Upload artifact
      uses: actions/upload-pages-artifact@v3
      with:
        path: './dist'
        
    - name: Deploy to GitHub Pages
      id: deployment
      uses: actions/deploy-pages@v4`;
  }

  /**
   * Get default configuration for a template
   */
  private getDefaultConfig(templateId: string): string {
    const config = {
      title: "Academic Paper",
      authors: [],
      date: new Date().toISOString().split('T')[0],
      theme: "academic",
      template: templateId,
    };

    return JSON.stringify(config, null, 2);
  }

  /**
   * Get README template
   */
  private getReadmeTemplate(template: PaperTemplate): string {
    return `# Academic Paper Website

This repository contains an academic paper website generated using the **${template.name}** template.

## Features

${template.features.map(feature => `- ${feature}`).join('\n')}

## How to Use

1. Upload your paper file (PDF, DOCX, or LaTeX) to this repository
2. Edit \`paper-config.json\` to customize your paper metadata
3. Push changes to trigger the conversion workflow
4. Your website will be available at: https://[username].github.io/[repository-name]

## Supported File Types

- PDF documents
- Microsoft Word (.docx) files
- LaTeX source files (.tex)
- ZIP archives containing LaTeX projects
- Overleaf projects (via Git integration)

## Configuration

Edit \`paper-config.json\` to customize:

- Paper title and authors
- Theme and styling options
- Template-specific settings

## Conversion Pipeline

The conversion process uses:

1. **Docling** - High-quality PDF/DOCX extraction
2. **Marker** - Advanced PDF to Markdown conversion
3. **Pandoc** - Fallback conversion for various formats

## Template Information

**Template:** ${template.name}
**Description:** ${template.description}
**Source:** ${template.repository_url}

---

Generated with [Paper to Website](https://github.com/your-username/paper-to-website)
`;
  }

  /**
   * Commit multiple files to repository
   */
  async commitFiles(
    owner: string,
    repo: string,
    request: CommitRequest
  ): Promise<GitHubCommit> {
    try {
      const branch = request.branch || 'main';

      // Get the current commit SHA
      const { data: ref } = await this.octokit.git.getRef({
        owner,
        repo,
        ref: `heads/${branch}`,
      });

      const currentCommitSha = ref.object.sha;

      // Get the current tree
      const { data: currentCommit } = await this.octokit.git.getCommit({
        owner,
        repo,
        commit_sha: currentCommitSha,
      });

      // Create blobs for each file
      const blobs = await Promise.all(
        request.files.map(async (file) => {
          const { data: blob } = await this.octokit.git.createBlob({
            owner,
            repo,
            content: file.content,
            encoding: file.encoding || 'utf-8',
          });
          return {
            path: file.path,
            mode: '100644' as const,
            type: 'blob' as const,
            sha: blob.sha,
          };
        })
      );

      // Create new tree
      const { data: newTree } = await this.octokit.git.createTree({
        owner,
        repo,
        base_tree: currentCommit.tree.sha,
        tree: blobs,
      });

      // Create new commit
      const { data: newCommit } = await this.octokit.git.createCommit({
        owner,
        repo,
        message: request.message,
        tree: newTree.sha,
        parents: [currentCommitSha],
      });

      // Update reference
      await this.octokit.git.updateRef({
        owner,
        repo,
        ref: `heads/${branch}`,
        sha: newCommit.sha,
      });

      return newCommit as GitHubCommit;
    } catch (error) {
      console.error('File commit error:', error);
      throw error;
    }
  }

  /**
   * Enable GitHub Pages for repository
   */
  async enableGitHubPages(owner: string, repo: string): Promise<void> {
    try {
      await this.octokit.repos.createPagesSite({
        owner,
        repo,
        source: {
          branch: 'main',
          path: '/',
        },
      });
    } catch (error) {
      // Pages might already be enabled
      if (error.status !== 409) {
        console.error('GitHub Pages setup error:', error);
        throw error;
      }
    }
  }

  /**
   * Get workflow runs for repository
   */
  async getWorkflowRuns(owner: string, repo: string): Promise<WorkflowRun[]> {
    try {
      const { data } = await this.octokit.actions.listWorkflowRunsForRepo({
        owner,
        repo,
        per_page: 10,
      });

      return data.workflow_runs as WorkflowRun[];
    } catch (error) {
      console.error('Workflow runs fetch error:', error);
      throw error;
    }
  }

  /**
   * Get repository information
   */
  async getRepository(owner: string, repo: string): Promise<GitHubRepository> {
    try {
      const { data } = await this.octokit.repos.get({
        owner,
        repo,
      });

      return data as GitHubRepository;
    } catch (error) {
      console.error('Repository fetch error:', error);
      throw error;
    }
  }

  /**
   * List user repositories
   */
  async listUserRepositories(): Promise<GitHubRepository[]> {
    try {
      const { data } = await this.octokit.repos.listForAuthenticatedUser({
        sort: 'updated',
        per_page: 50,
      });

      return data as GitHubRepository[];
    } catch (error) {
      console.error('Repository list error:', error);
      throw error;
    }
  }
}

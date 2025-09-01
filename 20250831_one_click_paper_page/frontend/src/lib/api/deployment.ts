/**
 * API client for deployment operations
 */
import React from 'react';

const API_BASE = 'http://localhost:8000/api';

export interface TemplateInfo {
  id: string;
  name: string;
  description: string;
  preview_url?: string;
  features: string[];
  repository_url: string;
}

export interface CreateRepositoryRequest {
  name: string;
  description?: string;
  private?: boolean;
  template: string;
  conversion_job_id: string;
}

export interface CreateRepositoryResponse {
  repository: {
    id: number;
    name: string;
    full_name: string;
    html_url: string;
    clone_url: string;
    private: boolean;
    owner: {
      login: string;
      avatar_url: string;
    };
  };
  deployment_id: string;
  status: string;
  message: string;
}

export interface DeploymentConfig {
  repository_name: string;
  template: string;
  paper_title?: string;
  paper_authors?: string[];
  paper_date?: string;
}

export interface DeploymentStatusResponse {
  deployment_id: string;
  status: 'pending' | 'queued' | 'in_progress' | 'success' | 'failure';
  repository: {
    html_url: string;
    name: string;
  };
  pages_url?: string;
  progress_percentage: number;
  message: string;
  error_message?: string;
}

export class DeploymentAPI {
  /**
   * Get available templates
   */
  static async getTemplates(): Promise<TemplateInfo[]> {
    const response = await fetch(`${API_BASE}/templates`);
    
    if (!response.ok) {
      throw new Error(`Failed to get templates: ${response.status}`);
    }
    
    return response.json();
  }

  /**
   * Create a new repository
   */
  static async createRepository(
    request: CreateRepositoryRequest,
    accessToken: string
  ): Promise<CreateRepositoryResponse> {
    const response = await fetch(`${API_BASE}/github/repository/create`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${accessToken}`,
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to create repository');
    }

    return response.json();
  }

  /**
   * Deploy converted content to repository
   */
  static async deployContent(
    deploymentId: string,
    config: DeploymentConfig,
    accessToken: string
  ): Promise<{ message: string }> {
    const response = await fetch(`${API_BASE}/github/deploy/${deploymentId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${accessToken}`,
      },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to deploy content');
    }

    return response.json();
  }

  /**
   * Get deployment status
   */
  static async getDeploymentStatus(
    deploymentId: string,
    accessToken: string
  ): Promise<DeploymentStatusResponse> {
    const response = await fetch(`${API_BASE}/github/deployment/${deploymentId}/status`, {
      headers: {
        'Authorization': `Bearer ${accessToken}`,
      },
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to get deployment status');
    }

    return response.json();
  }
}

/**
 * React hook for managing deployment operations
 */
export function useDeployment() {
  const getAccessToken = (): string => {
    const token = localStorage.getItem('github_token');
    if (!token) {
      throw new Error('GitHub access token not found. Please authenticate first.');
    }
    return token;
  };

  const createRepository = async (request: CreateRepositoryRequest) => {
    const token = getAccessToken();
    return DeploymentAPI.createRepository(request, token);
  };

  const deployContent = async (deploymentId: string, config: DeploymentConfig) => {
    const token = getAccessToken();
    return DeploymentAPI.deployContent(deploymentId, config, token);
  };

  const getDeploymentStatus = async (deploymentId: string) => {
    const token = getAccessToken();
    return DeploymentAPI.getDeploymentStatus(deploymentId, token);
  };

  const getTemplates = async () => {
    return DeploymentAPI.getTemplates();
  };

  // Memoize the returned object to prevent unnecessary re-renders
  return React.useMemo(() => ({
    createRepository,
    deployContent,
    getDeploymentStatus,
    getTemplates,
  }), []);
}

/**
 * Utility functions for deployment
 */
export const deploymentUtils = {
  /**
   * Generate a repository name from paper title
   */
  generateRepositoryName: (title: string): string => {
    return title
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, '')
      .replace(/\s+/g, '-')
      .substring(0, 50)
      .replace(/^-+|-+$/g, ''); // Remove leading/trailing hyphens
  },

  /**
   * Validate repository name
   */
  validateRepositoryName: (name: string): { valid: boolean; error?: string } => {
    if (!name) {
      return { valid: false, error: 'Repository name is required' };
    }

    if (name.length < 1 || name.length > 100) {
      return { valid: false, error: 'Repository name must be 1-100 characters' };
    }

    if (!/^[a-zA-Z0-9._-]+$/.test(name)) {
      return { 
        valid: false, 
        error: 'Repository name can only contain letters, numbers, hyphens, underscores, and periods' 
      };
    }

    if (name.startsWith('.') || name.startsWith('-')) {
      return { valid: false, error: 'Repository name cannot start with a period or hyphen' };
    }

    return { valid: true };
  },

  /**
   * Parse authors string into array
   */
  parseAuthors: (authorsString: string): string[] => {
    return authorsString
      .split(',')
      .map(author => author.trim())
      .filter(author => author.length > 0);
  },

  /**
   * Format authors array into string
   */
  formatAuthors: (authors: string[]): string => {
    return authors.join(', ');
  },

  /**
   * Get estimated deployment time based on template
   */
  getEstimatedDeploymentTime: (template: string): number => {
    // Return estimated time in seconds
    switch (template) {
      case 'minimal-academic':
        return 60; // 1 minute
      case 'academic-pages':
        return 120; // 2 minutes
      case 'al-folio':
        return 180; // 3 minutes
      default:
        return 90; // 1.5 minutes
    }
  },
};

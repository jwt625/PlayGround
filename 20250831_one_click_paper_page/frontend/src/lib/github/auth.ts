/**
 * GitHub OAuth authentication module
 */

import { OAuthTokenResponse, GitHubUser, GitHubError } from '../../types/github';

export class GitHubAuth {
  private clientId: string;
  private redirectUri: string;
  private scopes: string[];

  constructor(clientId: string, redirectUri: string, scopes: string[] = ['repo', 'user:email']) {
    this.clientId = clientId;
    this.redirectUri = redirectUri;
    this.scopes = scopes;
  }

  /**
   * Generate GitHub OAuth authorization URL
   */
  getAuthorizationUrl(state?: string): string {
    const params = new URLSearchParams({
      client_id: this.clientId,
      redirect_uri: this.redirectUri,
      scope: this.scopes.join(' '),
      response_type: 'code',
    });

    if (state) {
      params.append('state', state);
    }

    return `https://github.com/login/oauth/authorize?${params.toString()}`;
  }

  /**
   * Exchange authorization code for access token
   */
  async exchangeCodeForToken(code: string, state?: string): Promise<OAuthTokenResponse> {
    try {
      const response = await fetch('/api/github/oauth/token', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          code,
          state,
          redirect_uri: this.redirectUri,
        }),
      });

      if (!response.ok) {
        const error: GitHubError = await response.json();
        throw new Error(error.message || 'Failed to exchange code for token');
      }

      return await response.json();
    } catch (error) {
      console.error('Token exchange error:', error);
      throw error;
    }
  }

  /**
   * Get current authenticated user
   */
  async getCurrentUser(accessToken: string): Promise<GitHubUser> {
    try {
      const response = await fetch('https://api.github.com/user', {
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Accept': 'application/vnd.github.v3+json',
        },
      });

      if (!response.ok) {
        const error: GitHubError = await response.json();
        throw new Error(error.message || 'Failed to get user information');
      }

      return await response.json();
    } catch (error) {
      console.error('Get user error:', error);
      throw error;
    }
  }

  /**
   * Validate access token
   */
  async validateToken(accessToken: string): Promise<boolean> {
    try {
      const response = await fetch('https://api.github.com/user', {
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Accept': 'application/vnd.github.v3+json',
        },
      });

      return response.ok;
    } catch (error) {
      console.error('Token validation error:', error);
      return false;
    }
  }

  /**
   * Revoke access token
   */
  async revokeToken(accessToken: string): Promise<void> {
    try {
      const response = await fetch('/api/github/oauth/revoke', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          access_token: accessToken,
        }),
      });

      if (!response.ok) {
        const error: GitHubError = await response.json();
        throw new Error(error.message || 'Failed to revoke token');
      }
    } catch (error) {
      console.error('Token revocation error:', error);
      throw error;
    }
  }
}

/**
 * Local storage utilities for token management
 */
export class TokenStorage {
  private static readonly TOKEN_KEY = 'github_access_token';
  private static readonly USER_KEY = 'github_user';
  private static readonly EXPIRY_KEY = 'github_token_expiry';

  static saveToken(token: string, expiryHours: number = 24): void {
    const expiry = new Date();
    expiry.setHours(expiry.getHours() + expiryHours);
    
    localStorage.setItem(this.TOKEN_KEY, token);
    localStorage.setItem(this.EXPIRY_KEY, expiry.toISOString());
  }

  static getToken(): string | null {
    const token = localStorage.getItem(this.TOKEN_KEY);
    const expiry = localStorage.getItem(this.EXPIRY_KEY);

    if (!token || !expiry) {
      return null;
    }

    if (new Date() > new Date(expiry)) {
      this.clearToken();
      return null;
    }

    return token;
  }

  static clearToken(): void {
    localStorage.removeItem(this.TOKEN_KEY);
    localStorage.removeItem(this.USER_KEY);
    localStorage.removeItem(this.EXPIRY_KEY);
  }

  static saveUser(user: GitHubUser): void {
    localStorage.setItem(this.USER_KEY, JSON.stringify(user));
  }

  static getUser(): GitHubUser | null {
    const userStr = localStorage.getItem(this.USER_KEY);
    if (!userStr) {
      return null;
    }

    try {
      return JSON.parse(userStr);
    } catch (error) {
      console.error('Failed to parse stored user:', error);
      return null;
    }
  }

  static isAuthenticated(): boolean {
    return this.getToken() !== null;
  }
}

/**
 * React hook for GitHub authentication
 */
export function useGitHubAuth() {
  const clientId = import.meta.env.VITE_GITHUB_CLIENT_ID;
  const redirectUri = `${window.location.origin}/auth/callback`;
  
  const auth = new GitHubAuth(clientId, redirectUri);

  const login = () => {
    const state = Math.random().toString(36).substring(2, 15);
    sessionStorage.setItem('oauth_state', state);
    
    const authUrl = auth.getAuthorizationUrl(state);
    window.location.href = authUrl;
  };

  const logout = async () => {
    const token = TokenStorage.getToken();
    if (token) {
      try {
        await auth.revokeToken(token);
      } catch (error) {
        console.error('Failed to revoke token:', error);
      }
    }
    
    TokenStorage.clearToken();
    window.location.href = '/';
  };

  const handleCallback = async (code: string, state: string) => {
    const storedState = sessionStorage.getItem('oauth_state');
    sessionStorage.removeItem('oauth_state');

    if (state !== storedState) {
      throw new Error('Invalid state parameter');
    }

    const tokenResponse = await auth.exchangeCodeForToken(code, state);
    TokenStorage.saveToken(tokenResponse.access_token);

    const user = await auth.getCurrentUser(tokenResponse.access_token);
    TokenStorage.saveUser(user);

    return { token: tokenResponse.access_token, user };
  };

  const isAuthenticated = TokenStorage.isAuthenticated();
  const user = TokenStorage.getUser();
  const token = TokenStorage.getToken();

  return {
    login,
    logout,
    handleCallback,
    isAuthenticated,
    user,
    token,
  };
}

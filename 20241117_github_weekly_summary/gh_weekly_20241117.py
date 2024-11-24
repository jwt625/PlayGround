# %% [markdown]
# # GitHub Commit Analysis
# First, install required packages:
# ```bash
# pip install requests pytz matplotlib seaborn pandas
# ```

# %% Imports
import requests
from datetime import datetime, timedelta
import pytz
from collections import defaultdict
import os
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# %% GitHub API Setup
def setup_github_client(token: str = None):
    """
    Setup GitHub API client with token
    If token is not provided, tries to get it from environment variable
    """
    if token is None:
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            raise ValueError("Please set GITHUB_TOKEN environment variable or provide token")
    
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    return headers

# %% Base GitHub API Class
class GitHubCommitSummary:
    def __init__(self, token: str = None):
        """
        Initialize with GitHub personal access token
        """
        self.headers = setup_github_client(token)
        self.base_url = 'https://api.github.com'

    def get_commit_details(self, owner: str, repo: str, commit_sha: str) -> Dict:
        """
        Fetch detailed information about a specific commit
        """
        url = f'{self.base_url}/repos/{owner}/{repo}/commits/{commit_sha}'
        response = requests.get(url, headers=self.headers)
        return response.json() if response.status_code == 200 else None

    def get_repo_commits(self, owner: str, repo: str, since: datetime) -> List[Dict]:
        """
        Fetch commits for a repository since given date
        """
        commits = []
        page = 1
        
        while True:
            url = f'{self.base_url}/repos/{owner}/{repo}/commits'
            params = {
                'since': since.isoformat(),
                'per_page': 100,
                'page': page
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code != 200:
                print(f"Error fetching commits: {response.status_code}")
                break
                
            page_commits = response.json()
            if not page_commits:
                break
                
            commits.extend(page_commits)
            page += 1
            
        return commits

# %% Visualization Methods# %% Updated Visualization Methods 
def analyze_commit_patterns(commits: List[Dict], weeks_history: int = 1) -> Tuple[plt.Figure, plt.Figure]:
    """
    Analyze temporal patterns in commits and generate visualizations
    Times are converted to Pacific Time (PST/PDT)
    
    Args:
        commits: List of commit data
        weeks_history: Number of weeks being analyzed (for better visualization scaling)
    """
    if not commits:
        return None, None
        
    # Extract timestamp data and convert to Pacific Time
    pacific_tz = pytz.timezone('America/Los_Angeles')
    timestamps = []
    for commit in commits:
        # Parse UTC timestamp
        commit_date = datetime.strptime(
            commit['commit']['author']['date'],
            '%Y-%m-%dT%H:%M:%SZ'
        ).replace(tzinfo=pytz.UTC)
        # Convert to Pacific Time
        pacific_time = commit_date.astimezone(pacific_tz)
        timestamps.append(pacific_time)

    df = pd.DataFrame({'timestamp': timestamps})
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day_name()
    df['week'] = df['timestamp'].dt.isocalendar().week
    
    # Create heatmap of commit times
    plt.figure(figsize=(12, 6))
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hour_counts = pd.crosstab(df['day'], df['hour'])
    hour_counts = hour_counts.reindex(day_order)
    
    # Create heatmap with hour labels in 12-hour format
    sns.heatmap(hour_counts, cmap='YlOrRd', cbar_kws={'label': 'Number of Commits'})
    plt.title(f'Commit Activity Heatmap (Past {weeks_history} weeks) - Pacific Time')
    
    # Format x-axis labels to show AM/PM
    hour_labels = [f'{h%12 or 12}{["AM","PM"][h//12]}' for h in range(24)]
    plt.xticks(np.arange(24) + 0.5, hour_labels, rotation=45)
    
    plt.xlabel('Hour of Day (Pacific Time)')
    plt.ylabel('Day of Week')
    heatmap_fig = plt.gcf()
    
    # Create weekly distribution plot
    plt.figure(figsize=(max(10, weeks_history * 2), 6))  # Adjust figure size based on weeks
    if weeks_history > 1:
        # Group by week and day for longer periods
        weekly_commits = df.groupby(['week', 'day']).size().unstack()
        weekly_commits = weekly_commits.reindex(columns=day_order)
        weekly_commits.plot(kind='bar', stacked=True)
        plt.title('Commits Distribution by Week (Pacific Time)')
        plt.xlabel('Week Number')
        plt.ylabel('Number of Commits')
    else:
        # Show daily distribution for single week
        daily_commits = df['day'].value_counts().reindex(day_order)
        daily_commits.plot(kind='bar')
        plt.title('Commits Distribution by Day (Pacific Time)')
        plt.xlabel('Day of Week')
        plt.ylabel('Number of Commits')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    dist_fig = plt.gcf()
    
    return heatmap_fig, dist_fig


# %% Summary Generation
def generate_summary(commits: List[Dict], owner: str, repo: str, 
                    start_date: datetime, end_date: datetime, 
                    github_client: GitHubCommitSummary) -> str:
    """
    Generate formatted summary of commit activity
    """
    commits_by_day = defaultdict(lambda: defaultdict(list))
    total_additions = 0
    total_deletions = 0
    author_stats = defaultdict(lambda: {'commits': 0, 'additions': 0, 'deletions': 0})
    
    for commit in commits:
        if not commit.get('commit'):
            continue
            
        commit_date = datetime.strptime(
            commit['commit']['author']['date'],
            '%Y-%m-%dT%H:%M:%SZ'
        ).replace(tzinfo=pytz.UTC)
        
        if start_date <= commit_date <= end_date:
            day = commit_date.strftime('%Y-%m-%d')
            author = commit['commit']['author']['name']
            commits_by_day[day][author].append(commit['commit']['message'])
            
            # Fetch detailed commit information
            commit_detail = github_client.get_commit_details(owner, repo, commit['sha'])
            if commit_detail and 'stats' in commit_detail:
                stats = commit_detail['stats']
                additions = stats.get('additions', 0)
                deletions = stats.get('deletions', 0)
                
                total_additions += additions
                total_deletions += deletions
                
                author_stats[author]['commits'] += 1
                author_stats[author]['additions'] += additions
                author_stats[author]['deletions'] += deletions

    # Generate summary text
    summary = f"Weekly Commit Summary for {owner}/{repo}\n"
    summary += f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n\n"
    
    # Add statistics
    summary += "Overall Statistics:\n"
    summary += "-" * 40 + "\n"
    summary += f"Total Commits: {sum(stats['commits'] for stats in author_stats.values())}\n"
    summary += f"Total Lines Added: {total_additions:,}\n"
    summary += f"Total Lines Deleted: {total_deletions:,}\n"
    summary += f"Net Line Changes: {total_additions - total_deletions:,}\n\n"
    
    # Add author statistics
    summary += "Author Statistics:\n"
    summary += "-" * 40 + "\n"
    for author, stats in author_stats.items():
        summary += f"\n{author}:\n"
        summary += f"  Commits: {stats['commits']}\n"
        summary += f"  Lines Added: {stats['additions']:,}\n"
        summary += f"  Lines Deleted: {stats['deletions']:,}\n"
        summary += f"  Net Changes: {stats['additions'] - stats['deletions']:,}\n"
    
    return summary

# %% Example Usage
# Replace these with your repository details
OWNER = "jwt625"
REPO = "jwt625.github.io"
# TOKEN = os.getenv('GITHUB_TOKEN')  # Make sure to set this environment variable

# Initialize the client
client = GitHubCommitSummary(TOKEN)

# Set date range
end_date = datetime.now(pytz.UTC)
start_date = end_date - timedelta(days=7)

# Fetch commits
commits = client.get_repo_commits(OWNER, REPO, start_date)

# Generate summary
summary = generate_summary(commits, OWNER, REPO, start_date, end_date, client)
print(summary)

# Generate and display visualizations
heatmap_fig, dist_fig = analyze_commit_patterns(commits)
plt.show()



# %% Example Usage with Flexible Time Window
def analyze_repositories(repositories: List[Dict], weeks_history: int = 1, token: str = None):
    """
    Analyze commit history for multiple repositories over a specified number of weeks
    
    Args:
        repositories: List of dicts with 'owner' and 'repo' keys
        weeks_history: Number of weeks of history to analyze (default: 1)
        token: GitHub API token (optional, will use env var if not provided)
    """
    TOKEN = token or os.getenv('GITHUB_TOKEN')
    client = GitHubCommitSummary(TOKEN)

    # Set date range
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(weeks=weeks_history)  # Now using weeks parameter
    
    # Track commits across all repos
    all_commits = []
    combined_summary = f"Combined Repository Analysis - Past {weeks_history} weeks\n"
    combined_summary += f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n"
    combined_summary += "=" * 30 + "\n\n"

    # Analyze each repository
    for repo_info in repositories:
        owner = repo_info["owner"]
        repo = repo_info["repo"]
        
        print(f"Fetching data for {owner}/{repo}...")
        
        # Fetch commits for this repository
        repo_commits = client.get_repo_commits(owner, repo, start_date)
        all_commits.extend(repo_commits)
        
        # Generate individual repo summary
        repo_summary = generate_summary(repo_commits, owner, repo, start_date, end_date, client)
        combined_summary += f"\n\nRepository: {owner}/{repo}\n" + "-" * 40 + "\n" + repo_summary

    # Generate combined visualizations
    if all_commits:
        heatmap_fig, dist_fig = analyze_commit_patterns(all_commits)
        plt.show()
    else:
        print("No commits found in the specified time period.")

    return combined_summary, all_commits

# Example usage:
repositories = [
    {"owner": "jwt625", "repo": "jwt625.github.io"},
    {"owner": "jwt625", "repo": "PlayGround"},
    {"owner": "jwt625", "repo": "Obsidian-Journals"},
]

# Analyze last 4 weeks
summary, commits = analyze_repositories(repositories, weeks_history=4, token=TOKEN)
print(summary)

# Analyze just last week
summary, commits = analyze_repositories(repositories, weeks_history=1, token=TOKEN)
print(summary)

# Analyze last 12 weeks
summary, commits = analyze_repositories(repositories, weeks_history=12, token=TOKEN)
print(summary)

#%%
repositories = [
    {"owner": "jwt625", "repo": "jwt625.github.io"},
    {"owner": "jwt625", "repo": "PlayGround"},
    {"owner": "jwt625", "repo": "Obsidian-Journals"},
]
# Analyze yearly data
summary, commits = analyze_repositories(repositories, weeks_history=26, token=TOKEN)

# test

# %%

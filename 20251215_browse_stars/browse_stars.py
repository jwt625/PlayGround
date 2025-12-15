#!/usr/bin/env python3
"""
Playwright script to browse GitHub stars and find paper/literature management repos
"""
from playwright.sync_api import sync_playwright
import time
import json

def browse_github_stars():
    with sync_playwright() as p:
        # Launch Firefox
        browser = p.firefox.launch(headless=False)
        page = browser.new_page()
        
        # Navigate to the stars page
        print("Navigating to GitHub stars page...")
        page.goto("https://github.com/jwt625?tab=stars")

        # Wait for the page to load completely
        page.wait_for_load_state("networkidle")
        print("Waiting for content to render...")
        time.sleep(5)  # Give JavaScript time to render
        
        all_repos = []
        page_num = 1
        max_pages = 50  # Scan up to 50 pages
        
        # Keywords to look for
        keywords = ['paper', 'pdf', 'literature', 'reference', 'citation', 'zotero', 
                   'mendeley', 'research', 'annotate', 'annotation', 'bibliography',
                   'bibtex', 'scholar', 'academic', 'reading', 'reader']
        
        matching_repos = []
        
        while page_num <= max_pages:
            print(f"\n=== Page {page_num} ===")

            # Wait for content to load
            time.sleep(2)

            # Try multiple selectors
            selectors_to_try = [
                'div.col-12.d-block.width-full.py-4.border-bottom',  # The actual structure from the HTML
                'div[id^="user-list-item"]',  # GitHub uses IDs like user-list-item-0, user-list-item-1, etc
                'div[data-filterable-for="your-repos-filter"] > div',
                'article.Box-row',
                'div.Box-row',
                '[data-hovercard-type="repository"]',
                'h3 a[data-hovercard-type="repository"]',
                'div[itemprop="owns"]'  # Another common pattern
            ]

            repo_items = []
            for selector in selectors_to_try:
                items = page.locator(selector).all()
                if len(items) > 0:
                    repo_items = items
                    break

            print(f"Found {len(repo_items)} repositories")

            for item in repo_items:
                try:
                    # Get repo name - try to find h3 with link
                    name_elem = item.locator('h3 a').first
                    if name_elem.count() == 0:
                        continue

                    repo_name = name_elem.inner_text().strip()
                    repo_url = name_elem.get_attribute('href')

                    if not repo_url:
                        continue

                    # Get description if available
                    desc_elem = item.locator('p')
                    description = ""
                    if desc_elem.count() > 0:
                        description = desc_elem.first.inner_text().strip()

                    repo_info = {
                        'name': repo_name,
                        'url': f"https://github.com{repo_url}" if repo_url.startswith('/') else repo_url,
                        'description': description
                    }

                    all_repos.append(repo_info)

                    # Check if this matches our keywords
                    search_text = f"{repo_name} {description}".lower()
                    if any(keyword in search_text for keyword in keywords):
                        print(f"  ✓ MATCH: {repo_name}")
                        print(f"    Description: {description[:100]}...")
                        matching_repos.append(repo_info)

                except Exception as e:
                    print(f"  Error processing repo: {e}")
                    continue

            # Save results after each page
            with open('starred_repos_matching.json', 'w') as f:
                json.dump(matching_repos, f, indent=2)
            with open('all_starred_repos.json', 'w') as f:
                json.dump(all_repos, f, indent=2)

            # Try to find and click the "Next" button
            try:
                # Look for pagination - try different approaches
                next_button = page.locator('a:has-text("Next")')
                if next_button.count() > 0:
                    print(f"\nGoing to page {page_num + 1}...")
                    # Get the href and navigate directly instead of clicking
                    href = next_button.get_attribute('href')
                    if href:
                        page.goto(f"https://github.com{href}")
                        time.sleep(3)  # Wait for page to load
                        page_num += 1
                    else:
                        print("\nNo href found on Next button. Reached the end.")
                        break
                else:
                    print("\nNo 'Next' button found. Reached the end.")
                    break
            except Exception as e:
                print(f"\nError navigating to next page: {e}")
                break
        
        browser.close()
        
        # Print summary
        print("\n" + "="*80)
        print(f"SUMMARY: Scanned {len(all_repos)} repositories across {page_num} pages")
        print(f"Found {len(matching_repos)} repositories matching paper/literature keywords:")
        print("="*80)
        
        for repo in matching_repos:
            print(f"\n{repo['name']}")
            print(f"  URL: {repo['url']}")
            print(f"  Description: {repo['description']}")
        
        # Save results to file
        with open('starred_repos_matching.json', 'w') as f:
            json.dump(matching_repos, f, indent=2)
        
        with open('all_starred_repos.json', 'w') as f:
            json.dump(all_repos, f, indent=2)
        
        print(f"\n\nResults saved to:")
        print(f"  - starred_repos_matching.json ({len(matching_repos)} repos)")
        print(f"  - all_starred_repos.json ({len(all_repos)} repos)")

if __name__ == "__main__":
    browse_github_stars()


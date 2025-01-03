// Function to fetch the pinned tweet from a profile URL
async function fetchPinnedTweet(profileUrl) {
  try {
    const response = await fetch(profileUrl);
    const text = await response.text();
    const parser = new DOMParser();
    const doc = parser.parseFromString(text, "text/html");

    // Find the pinned tweet (this selector may need to be updated if Twitter/X changes their HTML structure)
    const pinnedTweet = doc.querySelector('[data-testid="tweet"]:first-child');
    if (pinnedTweet) {
      return pinnedTweet.outerHTML;
    } else {
      console.log("No pinned tweet found.");
      return null;
    }
  } catch (error) {
    console.error("Error fetching pinned tweet:", error);
    return null;
  }
}

// Function to inject the pinned tweet into the feed
function injectPinnedTweet(pinnedTweetHtml) {
  if (!pinnedTweetHtml) return;

  // Create a container for the pinned tweet
  const pinnedTweetContainer = document.createElement("div");
  pinnedTweetContainer.className = "pinned-tweet-container";
  pinnedTweetContainer.innerHTML = pinnedTweetHtml;

  // Find the Twitter/X feed container
  const feedContainer = document.querySelector('[aria-label="Timeline: Your Home Timeline"]') || document.querySelector("main");
  if (feedContainer) {
    // Insert the pinned tweet at the top of the feed
    feedContainer.insertBefore(pinnedTweetContainer, feedContainer.firstChild);
  } else {
    console.error("Could not find the feed container.");
  }
}

// Main function to run the script
async function main() {
  const profileUrl = "https://x.com/jwt0625"; // Replace with your profile URL
  const pinnedTweetHtml = await fetchPinnedTweet(profileUrl);
  injectPinnedTweet(pinnedTweetHtml);
}

// Run the script when the page loads
main();
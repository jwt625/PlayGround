# ChatGPT Conversation Extractor

A Chrome extension that extracts ChatGPT conversations and saves them as nicely formatted HTML files. Now with bulk export functionality!

## Features

- **Single Conversation Export**: Extract the current ChatGPT conversation with one click
- **Bulk Export All Conversations**: Export your entire ChatGPT history at once
- **Smart Extraction**: Uses ChatGPT's specific HTML structure to accurately capture messages
- **Formatting Preserved**: Maintains code blocks, markdown formatting, and message structure
- **Customizable Output**:
  - Include/exclude code blocks
  - Include/exclude timestamps
  - Light/dark mode themes
- **Organized Export**: Bulk exports create an index page with links to all conversations

## Installation

1. Download or clone this repository
2. Open Chrome and navigate to `chrome://extensions/`
3. Enable "Developer mode" toggle in the top right
4. Click "Load unpacked"
5. Select the `chatgpt-extractor` folder
6. The extension will appear in your toolbar

## Usage

### Single Conversation Export

1. Navigate to a ChatGPT conversation (https://chatgpt.com)
2. Click the extension icon in your toolbar
3. Configure your export options
4. Click "Extract Current Conversation"
5. Choose where to save the HTML file

### Bulk Export All Conversations

1. Navigate to ChatGPT (https://chatgpt.com)
2. **Important**: Make sure the sidebar is open (shows your chat history)
3. Click the extension icon
4. Click "Export All Conversations"
5. The extension will:
   - Extract your chat history list
   - Navigate to each conversation automatically
   - Extract and save each conversation
   - Create an index.html with links to all chats
   - Show progress throughout the process

### Export Structure

Bulk exports create a folder structure like:
```
chatgpt_export_2024-06-20_1750394123456/
├── index.html                    # Overview with statistics and links
└── conversations/
    ├── 1_first_chat_title.html
    ├── 2_second_chat_title.html
    └── ...
```

## Options

- **Include code blocks**: Preserves code snippets with syntax highlighting
- **Include timestamps**: Shows when messages were sent (if available)
- **Dark mode theme**: Exports with a dark theme for comfortable reading

## Technical Details

The extension works by:
1. Finding chat history in `div#history` containing links with `span[dir="auto"]` titles
2. Extracting messages using:
   - `.text-token-text-primary.w-full` for message containers
   - `.whitespace-pre-wrap` for plain text content
   - `.markdown.prose` for formatted ChatGPT responses
   - `[data-message-author-role]` for identifying user vs assistant
3. Preserving the complete conversation structure in clean HTML

## Troubleshooting

### Buttons not working?

1. **Check the Console**:
   - Right-click the extension popup → "Inspect"
   - Check for errors in the Console tab
   - Also check the main page console (F12)

2. **Reload the extension**:
   - Go to `chrome://extensions/`
   - Click the reload button on the extension
   - Refresh the ChatGPT page

3. **Common issues**:
   - Make sure you're on chatgpt.com
   - For bulk export: ensure the sidebar is visible
   - Try refreshing the page if content script isn't loaded

### Debugging

The extension includes extensive console logging. To see logs:
- Extension popup logs: Right-click popup → Inspect → Console
- Content script logs: F12 on ChatGPT page → Console
- Look for "ChatGPT Extractor content script loaded" message

## Privacy & Security

- **Local Processing**: All extraction happens in your browser
- **No External Servers**: No data is sent anywhere
- **Minimal Permissions**: Only requires access to ChatGPT domains
- **Open Source**: Inspect the code to verify functionality

## Limitations

- Cannot extract conversations that haven't fully loaded
- Some complex elements (interactive widgets) may not be preserved
- Rate limiting: Bulk export processes one chat at a time to avoid issues
- Very long conversations may take time to process

## File Structure

```
chatgpt-extractor/
├── manifest.json          # Extension configuration
├── popup.html            # Extension popup interface
├── popup.js              # Popup functionality and export logic
├── content.js            # Content script for extracting data
├── README.md             # This file
├── create_icons.html     # Utility to create extension icons
└── generate_icons.py     # Python script for icon generation
```

## Contributing

Feel free to submit issues or pull requests. The code is structured to be easily extendable for additional features.

## License

This extension is provided as-is for personal use. Please respect OpenAI's terms of service when using this tool.
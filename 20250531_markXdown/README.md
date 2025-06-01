# MarkXdown

A Chrome extension that renders markdown and LaTeX equations in Twitter/X posts.

## Features

- Renders markdown syntax in tweets
- Supports LaTeX equations using MathJax
- Works with both Twitter.com and X.com
- Automatically processes new tweets as they load

## Installation

1. Download or clone this repository
2. Open Chrome and go to `chrome://extensions/`
3. Enable "Developer mode" in the top right
4. Click "Load unpacked" and select the extension directory

## Usage

The extension will automatically process markdown and LaTeX equations in tweets. Here are some examples:

### Markdown Examples

```markdown
# Heading 1
## Heading 2

**Bold text**
*Italic text*

- List item 1
- List item 2

> Blockquote

`inline code`
```

### LaTeX Examples

Inline math: $E = mc^2$

Display math:
$$
\frac{d}{dx}e^x = e^x
$$

## Dependencies

- marked.js for markdown parsing
- MathJax for LaTeX rendering

## Development

To modify the extension:

1. Make your changes to the source files
2. Go to `chrome://extensions/`
3. Click the refresh icon on the extension card
4. The changes will be applied after reloading Twitter/X

## License

MIT License 
# Text Inversion MCP Server

A Model Context Protocol (MCP) server that provides text inversion/reversal functionality. This server exposes a single tool that can reverse the character order of any input string.

## Features

- **Text Inversion Tool**: Reverses the order of characters in a text string
- **MCP Protocol Compliant**: Follows the official MCP specification
- **TypeScript Implementation**: Written in TypeScript with full type safety
- **Error Handling**: Robust error handling and validation
- **Stdio Transport**: Uses standard input/output for communication

## Project Structure

```
text-inversion-mcp/
├── index.ts           # Main server implementation
├── package.json       # Node.js dependencies and scripts
├── tsconfig.json      # TypeScript configuration
├── build/             # Compiled JavaScript output
└── README.md          # This file
```

## Installation

1. **Create project directory:**
   ```bash
   mkdir text-inversion-mcp
   cd text-inversion-mcp
   ```

2. **Initialize npm project:**
   ```bash
   npm init -y
   ```

3. **Install dependencies:**
   ```bash
   npm install @modelcontextprotocol/sdk
   npm install -D typescript @types/node tsx
   ```

4. **Create the required files:**
   - Copy `index.ts` (main server code)
   - Copy `package.json` (with proper dependencies and scripts)
   - Copy `tsconfig.json` (TypeScript configuration)

## Usage

### Development Mode
```bash
npm run dev
```

### Production Build and Run
```bash
npm run build
npm start
```

### Available Tool

The server exposes one tool:

- **`invert_text`**: Takes a text string and returns both original and inverted versions
  - Input: `{ "text": "hello world" }`
  - Output: `Original: "hello world"\nInverted: "dlrow olleh"`

## MCP Client Integration

To use this server with Claude Desktop or other MCP clients, add it to your MCP configuration:

### Claude Desktop Configuration
Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "text-inversion": {
      "command": "node",
      "args": ["path/to/text-inversion-mcp/build/index.js"]
    }
  }
}
```

### Testing the Server

Once running, the server will accept MCP protocol messages via stdio. You can test it by calling the `invert_text` tool with any text string.

## Technical Details

- **Protocol**: Model Context Protocol (MCP) v0.5.0
- **Transport**: Standard I/O (stdio)
- **Language**: TypeScript/Node.js
- **Error Handling**: Validates input parameters and provides meaningful error messages
- **Build System**: TypeScript compiler with source maps and declarations

## Dependencies

- `@modelcontextprotocol/sdk`: Core MCP SDK for server implementation
- `typescript`: TypeScript compiler
- `@types/node`: Node.js type definitions
- `tsx`: TypeScript execution for development

## License

MIT License

## Example Usage

Once integrated with an MCP client:

```
User: "Can you invert the text 'hello world'?"
Assistant: [Calls invert_text tool]
Result: Original: "hello world"
        Inverted: "dlrow olleh"
```

This server provides a simple but complete example of how to build custom MCP servers for extending AI assistant capabilities with domain-specific tools.
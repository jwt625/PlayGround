#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';

/**
 * Text Inversion MCP Server
 * 
 * This server provides a single tool for inverting/reversing text strings.
 * It demonstrates a simple but complete MCP server implementation.
 */

class TextInversionServer {
  private server: Server;

  constructor() {
    this.server = new Server(
      {
        name: 'text-inversion-server',
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupToolHandlers();
    this.setupErrorHandling();
  }

  private setupToolHandlers(): void {
    // Handle tool listing
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          {
            name: 'invert_text',
            description: 'Inverts/reverses the order of characters in a text string',
            inputSchema: {
              type: 'object',
              properties: {
                text: {
                  type: 'string',
                  description: 'The text string to invert',
                },
              },
              required: ['text'],
            },
          },
        ],
      };
    });

    // Handle tool calls
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      if (name === 'invert_text') {
        return await this.handleInvertText(args);
      } else {
        throw new Error(`Unknown tool: ${name}`);
      }
    });
  }

  private async handleInvertText(args: any): Promise<any> {
    try {
      // Validate input
      if (!args || typeof args.text !== 'string') {
        throw new Error('Invalid input: text parameter must be a string');
      }

      const originalText = args.text;
      const invertedText = originalText.split('').reverse().join('');

      return {
        content: [
          {
            type: 'text',
            text: `Original: "${originalText}"\nInverted: "${invertedText}"`,
          },
        ],
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      return {
        content: [
          {
            type: 'text',
            text: `Error: ${errorMessage}`,
          },
        ],
        isError: true,
      };
    }
  }

  private setupErrorHandling(): void {
    this.server.onerror = (error) => {
      console.error('[MCP Error]', error);
    };

    process.on('SIGINT', async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  async run(): Promise<void> {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Text Inversion MCP Server running on stdio');
  }
}

// Start the server
const server = new TextInversionServer();
server.run().catch((error) => {
  console.error('Failed to start server:', error);
  process.exit(1);
});

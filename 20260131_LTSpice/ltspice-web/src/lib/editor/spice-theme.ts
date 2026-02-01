/**
 * Dark theme for SPICE editor matching LTSpice aesthetic
 */

import { EditorView } from '@codemirror/view';
import { HighlightStyle, syntaxHighlighting } from '@codemirror/language';
import { tags } from '@lezer/highlight';

// Editor theme (UI elements)
export const spiceEditorTheme = EditorView.theme({
	'&': {
		backgroundColor: '#1a1a1a',
		color: '#e0e0e0',
		height: '100%'
	},
	'.cm-content': {
		fontFamily: 'Consolas, "Courier New", monospace',
		fontSize: '13px',
		padding: '8px 0'
	},
	'.cm-line': {
		padding: '0 8px'
	},
	'&.cm-focused': {
		outline: 'none'
	},
	'.cm-gutters': {
		backgroundColor: '#1a1a1a',
		color: '#666',
		border: 'none',
		borderRight: '1px solid #333'
	},
	'.cm-activeLineGutter': {
		backgroundColor: '#252525'
	},
	'.cm-activeLine': {
		backgroundColor: '#252525'
	},
	'.cm-cursor': {
		borderLeftColor: '#fff'
	},
	'.cm-selectionBackground': {
		backgroundColor: '#264f78 !important'
	},
	'&.cm-focused .cm-selectionBackground': {
		backgroundColor: '#264f78 !important'
	},
	'.cm-matchingBracket': {
		backgroundColor: '#3a3a3a',
		outline: '1px solid #666'
	}
}, { dark: true });

// Syntax highlighting colors
export const spiceHighlightStyle = HighlightStyle.define([
	{ tag: tags.comment, color: '#6a9955' },           // Green for comments
	{ tag: tags.keyword, color: '#569cd6' },           // Blue for directives
	{ tag: tags.variableName, color: '#dcdcaa' },      // Yellow for component names
	{ tag: tags.definition(tags.variableName), color: '#9cdcfe' }, // Light blue for node names
	{ tag: tags.number, color: '#b5cea8' },            // Light green for numbers
	{ tag: tags.bracket, color: '#ffd700' },           // Gold for brackets
	{ tag: tags.operator, color: '#d4d4d4' }           // White for operators
]);

// Combined syntax highlighting extension
export const spiceSyntaxHighlighting = syntaxHighlighting(spiceHighlightStyle);


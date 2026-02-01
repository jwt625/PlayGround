/**
 * SPICE Language Mode for CodeMirror 6
 * Provides syntax highlighting for SPICE netlists
 */

import { StreamLanguage } from '@codemirror/language';

// SPICE language definition using StreamLanguage (simpler than Lezer)
export const spiceLanguage = StreamLanguage.define({
	name: 'spice',
	
	startState() {
		return { inComment: false };
	},
	
	token(stream, state) {
		// Skip whitespace
		if (stream.eatSpace()) return null;
		
		// Comments: lines starting with * or ;
		if (stream.sol() && (stream.peek() === '*' || stream.peek() === ';')) {
			stream.skipToEnd();
			return 'comment';
		}
		
		// Inline comments after ;
		if (stream.peek() === ';') {
			stream.skipToEnd();
			return 'comment';
		}
		
		// SPICE directives: .tran, .ac, .dc, .op, .model, .subckt, .ends, .end, etc.
		if (stream.sol() && stream.peek() === '.') {
			stream.next();
			stream.eatWhile(/[a-zA-Z]/);
			return 'keyword';
		}
		
		// Component prefixes at start of line: R, C, L, V, I, D, Q, M, X, etc.
		if (stream.sol()) {
			const ch = stream.peek();
			if (ch && /[RCLVIDQMXEGHFBKSTUW]/i.test(ch)) {
				stream.eatWhile(/[a-zA-Z0-9_]/);
				return 'variableName';
			}
		}
		
		// Numbers with optional SI prefixes: 1k, 10u, 100n, 1meg, etc.
		if (stream.match(/^-?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?/)) {
			// Check for SI prefix
			stream.match(/^(meg|mil|[TGMKkmunpfa])/i);
			// Check for unit
			stream.match(/^(Hz|V|A|F|H|ohm|Ohm|OHM)?/);
			return 'number';
		}
		
		// Node names and identifiers
		if (stream.match(/^[a-zA-Z_][a-zA-Z0-9_]*/)) {
			const word = stream.current().toUpperCase();
			// Check for SPICE keywords
			if (['PULSE', 'SINE', 'SIN', 'EXP', 'PWL', 'SFFM', 'AC', 'DC', 'TRAN'].includes(word)) {
				return 'keyword';
			}
			return 'definition';
		}
		
		// Parentheses for source definitions
		if (stream.peek() === '(' || stream.peek() === ')') {
			stream.next();
			return 'bracket';
		}
		
		// Equals sign for parameters
		if (stream.peek() === '=') {
			stream.next();
			return 'operator';
		}
		
		// Skip unknown characters
		stream.next();
		return null;
	},
	
	languageData: {
		commentTokens: { line: '*' }
	}
});


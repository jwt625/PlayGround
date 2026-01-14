#!/usr/bin/env python3
"""
Comprehensive data mining script to extract patterns from Augment conversation histories.
Combines statistics, deep pattern analysis, and actionable preference extraction.
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Any
import sys


class ConversationMiner:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.conversations = []
        self.exchanges = []
        self.stats = defaultdict(int)
        self.patterns = defaultdict(list)
        
    def load_all_conversations(self):
        """Load all conversation JSON files."""
        print("Loading conversation data...")
        
        json_files = list(self.data_dir.glob("*.json"))
        json_files = [f for f in json_files if f.name != "extraction_summary.json"]
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                workspace_id = data.get('workspace_id', '')
                folder_path = data.get('folder_path', '')
                
                for conv in data.get('conversations', []):
                    conv['_workspace_id'] = workspace_id
                    conv['_folder_path'] = folder_path
                    self.conversations.append(conv)
                    
                    for exchange in conv.get('exchanges', []):
                        exchange['_workspace_id'] = workspace_id
                        exchange['_folder_path'] = folder_path
                        exchange['_conversation_id'] = conv.get('conversation_id', '')
                        self.exchanges.append(exchange)
                        
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                
        print(f"Loaded {len(self.conversations)} conversations with {len(self.exchanges)} exchanges")
        
    def compute_basic_statistics(self):
        """Compute basic statistics about the dataset."""
        print("\nComputing basic statistics...")
        
        self.stats['total_conversations'] = len(self.conversations)
        self.stats['total_exchanges'] = len(self.exchanges)
        
        # Workspace distribution
        workspace_counts = Counter(e['_workspace_id'] for e in self.exchanges)
        self.stats['workspaces'] = dict(workspace_counts)
        
        # Folder path distribution
        folder_counts = Counter(e['_folder_path'] for e in self.exchanges)
        self.stats['folders'] = dict(folder_counts)
        
        # Exchanges per conversation
        exchanges_per_conv = [len(c.get('exchanges', [])) for c in self.conversations]
        self.stats['avg_exchanges_per_conversation'] = sum(exchanges_per_conv) / len(exchanges_per_conv) if exchanges_per_conv else 0
        self.stats['max_exchanges_per_conversation'] = max(exchanges_per_conv) if exchanges_per_conv else 0
        self.stats['min_exchanges_per_conversation'] = min(exchanges_per_conv) if exchanges_per_conv else 0
        
        # Message lengths
        request_lengths = [len(e.get('request_message', '')) for e in self.exchanges if e.get('request_message')]
        response_lengths = [len(e.get('response_text', '')) for e in self.exchanges if e.get('response_text')]
        
        self.stats['avg_request_length'] = sum(request_lengths) / len(request_lengths) if request_lengths else 0
        self.stats['avg_response_length'] = sum(response_lengths) / len(response_lengths) if response_lengths else 0
        
        # Exchanges with content
        self.stats['exchanges_with_request'] = len(request_lengths)
        self.stats['exchanges_with_response'] = len(response_lengths)
        
    def extract_explicit_preferences(self):
        """Extract explicit preference statements."""
        print("\nExtracting explicit preferences...")
        
        # Preference indicators
        preference_patterns = [
            (r'\b(always|ALWAYS)\s+(\w+(?:\s+\w+){0,10})', 'always'),
            (r'\b(never|NEVER)\s+(\w+(?:\s+\w+){0,10})', 'never'),
            (r'\b(prefer|PREFER)\s+(\w+(?:\s+\w+){0,10})', 'prefer'),
            (r'\b(don\'t|do not|DON\'T|DO NOT)\s+(\w+(?:\s+\w+){0,10})', 'dont'),
            (r'\b(must|MUST)\s+(\w+(?:\s+\w+){0,10})', 'must'),
            (r'\b(should|SHOULD)\s+(\w+(?:\s+\w+){0,10})', 'should'),
        ]
        
        for exchange in self.exchanges:
            request = exchange.get('request_message', '')
            if not request:
                continue
                
            for pattern, pref_type in preference_patterns:
                matches = re.finditer(pattern, request, re.IGNORECASE)
                for match in matches:
                    context = request[max(0, match.start()-50):min(len(request), match.end()+50)]
                    self.patterns[f'preference_{pref_type}'].append({
                        'text': match.group(0),
                        'context': context,
                        'folder': exchange['_folder_path'],
                        'conversation_id': exchange['_conversation_id']
                    })
        
        for pref_type in ['always', 'never', 'prefer', 'dont', 'must', 'should']:
            self.stats[f'explicit_preferences_{pref_type}'] = len(self.patterns[f'preference_{pref_type}'])
            
    def extract_tool_mentions(self):
        """Extract mentions of tools, commands, and technologies."""
        print("\nExtracting tool mentions...")
        
        # Common tools and technologies
        tools = {
            'package_managers': ['npm', 'yarn', 'pnpm', 'pip', 'uv', 'poetry', 'conda', 'cargo', 'go mod'],
            'languages': ['python', 'javascript', 'typescript', 'rust', 'go', 'java', 'c++'],
            'frameworks': ['react', 'vue', 'angular', 'fastapi', 'flask', 'django', 'express', 'next.js'],
            'databases': ['postgres', 'mysql', 'mongodb', 'redis', 'sqlite', 'leveldb'],
            'devops': ['docker', 'kubernetes', 'k8s', 'terraform', 'ansible'],
            'testing': ['pytest', 'jest', 'mocha', 'vitest', 'unittest'],
        }
        
        tool_counts = defaultdict(Counter)
        
        for exchange in self.exchanges:
            text = (exchange.get('request_message', '') + ' ' + exchange.get('response_text', '')).lower()

            for category, tool_list in tools.items():
                for tool in tool_list:
                    if re.search(r'\b' + re.escape(tool.lower()) + r'\b', text):
                        tool_counts[category][tool] += 1

        self.patterns['tool_mentions'] = dict(tool_counts)
        for category, counts in tool_counts.items():
            self.stats[f'tool_category_{category}'] = sum(counts.values())

    def identify_correction_patterns(self):
        """Identify instances where user corrects LLM behavior."""
        print("\nIdentifying correction patterns...")

        correction_indicators = [
            r'\b(no|No|NO)[,.]?\s+(that\'s|thats|that is)\s+(wrong|incorrect|not right)',
            r'\b(don\'t|dont|do not)\s+(do that|use that|create that)',
            r'\b(you should|you need to|you must)\s+(\w+(?:\s+\w+){0,5})\s+instead',
            r'\b(use|Use)\s+(\w+)\s+instead\s+of\s+(\w+)',
            r'\b(why|Why)\s+(did you|are you)\s+(\w+(?:\s+\w+){0,5})',
            r'\b(I said|I told you|I asked you)\s+(\w+(?:\s+\w+){0,10})',
            r'\b(that\'s|thats)\s+not\s+what\s+I\s+(asked|wanted|said)',
        ]

        for exchange in self.exchanges:
            request = exchange.get('request_message', '')
            if not request:
                continue

            for pattern in correction_indicators:
                if re.search(pattern, request, re.IGNORECASE):
                    context = request[:200] if len(request) > 200 else request
                    self.patterns['corrections'].append({
                        'text': request,
                        'context': context,
                        'folder': exchange['_folder_path'],
                        'conversation_id': exchange['_conversation_id']
                    })
                    break  # Count each exchange only once

        self.stats['correction_count'] = len(self.patterns['corrections'])

    def extract_command_patterns(self):
        """Extract shell commands and code snippets."""
        print("\nExtracting command patterns...")

        # Look for code blocks and commands
        code_block_pattern = r'```(\w+)?\n(.*?)```'
        command_pattern = r'^\$\s+(.+)$'

        commands = []
        code_blocks = defaultdict(list)

        for exchange in self.exchanges:
            # Check both request and response
            for text_field in ['request_message', 'response_text']:
                text = exchange.get(text_field, '')
                if not text:
                    continue

                # Extract code blocks
                for match in re.finditer(code_block_pattern, text, re.DOTALL):
                    lang = match.group(1) or 'unknown'
                    code = match.group(2)
                    code_blocks[lang].append(code[:100])  # First 100 chars

                # Extract shell commands
                for line in text.split('\n'):
                    cmd_match = re.match(command_pattern, line.strip())
                    if cmd_match:
                        commands.append(cmd_match.group(1))

        self.patterns['code_blocks'] = dict(code_blocks)
        self.patterns['shell_commands'] = Counter(commands).most_common(50)

        self.stats['total_code_blocks'] = sum(len(blocks) for blocks in code_blocks.values())
        self.stats['total_shell_commands'] = len(commands)

        for lang, blocks in code_blocks.items():
            self.stats[f'code_blocks_{lang}'] = len(blocks)

    def analyze_communication_style(self):
        """Analyze communication patterns and style."""
        print("\nAnalyzing communication style...")

        # Question patterns
        question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which']
        questions = 0

        # Imperative vs interrogative
        imperative_patterns = [r'^(create|build|make|add|remove|delete|update|fix|implement)\s+']
        imperatives = 0

        # Politeness markers
        politeness = ['please', 'thanks', 'thank you', 'could you', 'would you']
        polite_count = 0

        # Frustration indicators
        frustration = ['again', 'still', 'why did you', 'I already', 'stop']
        frustration_count = 0

        for exchange in self.exchanges:
            request = exchange.get('request_message', '').lower()
            if not request:
                continue

            # Count questions
            if any(request.strip().startswith(q) for q in question_words) or '?' in request:
                questions += 1

            # Count imperatives
            if any(re.match(p, request.strip(), re.IGNORECASE) for p in imperative_patterns):
                imperatives += 1

            # Count politeness
            if any(p in request for p in politeness):
                polite_count += 1

            # Count frustration
            if any(f in request for f in frustration):
                frustration_count += 1

        total_requests = len([e for e in self.exchanges if e.get('request_message')])

        self.stats['questions'] = questions
        self.stats['imperatives'] = imperatives
        self.stats['polite_requests'] = polite_count
        self.stats['frustration_indicators'] = frustration_count

        if total_requests > 0:
            self.stats['question_ratio'] = questions / total_requests
            self.stats['imperative_ratio'] = imperatives / total_requests
            self.stats['politeness_ratio'] = polite_count / total_requests
            self.stats['frustration_ratio'] = frustration_count / total_requests

    def generate_report(self, output_file: str):
        """Generate markdown report with findings."""
        print(f"\nGenerating report: {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Augment Conversation Data Mining Report\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")

            # Basic Statistics
            f.write("## Basic Statistics\n\n")
            f.write(f"- **Total Conversations**: {self.stats['total_conversations']:,}\n")
            f.write(f"- **Total Exchanges**: {self.stats['total_exchanges']:,}\n")
            f.write(f"- **Exchanges with Request**: {self.stats['exchanges_with_request']:,}\n")
            f.write(f"- **Exchanges with Response**: {self.stats['exchanges_with_response']:,}\n")
            f.write(f"- **Average Exchanges per Conversation**: {self.stats['avg_exchanges_per_conversation']:.1f}\n")
            f.write(f"- **Max Exchanges in a Conversation**: {self.stats['max_exchanges_per_conversation']}\n")
            f.write(f"- **Min Exchanges in a Conversation**: {self.stats['min_exchanges_per_conversation']}\n")
            f.write(f"- **Average Request Length**: {self.stats['avg_request_length']:.0f} chars\n")
            f.write(f"- **Average Response Length**: {self.stats['avg_response_length']:.0f} chars\n\n")

            # Workspace Distribution
            f.write("## Workspace Distribution\n\n")
            f.write("Top 10 workspaces by exchange count:\n\n")
            sorted_folders = sorted(self.stats['folders'].items(), key=lambda x: x[1], reverse=True)[:10]
            for folder, count in sorted_folders:
                folder_name = Path(folder).name if folder else 'Unknown'
                f.write(f"- **{folder_name}**: {count:,} exchanges\n")
            f.write("\n")

            # Explicit Preferences
            f.write("## Explicit Preferences\n\n")
            f.write(f"- **Always statements**: {self.stats.get('explicit_preferences_always', 0)}\n")
            f.write(f"- **Never statements**: {self.stats.get('explicit_preferences_never', 0)}\n")
            f.write(f"- **Prefer statements**: {self.stats.get('explicit_preferences_prefer', 0)}\n")
            f.write(f"- **Don't statements**: {self.stats.get('explicit_preferences_dont', 0)}\n")
            f.write(f"- **Must statements**: {self.stats.get('explicit_preferences_must', 0)}\n")
            f.write(f"- **Should statements**: {self.stats.get('explicit_preferences_should', 0)}\n\n")

            # Sample preferences
            for pref_type in ['always', 'never', 'prefer']:
                patterns = self.patterns.get(f'preference_{pref_type}', [])
                if patterns:
                    f.write(f"### Sample '{pref_type.upper()}' Statements (first 10)\n\n")
                    for i, p in enumerate(patterns[:10], 1):
                        f.write(f"{i}. `{p['text']}` - {Path(p['folder']).name}\n")
                    f.write("\n")

            # Tool Mentions
            f.write("## Tool and Technology Mentions\n\n")
            tool_mentions = self.patterns.get('tool_mentions', {})
            for category, tools in tool_mentions.items():
                if tools:
                    f.write(f"### {category.replace('_', ' ').title()}\n\n")
                    sorted_tools = sorted(tools.items(), key=lambda x: x[1], reverse=True)
                    for tool, count in sorted_tools:
                        f.write(f"- **{tool}**: {count:,} mentions\n")
                    f.write("\n")

            # Code Blocks
            f.write("## Code and Commands\n\n")
            f.write(f"- **Total Code Blocks**: {self.stats.get('total_code_blocks', 0):,}\n")
            f.write(f"- **Total Shell Commands**: {self.stats.get('total_shell_commands', 0):,}\n\n")

            code_blocks = self.patterns.get('code_blocks', {})
            if code_blocks:
                f.write("### Code Blocks by Language\n\n")
                sorted_langs = sorted(code_blocks.items(), key=lambda x: len(x[1]), reverse=True)
                for lang, blocks in sorted_langs[:10]:
                    f.write(f"- **{lang}**: {len(blocks):,} blocks\n")
                f.write("\n")

            shell_commands = self.patterns.get('shell_commands', [])
            if shell_commands:
                f.write("### Most Common Shell Commands (Top 20)\n\n")
                for cmd, count in shell_commands[:20]:
                    f.write(f"- `{cmd}` ({count} times)\n")
                f.write("\n")

            # Corrections
            f.write("## Correction Patterns\n\n")
            f.write(f"- **Total Corrections Detected**: {self.stats.get('correction_count', 0)}\n")
            f.write(f"- **Correction Rate**: {self.stats.get('correction_count', 0) / self.stats['total_exchanges'] * 100:.2f}%\n\n")

            corrections = self.patterns.get('corrections', [])
            if corrections:
                f.write("### Sample Corrections (first 10)\n\n")
                for i, c in enumerate(corrections[:10], 1):
                    f.write(f"{i}. {Path(c['folder']).name}\n")
                    f.write(f"   ```\n   {c['context']}\n   ```\n\n")

            # Communication Style
            f.write("## Communication Style\n\n")
            f.write(f"- **Questions**: {self.stats.get('questions', 0):,} ({self.stats.get('question_ratio', 0)*100:.1f}%)\n")
            f.write(f"- **Imperatives**: {self.stats.get('imperatives', 0):,} ({self.stats.get('imperative_ratio', 0)*100:.1f}%)\n")
            f.write(f"- **Polite Requests**: {self.stats.get('polite_requests', 0):,} ({self.stats.get('politeness_ratio', 0)*100:.1f}%)\n")
            f.write(f"- **Frustration Indicators**: {self.stats.get('frustration_indicators', 0):,} ({self.stats.get('frustration_ratio', 0)*100:.1f}%)\n\n")

            # Data Quality
            f.write("## Data Quality and Coverage\n\n")
            total_exchanges = self.stats['total_exchanges']
            with_request = self.stats['exchanges_with_request']
            with_response = self.stats['exchanges_with_response']

            f.write(f"- **Data Completeness**: {with_request / total_exchanges * 100:.1f}% of exchanges have requests\n")
            f.write(f"- **Response Coverage**: {with_response / total_exchanges * 100:.1f}% of exchanges have responses\n")
            f.write(f"- **Usable Exchanges**: {min(with_request, with_response):,} (both request and response)\n\n")

            # Recommendations
            f.write("## Recommendations for Next Steps\n\n")
            f.write("Based on the data mining results:\n\n")

            if self.stats.get('explicit_preferences_always', 0) > 0:
                f.write("1. **High-confidence preferences detected**: Start with 'always' and 'never' statements for initial digital twin\n")

            if self.stats.get('correction_count', 0) > 100:
                f.write("2. **Significant correction patterns found**: Analyze these to identify recurring LLM mistakes\n")

            if tool_mentions:
                f.write("3. **Tool preferences clear**: Extract tool selection matrix from frequency data\n")

            if self.stats.get('total_code_blocks', 0) > 1000:
                f.write("4. **Rich code examples available**: Can be used for workflow pattern extraction\n")

            f.write("\n")

        print(f"Report saved to: {output_file}")

    def extract_tool_preferences_detailed(self):
        """Extract tool preferences with confidence scores for actionable output."""
        print("\nExtracting detailed tool preferences...")

        tools = {
            'python_package_manager': {
                'uv': [r'\buv\s+(pip|venv|run)\b', r'\buse\s+uv\b'],
                'pip': [r'\bpip\s+install\b', r'\bpip3\s+install\b'],
                'poetry': [r'\bpoetry\s+(add|install)\b'],
            },
            'javascript_package_manager': {
                'pnpm': [r'\bpnpm\s+(install|add)\b', r'\buse\s+pnpm\b'],
                'npm': [r'\bnpm\s+(install|run)\b'],
                'yarn': [r'\byarn\s+(add|install)\b'],
            },
            'python_testing': {
                'pytest': [r'\bpytest\b', r'\buse\s+pytest\b'],
                'unittest': [r'\bunittest\b'],
            },
            'python_web_framework': {
                'fastapi': [r'\bfastapi\b', r'\bFastAPI\b'],
                'flask': [r'\bflask\b', r'\bFlask\b'],
                'django': [r'\bdjango\b', r'\bDjango\b'],
            },
        }

        results = {}

        # Only use complete exchanges
        complete_exchanges = [e for e in self.exchanges if e.get('request_message') and e.get('response_text')]

        for category, tool_patterns in tools.items():
            counts = Counter()

            for exchange in complete_exchanges:
                text = (exchange.get('request_message', '') + ' ' +
                       exchange.get('response_text', '')).lower()

                for tool, patterns in tool_patterns.items():
                    if any(re.search(p, text, re.IGNORECASE) for p in patterns):
                        counts[tool] += 1

            if counts:
                total = sum(counts.values())
                results[category] = {
                    'counts': dict(counts),
                    'total': total,
                    'preference': counts.most_common(1)[0][0],
                    'confidence': counts.most_common(1)[0][1] / total,
                }

        self.patterns['tool_preferences_detailed'] = results

    def extract_hard_constraints(self):
        """Extract NEVER and ALWAYS rules as hard constraints."""
        print("\nExtracting hard constraints...")

        constraints = {
            'never': [],
            'always': [],
        }

        for exchange in self.exchanges:
            request = exchange.get('request_message', '')
            if not request:
                continue

            # NEVER rules (high priority)
            never_patterns = [
                r'\b(NEVER|never)\s+([A-Z][^.!?\n]{10,100})',
                r'\b(DON\'T|don\'t|DO NOT)\s+(EVER|ever)\s+([^.!?\n]{10,100})',
            ]

            for pattern in never_patterns:
                for match in re.finditer(pattern, request):
                    rule_text = match.group(0).strip()
                    # Filter out project-specific rules
                    if len(rule_text) > 20 and not any(x in rule_text.lower() for x in ['conversationid', 'username', 'session']):
                        constraints['never'].append({
                            'rule': rule_text,
                            'folder': exchange.get('_folder_path', ''),
                        })

            # ALWAYS rules
            always_patterns = [
                r'\b(ALWAYS|always)\s+([A-Z][^.!?\n]{10,100})',
            ]

            for pattern in always_patterns:
                for match in re.finditer(pattern, request):
                    rule_text = match.group(0).strip()
                    if len(rule_text) > 20 and not any(x in rule_text.lower() for x in ['conversationid', 'username', 'session']):
                        constraints['always'].append({
                            'rule': rule_text,
                            'folder': exchange.get('_folder_path', ''),
                        })

        self.patterns['hard_constraints'] = constraints

    def extract_anti_patterns(self):
        """Extract anti-patterns from corrections."""
        print("\nExtracting anti-patterns...")

        anti_patterns = []

        # Patterns indicating LLM mistakes
        mistake_indicators = [
            (r'why (did you|are you) (chang|edit|creat|delet|remov)', 'unauthorized_change'),
            (r'I (said|told you|asked you) (no|not to|don\'t)', 'ignored_instruction'),
            (r'(DON\'T|DO NOT) (CREATE|EDIT|CHANGE|DELETE)', 'prohibited_action'),
            (r'why are you (creating|making|adding) (\w+)', 'unsolicited_creation'),
        ]

        for exchange in self.exchanges:
            request = exchange.get('request_message', '')
            if not request:
                continue

            for pattern, category in mistake_indicators:
                if re.search(pattern, request, re.IGNORECASE):
                    anti_patterns.append({
                        'category': category,
                        'example': request[:150],
                        'folder': exchange.get('_folder_path', ''),
                    })
                    break

        # Aggregate by category
        by_category = defaultdict(list)
        for ap in anti_patterns:
            by_category[ap['category']].append(ap)

        self.patterns['anti_patterns'] = dict(by_category)

    def generate_consolidated_report(self, output_file: str):
        """Generate single consolidated markdown report with all findings."""
        print(f"\nGenerating consolidated report: {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Augment Conversation Data Mining - Consolidated Report\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
            f.write("---\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            complete_exchanges = len([e for e in self.exchanges if e.get('request_message') and e.get('response_text')])
            f.write(f"Analyzed **{self.stats['total_exchanges']:,} exchanges** from **{self.stats['total_conversations']} conversations** across **{len(self.stats['workspaces'])} workspaces**.\n\n")
            f.write(f"- **Complete exchanges** (request + response): {complete_exchanges:,} ({complete_exchanges/self.stats['total_exchanges']*100:.1f}%)\n")
            f.write(f"- **Tool preferences identified**: {len(self.patterns.get('tool_preferences_detailed', {}))}\n")
            f.write(f"- **Hard constraints found**: {len(self.patterns.get('hard_constraints', {}).get('never', [])) + len(self.patterns.get('hard_constraints', {}).get('always', []))}\n")
            f.write(f"- **Correction instances**: {self.stats.get('correction_count', 0)} ({self.stats.get('correction_count', 0)/self.stats['total_exchanges']*100:.2f}%)\n\n")
            f.write("---\n\n")

            # Data Quality
            f.write("## Data Quality Assessment\n\n")
            f.write(f"- **Total Conversations**: {self.stats['total_conversations']:,}\n")
            f.write(f"- **Total Exchanges**: {self.stats['total_exchanges']:,}\n")
            f.write(f"- **Exchanges with Request**: {self.stats['exchanges_with_request']:,} ({self.stats['exchanges_with_request']/self.stats['total_exchanges']*100:.1f}%)\n")
            f.write(f"- **Exchanges with Response**: {self.stats['exchanges_with_response']:,} ({self.stats['exchanges_with_response']/self.stats['total_exchanges']*100:.1f}%)\n")
            f.write(f"- **Usable Exchanges**: {complete_exchanges:,} (both request and response)\n\n")
            f.write("**Note**: Low request coverage (17.7%) is due to Augment storing intermediate tool calls and streaming responses as separate exchanges.\n\n")
            f.write("---\n\n")

            # Basic Statistics
            f.write("## Basic Statistics\n\n")
            f.write(f"- **Average Exchanges per Conversation**: {self.stats['avg_exchanges_per_conversation']:.1f}\n")
            f.write(f"- **Max Exchanges in a Conversation**: {self.stats['max_exchanges_per_conversation']}\n")
            f.write(f"- **Min Exchanges in a Conversation**: {self.stats['min_exchanges_per_conversation']}\n")
            f.write(f"- **Average Request Length**: {self.stats['avg_request_length']:.0f} chars\n")
            f.write(f"- **Average Response Length**: {self.stats['avg_response_length']:.0f} chars\n\n")

            # Workspace Distribution
            f.write("### Top 10 Workspaces by Exchange Count\n\n")
            sorted_folders = sorted(self.stats['folders'].items(), key=lambda x: x[1], reverse=True)[:10]
            for folder, count in sorted_folders:
                folder_name = Path(folder).name if folder else 'Unknown'
                f.write(f"- **{folder_name}**: {count:,} exchanges\n")
            f.write("\n---\n\n")

            # Tool Preferences (Actionable)
            f.write("## Tool Preferences (Actionable)\n\n")
            tool_prefs = self.patterns.get('tool_preferences_detailed', {})

            if tool_prefs:
                for category, data in tool_prefs.items():
                    f.write(f"### {category.replace('_', ' ').title()}\n\n")
                    f.write(f"**Preference**: `{data['preference']}`  \n")
                    f.write(f"**Confidence**: {data['confidence']:.1%}\n\n")

                    f.write("**Usage Distribution**:\n")
                    sorted_tools = sorted(data['counts'].items(), key=lambda x: x[1], reverse=True)
                    for tool, count in sorted_tools:
                        pct = count / data['total'] * 100
                        f.write(f"- {tool}: {count} ({pct:.1f}%)\n")
                    f.write("\n")

                    # Generate instruction
                    if data['confidence'] > 0.7:
                        f.write(f"**Instruction**: Always use `{data['preference']}` for {category.replace('_', ' ')}\n\n")

            f.write("---\n\n")

            # Hard Constraints
            f.write("## Hard Constraints\n\n")
            constraints = self.patterns.get('hard_constraints', {})

            if constraints:
                f.write("### NEVER Rules\n\n")
                never_rules = constraints.get('never', [])
                unique_never = {}
                for rule in never_rules:
                    rule_lower = rule['rule'].lower()
                    if rule_lower not in unique_never:
                        unique_never[rule_lower] = rule

                for i, rule in enumerate(list(unique_never.values())[:20], 1):
                    f.write(f"{i}. {rule['rule']}\n")

                if len(unique_never) > 20:
                    f.write(f"\n... and {len(unique_never) - 20} more\n")
                f.write("\n")

                f.write("### ALWAYS Rules\n\n")
                always_rules = constraints.get('always', [])
                unique_always = {}
                for rule in always_rules:
                    rule_lower = rule['rule'].lower()
                    if rule_lower not in unique_always:
                        unique_always[rule_lower] = rule

                for i, rule in enumerate(list(unique_always.values())[:20], 1):
                    f.write(f"{i}. {rule['rule']}\n")

                if len(unique_always) > 20:
                    f.write(f"\n... and {len(unique_always) - 20} more\n")
                f.write("\n")

            f.write("---\n\n")

            # Anti-Patterns
            f.write("## Anti-Patterns (What NOT to Do)\n\n")
            anti_patterns = self.patterns.get('anti_patterns', {})

            if anti_patterns:
                f.write("Extracted from user corrections:\n\n")

                for category, examples in sorted(anti_patterns.items(), key=lambda x: len(x[1]), reverse=True):
                    f.write(f"### {category.replace('_', ' ').title()} ({len(examples)} instances)\n\n")

                    # Show a few examples
                    for i, ex in enumerate(examples[:3], 1):
                        f.write(f"{i}. `{ex['example'][:100]}...`\n")
                    f.write("\n")

                    # Generate instruction
                    if category == 'unauthorized_change':
                        f.write("**Instruction**: Always ask permission before editing, creating, or deleting files\n\n")
                    elif category == 'ignored_instruction':
                        f.write("**Instruction**: Pay close attention to explicit user instructions, especially negations\n\n")
                    elif category == 'unsolicited_creation':
                        f.write("**Instruction**: Do not create files, documentation, or scripts unless explicitly requested\n\n")
                    elif category == 'prohibited_action':
                        f.write("**Instruction**: Respect explicit prohibitions and constraints\n\n")

            f.write("---\n\n")

            # Explicit Preferences (from original analysis)
            f.write("## Explicit Preferences\n\n")
            f.write(f"- **Always statements**: {self.stats.get('explicit_preferences_always', 0)}\n")
            f.write(f"- **Never statements**: {self.stats.get('explicit_preferences_never', 0)}\n")
            f.write(f"- **Prefer statements**: {self.stats.get('explicit_preferences_prefer', 0)}\n")
            f.write(f"- **Don't statements**: {self.stats.get('explicit_preferences_dont', 0)}\n")
            f.write(f"- **Must statements**: {self.stats.get('explicit_preferences_must', 0)}\n")
            f.write(f"- **Should statements**: {self.stats.get('explicit_preferences_should', 0)}\n\n")

            # Sample preferences
            for pref_type in ['always', 'never', 'prefer']:
                patterns = self.patterns.get(f'preference_{pref_type}', [])
                if patterns:
                    f.write(f"### Sample '{pref_type.upper()}' Statements (first 10)\n\n")
                    for i, p in enumerate(patterns[:10], 1):
                        f.write(f"{i}. `{p['text']}` - {Path(p['folder']).name}\n")
                    f.write("\n")

            f.write("---\n\n")

            # Tool Mentions (from original analysis)
            f.write("## Tool and Technology Mentions\n\n")
            tool_mentions = self.patterns.get('tool_mentions', {})
            for category, tools in tool_mentions.items():
                if tools:
                    f.write(f"### {category.replace('_', ' ').title()}\n\n")
                    sorted_tools = sorted(tools.items(), key=lambda x: x[1], reverse=True)
                    for tool, count in sorted_tools:
                        f.write(f"- **{tool}**: {count:,} mentions\n")
                    f.write("\n")

            f.write("---\n\n")

            # Corrections
            f.write("## Correction Patterns\n\n")
            f.write(f"- **Total Corrections Detected**: {self.stats.get('correction_count', 0)}\n")
            f.write(f"- **Correction Rate**: {self.stats.get('correction_count', 0) / self.stats['total_exchanges'] * 100:.2f}%\n\n")

            corrections = self.patterns.get('corrections', [])
            if corrections:
                f.write("### Sample Corrections (first 10)\n\n")
                for i, c in enumerate(corrections[:10], 1):
                    f.write(f"{i}. {Path(c['folder']).name}\n")
                    f.write(f"   ```\n   {c['context']}\n   ```\n\n")

            f.write("---\n\n")

            # Communication Style
            f.write("## Communication Style\n\n")
            f.write(f"- **Questions**: {self.stats.get('questions', 0):,} ({self.stats.get('question_ratio', 0)*100:.1f}%)\n")
            f.write(f"- **Imperatives**: {self.stats.get('imperatives', 0):,} ({self.stats.get('imperative_ratio', 0)*100:.1f}%)\n")
            f.write(f"- **Polite Requests**: {self.stats.get('polite_requests', 0):,} ({self.stats.get('politeness_ratio', 0)*100:.1f}%)\n")
            f.write(f"- **Frustration Indicators**: {self.stats.get('frustration_indicators', 0):,} ({self.stats.get('frustration_ratio', 0)*100:.1f}%)\n\n")

            f.write("---\n\n")

            # Code and Commands
            f.write("## Code and Commands\n\n")
            f.write(f"- **Total Code Blocks**: {self.stats.get('total_code_blocks', 0):,}\n")
            f.write(f"- **Total Shell Commands**: {self.stats.get('total_shell_commands', 0):,}\n\n")

            code_blocks = self.patterns.get('code_blocks', {})
            if code_blocks:
                f.write("### Code Blocks by Language\n\n")
                sorted_langs = sorted(code_blocks.items(), key=lambda x: len(x[1]), reverse=True)
                for lang, blocks in sorted_langs[:10]:
                    f.write(f"- **{lang}**: {len(blocks):,} blocks\n")
                f.write("\n")

            shell_commands = self.patterns.get('shell_commands', [])
            if shell_commands:
                f.write("### Most Common Shell Commands (Top 20)\n\n")
                for cmd, count in shell_commands[:20]:
                    f.write(f"- `{cmd}` ({count} times)\n")
                f.write("\n")

            f.write("---\n\n")

            # Recommendations
            f.write("## Recommendations for Next Steps\n\n")
            f.write("Based on the data mining results:\n\n")

            if self.stats.get('explicit_preferences_always', 0) > 0:
                f.write("1. **High-confidence preferences detected**: Start with 'always' and 'never' statements for initial digital twin\n")

            if self.stats.get('correction_count', 0) > 100:
                f.write("2. **Significant correction patterns found**: Analyze these to identify recurring LLM mistakes\n")

            if tool_mentions:
                f.write("3. **Tool preferences clear**: Extract tool selection matrix from frequency data\n")

            if self.stats.get('total_code_blocks', 0) > 1000:
                f.write("4. **Rich code examples available**: Can be used for workflow pattern extraction\n")

            f.write("\n")

        print(f"Report saved to: {output_file}")

    def run_full_analysis(self, output_file: str = "pattern_analysis.md"):
        """Run complete consolidated analysis pipeline."""
        self.load_all_conversations()
        self.compute_basic_statistics()
        self.extract_explicit_preferences()
        self.extract_tool_mentions()
        self.identify_correction_patterns()
        self.extract_command_patterns()
        self.analyze_communication_style()
        self.extract_tool_preferences_detailed()
        self.extract_hard_constraints()
        self.extract_anti_patterns()
        self.generate_consolidated_report(output_file)

        print("\n✓ Consolidated analysis complete!")


def main():
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "augment_conversations_export_leveldb"

    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = "pattern_analysis.md"

    miner = ConversationMiner(data_dir)
    miner.run_full_analysis(output_file)


if __name__ == "__main__":
    main()



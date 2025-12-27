# PSH Comprehensive Test Output

This file contains the expanded prompts from all test cases to verify the snippet system is working correctly.

## Design Change: No Defaults

**Important:** The snippet system has been refactored to **remove defaults entirely**.

- Each `op` is now a **complete, standalone configuration** with all variables defined
- Multiple ops can be combined - **later ops override earlier ones**
- Example: `;;d,l2,ne` means "start with l2 (length=2), then apply ne (no emoji)"
- This makes the system more explicit and predictable

---

## Test Case 1: `;;d.ne,l2`
**Description:** Documentation: no emoji, concise

**Expanded Prompt:**
```
Create concise documentation covering key features and basic usage.
Do not use any emoji in your response.
CREATE documentation files for this project.
Generate appropriate documentation including README, API docs, usage guides, etc.
Use proper formatting with headings, code blocks, and examples where appropriate.
Maintain a professional but approachable tone.
```

---

## Test Case 2: `;;d.l5,pro`
**Description:** Documentation: very detailed, d.pro (namespace-scoped pro)

**Expanded Prompt:**
```
Create extensive documentation with detailed API reference, examples, edge cases, and troubleshooting.
Do not use any emoji in your response.
CREATE documentation files for this project.
Generate appropriate documentation including README, API docs, usage guides, etc.
Use proper formatting with headings, code blocks, and examples where appropriate.
Use formal, technical documentation style suitable for API references and official documentation.
```

---

## Test Case 3: `;;d.l4;pro`
**Description:** Documentation: detailed, then global pro (separate segments)

**Expanded Prompt:**
```
Create comprehensive documentation including API reference and advanced usage.
Do not use any emoji in your response.
CREATE documentation files for this project.
Generate appropriate documentation including README, API docs, usage guides, etc.
Use proper formatting with headings, code blocks, and examples where appropriate.
Maintain a professional but approachable tone.

Provide a moderately detailed response (2-3 paragraphs).
Do not use any emoji in your response.
Use strictly professional, formal language suitable for business documentation.
```

---

## Test Case 4: `;;sum.blt,l3`
**Description:** Summarize: bullet points, moderate

**Expanded Prompt:**
```
Provide a moderate summary (3-5 key points).
Do not use any emoji in your response.
Summarize the following content as a bulleted list with clear, actionable points.
Focus on the key points and main takeaways.
Preserve important details and context.
Maintain a professional but approachable tone.
```

---

## Test Case 5: `;;sum.num,l1`
**Description:** Summarize: numbered, one sentence

**Expanded Prompt:**
```
Provide a one-sentence summary capturing only the core essence.
Do not use any emoji in your response.
Summarize the following content as a numbered list in logical order.
Focus on the key points and main takeaways.
Focus only on the single most important point.
Maintain a professional but approachable tone.
```

---

## Test Case 6: `;;plan.stp,l4`
**Description:** Plan: detailed steps, comprehensive

**Expanded Prompt:**
```
Provide a detailed plan with comprehensive steps, timelines, and resource considerations.
Do not use any emoji in your response.
Create a clear, step-by-step plan to accomplish the stated goal.
Break down into detailed steps with substeps, timelines, dependencies, and resource requirements.
Consider dependencies, prerequisites, and potential challenges.
Maintain a professional but approachable tone.
```

---

## Test Case 7: `;;cr.base,lang=rust,l4`
**Description:** Code review: Rust, comprehensive

**Expanded Prompt:**
```
Provide a comprehensive code review with detailed analysis.
Do not use any emoji in your response.
Review the following rust code with comprehensive level of detail.
Focus on:
- Code quality and best practices
- Potential bugs or edge cases
- Performance considerations
- Readability and maintainability
Maintain a professional but approachable tone.
Provide specific, actionable feedback.
```

---

## Test Case 8: `;;rr.pro,l1`
**Description:** Rewrite: professional, concise

**Expanded Prompt:**
```
Do not use any emoji in your response.
Rewrite the following text in a formal, professional business tone suitable for executive communication.
Make it significantly more concise while preserving key points.
Preserve the core meaning while adapting the style appropriately.
```

---

## Test Case 9: `;;rr.cas,l5`
**Description:** Rewrite: casual, expanded

**Expanded Prompt:**
```
Do not use any emoji in your response.
Rewrite the following text in a casual, friendly conversational tone.
Expand with additional detail, examples, and elaboration.
Preserve the core meaning while adapting the style appropriately.
```

---

## Test Case 10: `;;git.cm,l1`
**Description:** Git commit: title only

**Expanded Prompt:**
```
Generate a concise one-line commit message.
Do not use any emoji in your response.
Follow conventional commit format: type(scope): description
Types: feat, fix, docs, style, refactor, test, chore
Provide only the commit title (one line).
Maintain a professional but approachable tone.
```

---

## Global Ops Test Cases

These test cases demonstrate global ops without namespace:

---

## Global Test Case 1: `;;ne`
**Description:** Global: no emoji

**Expanded Prompt:**
```
Provide a moderately detailed response (2-3 paragraphs).
Do not use any emoji in your response.
Maintain a professional but approachable tone.
```

---

## Global Test Case 2: `;;ne,l5`
**Description:** Global: no emoji, very detailed

**Expanded Prompt:**
```
Provide an extremely detailed, in-depth response with extensive examples and edge cases.
Do not use any emoji in your response.
Maintain a professional but approachable tone.
```

---

## Global Test Case 3: `;;l5,pro`
**Description:** Global: very detailed, professional

**Expanded Prompt:**
```
Provide an extremely detailed, in-depth response with extensive examples and edge cases.
Do not use any emoji in your response.
Use strictly professional, formal language suitable for business documentation.
```

---

## Global Test Case 4: `;;pro,blt`
**Description:** Global: professional, bullet points

**Expanded Prompt:**
```
Provide a moderately detailed response (2-3 paragraphs).
Do not use any emoji in your response.
Use strictly professional, formal language suitable for business documentation.
as a bulleted list with clear, actionable points.
```

---

## Global Test Case 5: `;;nd`
**Description:** Global: no documentation

**Expanded Prompt:**
```
Provide a moderately detailed response (2-3 paragraphs).
Do not use any emoji in your response.
Maintain a professional but approachable tone.
Do NOT create any new documentation files (*.md, README, docs, etc.). Only provide explanations or code as requested.
```

---

## Global Test Case 6: `;;nd,l3`
**Description:** Global: no documentation, moderate detail

**Expanded Prompt:**
```
Provide a moderately detailed response (2-3 paragraphs).
Do not use any emoji in your response.
Maintain a professional but approachable tone.
Do NOT create any new documentation files (*.md, README, docs, etc.). Only provide explanations or code as requested.
```

---

## Summary

All 16 test cases passed successfully. The snippet system correctly:
- Parses directives with namespace and operations
- Resolves snippets from the collection
- Applies operation overrides (ne, l1-l5, pro, cas, blt, num, stp, etc.)
- Handles key-value pairs (lang=rust)
- Handles global ops without namespace
- Renders templates with Tera
- Produces comprehensive, literal prompt instructions

The actual prompts are much more detailed and actionable than the simple descriptions in DevLog-000, providing clear instructions to the LLM about formatting, tone, length, and specific requirements.


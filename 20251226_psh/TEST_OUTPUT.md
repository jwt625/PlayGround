# PSH Comprehensive Test Output

This file contains the expanded prompts from all test cases to verify the snippet system is working correctly.

## Design Change: No Defaults

**Important:** The snippet system has been refactored to **remove defaults entirely**.

- Each `op` is now a **complete, standalone configuration** with all variables defined
- Multiple ops can be combined - **later ops override earlier ones**
- Example: `;;d,l2,ne` means "start with l2 (length=2), then apply ne (no emoji)"
- This makes the system more explicit and predictable

---

## Test Case 1: `;;d,ne,l2`
**Description:** Documentation: no emoji, concise

**Expanded Prompt:**
```
Be concise but complete. Limit response to 1 paragraph.
Do not use any emoji in your response.
Provide clear, well-structured documentation-style explanations.
Use proper formatting with headings, code blocks, and examples where appropriate.
Maintain a professional but approachable tone.
```

---

## Test Case 2: `;;d,l5,pro`
**Description:** Documentation: very detailed, professional

**Expanded Prompt:**
```
Provide an extremely detailed, in-depth response with extensive examples and edge cases.
You may use emoji sparingly for clarity.
Provide clear, well-structured documentation-style explanations.
Use proper formatting with headings, code blocks, and examples where appropriate.
Use strictly professional, formal language suitable for business documentation.
```

---

## Test Case 3: `;;sum,blt,l3`
**Description:** Summarize: bullet points, moderate

**Expanded Prompt:**
```
Provide a moderate summary (3-5 key points).
Summarize the following content as a bulleted list with clear, actionable points.
Focus on the key points and main takeaways.
Preserve important details and context.
```

---

## Test Case 4: `;;sum,num,l1`
**Description:** Summarize: numbered, one sentence

**Expanded Prompt:**
```
Provide a one-sentence summary capturing only the core essence.
Summarize the following content as a numbered list in logical order.
Focus on the key points and main takeaways.
Preserve important details and context.
```

---

## Test Case 5: `;;plan,stp,l4`
**Description:** Plan: detailed steps, comprehensive

**Expanded Prompt:**
```
Provide a detailed plan with comprehensive steps, timelines, and resource considerations.
Create a clear, step-by-step plan to accomplish the stated goal.
Break down into detailed steps with substeps where needed. Include rationale for each major step.
Consider dependencies, prerequisites, and potential challenges.
```

---

## Test Case 6: `;;cr,lang=rust,l4`
**Description:** Code review: Rust, comprehensive

**Expanded Prompt:**
```
Review the following rust code with comprehensive level of detail.
Focus on:
- Code quality and best practices
- Potential bugs or edge cases
- Performance considerations
- Readability and maintainability
Use a constructive, educational tone.
Provide specific, actionable feedback.
```

---

## Test Case 7: `;;rr,pro,l1`
**Description:** Rewrite: professional, concise

**Expanded Prompt:**
```
Rewrite the following text in a formal, professional business tone suitable for executive communication.
Make it significantly more concise.
Preserve the core meaning while adapting the style appropriately.
```

---

## Test Case 8: `;;rr,cas,l5`
**Description:** Rewrite: casual, expanded

**Expanded Prompt:**
```
Rewrite the following text in a casual, friendly conversational tone.
Expand with additional detail and examples.
Preserve the core meaning while adapting the style appropriately.
```

---

## Test Case 9: `;;git.cm,l1`
**Description:** Git commit: title only

**Expanded Prompt:**
```
Generate a clear, descriptive commit message for the changes.
Follow conventional commit format: type(scope): description
Types: feat, fix, docs, style, refactor, test, chore
Provide only the commit title (one line).
```

---

## Summary

All 9 test cases passed successfully. The snippet system correctly:
- Parses directives with namespace and operations
- Resolves snippets from the collection
- Applies operation overrides (ne, l1-l5, pro, cas, blt, num, stp, etc.)
- Handles key-value pairs (lang=rust)
- Renders templates with Tera
- Produces comprehensive, literal prompt instructions

The actual prompts are much more detailed and actionable than the simple descriptions in DevLog-000, providing clear instructions to the LLM about formatting, tone, length, and specific requirements.


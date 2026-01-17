#!/bin/bash
# Consolidate insight types from ~190 variants to 7 canonical types
# Adds "canonical_type" field to each insight, preserving original "type"

set -e

MAPPING='{
  "workflow": [
    "workflow_pattern", "workflow", "workflow_preference", "interaction_pattern",
    "development_approach", "implementation_approach", "debugging_approach",
    "debugging_pattern", "debugging_style", "debugging_preference", "testing_approach",
    "testing_pattern", "testing_preference", "planning_approach", "planning_preference",
    "mvp_strategy", "problem_solving", "investigation_method", "fix_strategy",
    "recovery_strategy", "recovery_pattern", "analysis_approach", "refactoring_preference",
    "decision_approach", "decision_pattern"
  ],
  "constraint": [
    "constraint", "requirement", "functional_requirement", "logic_requirement",
    "feature_requirement", "data_requirement", "data_requirements", "scalability_requirement",
    "consistency_requirement", "technical_requirement", "domain_requirement",
    "analysis_requirement", "precision_constraint", "environment_constraint",
    "design_constraint", "resource_constraint", "logic_preference"
  ],
  "quality": [
    "quality_standard", "quality_standards", "code_quality", "code_style",
    "code_organization", "code_structure", "code_pattern", "coding_convention",
    "coding_preference", "naming_convention", "notation_convention", "file_organization",
    "file_convention", "documentation_standard", "documentation_practice",
    "documentation_style", "documentation_preference", "documentation_pattern",
    "syntax_preference", "code_review_approach"
  ],
  "communication": [
    "communication_preference", "communication", "communication_style", "tone_preference",
    "interaction_style", "interaction_preference", "expectation", "technical_expectation",
    "expected_behavior", "frustration", "correction", "approval", "clarification",
    "context", "context_preference", "context_clarification", "background",
    "clarification_pattern"
  ],
  "tool": [
    "tool_preference", "tool_usage", "tool_usage_correction", "framework",
    "integration_preference", "integration_pattern", "configuration", "configuration_pattern",
    "deployment_preference", "deployment_config", "infrastructure_preference",
    "hosting_preference", "database_preference", "environment_preference"
  ],
  "ui_ux": [
    "ui_preference", "ux_preference", "ui_pattern", "ux_pattern", "ui_requirement",
    "ui_pattern_preference", "ux_standard", "ux_concern", "visual_preference",
    "visualization_preference", "visualization_pattern", "layout_preference",
    "design_preference", "design_pattern", "design_philosophy", "design_rule",
    "aesthetic_preference", "styling_preference", "stylistic_preference",
    "responsive_design", "content_preference", "content_strategy", "branding_consistency",
    "data_display_preference"
  ],
  "architecture": [
    "architecture_preference", "architectural_preference", "architectural_expectation",
    "architecture_pattern", "architecture_detail", "project_structure",
    "data_model_preference", "data_handling", "data_preference", "data_format",
    "data_format_preference", "data_processing", "data_validation", "state_management",
    "domain_pattern", "domain_logic", "domain_knowledge", "domain_preference",
    "domain_insight", "data_source_preference", "data_source_pattern"
  ]
}'

# Build reverse lookup: type -> canonical
build_lookup() {
  echo "$MAPPING" | jq -r '
    to_entries | map(.key as $canonical | .value[] | {(.): $canonical}) | add
  '
}

LOOKUP=$(build_lookup)

process_file() {
  local input="$1"
  local output="$2"
  
  echo "Processing: $input -> $output"
  
  jq --argjson lookup "$LOOKUP" '
    .results = [.results[] | 
      if .insights != null then
        .insights = [.insights[] |
          .canonical_type = ($lookup[.type] // "misc")
        ]
      else . end
    ]
  ' "$input" > "$output"
  
  echo "Done: $(jq '.results | length' "$output") results processed"
}

# Process main dataset
process_file \
  "analysis/classification_results/stage2_classification_results.json" \
  "analysis/classification_results/stage2_consolidated.json"

# Process archive dataset
process_file \
  "augment_export_archive/analysis/classification_results/stage2_classification_results.json" \
  "augment_export_archive/analysis/classification_results/stage2_consolidated.json"

# Print summary
echo ""
echo "=== Consolidation Summary ==="
for f in analysis/classification_results/stage2_consolidated.json augment_export_archive/analysis/classification_results/stage2_consolidated.json; do
  echo ""
  echo "File: $f"
  jq -r '.results[].insights[]?.canonical_type' "$f" | sort | uniq -c | sort -rn
done


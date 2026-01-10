# DevLog-002: Search and Advanced Filtering Enhancement

## Overview

This document outlines the implementation plan for comprehensive search and advanced filtering capabilities for the Claude Log Viewer.

## Current State Analysis

The log viewer currently supports:
- Basic status filtering (all, success, errors)
- Duration range filtering with min/max sliders
- Static filter implementation in `App.jsx:40-55`

## Available Log Data

### Log Structure Key Fields

**Request Metadata:**
- `timestamp` - ISO 8601 format timestamp
- `method` - HTTP method (e.g., POST)
- `path` - API endpoint path
- `url` - Full request URL

**Request Body:**
- `model` - Request model identifier
- `messages[]` - User conversation messages
- `system[]` - System instructions
- `tools[]` - Available tools in request
- `metadata.user_id` - User identifier
- `temperature` - Temperature parameter (0-2)
- `max_tokens` - Token limit
- `stream` - Boolean streaming flag

**Response:**
- `status` - HTTP status code
- `body.id` - Response identifier
- `body.model` - Actual model used
- `body.usage` - Token usage input/output/total
- `body.content[]` - Response content array
- `body.stop_reason` - Stop condition (e.g., "end_turn")
- `body.stop_sequence` - Stop sequence if applicable
- `duration_ms` - Request duration in milliseconds

### Available Filters Identified

**Request Filters:**
- Model: Request model selection
- Endpoint/Path: API endpoint path
- Tools Available: Tools provided in request
- HTTP Method: Request method type
- Temperature: Temperature parameter range
- Max Tokens: Token limit
- Stream: Streaming enabled/disabled
- User ID: User metadata identifier

**Response Filters:**
- Response Model: Actual model used
- Status Code: Specific HTTP status codes (200, 404, 500, etc.)
- Stop Reason: Stop condition type
- Duration: Request duration (already implemented)
- Token Usage: Input/output/total token counts
- Tools Used: Tools actually invoked during response

**Content-Based Search:**
- User Message: Text content from user messages
- Assistant Response: Text content from assistant response
- Tool Names: Names of available and used tools
- Request Body: Full JSON request body
- Response Body: Full JSON response body

**Temporal Filters:**
- Date Range: Start and end date filtering
- Time Range: Time of day filtering

### Available Tools

From `scripts/tools_extracted.json`:

**Code Operations:**
- Task - Launch specialized subagent
- Bash - Execute bash commands
- BashOutput - Retrieve background shell output
- KillShell - Terminate background shell

**File Operations:**
- Glob - File pattern matching
- Grep - Content search with regex
- Read - Read file contents
- Edit - Exact string replacement
- Write - Write file contents
- NotebookEdit - Jupyter notebook editing

**Development Tools:**
- WebFetch - Fetch and analyze web content
- WebSearch - Web search with up-to-date information
- AskUserQuestion - Query user during execution

**Planning & Workflow:**
- ExitPlanMode - Signal planning completion
- EnterPlanMode - Enter comprehensive planning mode
- TodoWrite - Task list management

**Specialized:**
- Skill - Execute specialized skills
- SlashCommand - Execute custom slash commands

## Proposed Architecture

### State Management

**Search State:**
- `searchQuery` - Active search text
- `searchType` - Text or regex mode
- `searchFields` - Array of fields to search in

**Advanced Filters:**
- `model[]` - Multi-select model filter
- `path[]` - Multi-select endpoint filter
- `tools[]` - Tools available filter
- `toolsUsed[]` - Tools invoked filter
- `statusCodes[]` - Specific HTTP status codes
- `userId` - User ID exact match
- `stopReason[]` - Multi-select stop reasons
- `temperatureRange` - Min/max temperature
- `tokensRange` - Token count range
- `toolUseCount` - Number of tools used range
- `dateRange` - Start/end date range

### Search Functionality

**Text Search:** Case-insensitive substring matching across selected fields

**Regex Search:** Full regular expression support with error handling

**Field Selection:** User can choose which fields to search:
- User message content
- Assistant response text
- Request body (JSON)
- Response body (JSON)
- Tool names

**Match Highlighting:** Highlight matching text within filtered results

### Filter Logic

All filters use AND logic - a log entry must match all active filters:
1. Status filter (existing)
2. Duration filter (existing)
3. Search query match
4. Advanced field filters
5. Range filters (temperature, tokens, etc.)
6. Date/time filters

### Dynamic Options

Filter options populated dynamically from log data:
- Extract unique models from request/response
- Extract unique API paths
- Extract unique tool names
- Extract unique stop reasons
- Extract unique user IDs

Options computed once and memoized for performance.

## Implementation Phases

### Phase 1: Search Foundation (Day 1-2)

**Goals:** Implement core search functionality

**Tasks:**
1. Add search state variables to App.jsx
2. Implement matchSearchQuery function with text/regex modes
3. Create SearchBar component with:
   - Text input field
   - Search type dropdown (text/regex)
   - Field selection checkboxes
4. Integrate search with existing filteredLogs useMemo
5. Add search match highlighting in log entries

**Components Created:**
- `SearchBar.jsx` - Search input and field selection

### Phase 2: Advanced Filters UI (Day 3-4)

**Goals:** Build comprehensive filter interface

**Tasks:**
1. Add advancedFilters state object
2. Implement matchAdvancedFilters function with all filter types
3. Create AdvancedFilters component with sections for:
   - Model multi-select dropdown
   - Endpoint/Path multi-select
   - Tools available multi-select
   - Tools used multi-select
   - Status code multi-select
   - Stop reason multi-select
   - User ID text input
   - Temperature range dual slider
   - Token count min/max sliders
   - Tool use count min/max sliders
   - Date range picker
   - Clear all filters button
4. Implement helper components:
   - `MultiSelect.jsx` - Multi-select dropdown
   - `DualSlider.jsx` - Range slider
   - `MinSlider.jsx` / `MaxSlider.jsx` - Individual range sliders

**Components Created:**
- `AdvancedFilters.jsx` - Main filter interface
- `MultiSelect.jsx` - Multi-select dropdown component
- `DualSlider.jsx` - Range slider component

### Phase 3: Dynamic Options (Day 5-6)

**Goals:** Auto-populate filter options from data

**Tasks:**
1. Implement extractFilterOptions function to gather unique values
2. Add filterOptions memoization in App.jsx
3. Populate filter dropdowns dynamically on log load
4. Add filter count badges showing active filter count
5. Handle edge cases for empty logs or missing fields

**Functions Created:**
- `extractFilterOptions(logs)` - Extract unique values for all filter types

### Phase 4: Filter Persistence (Day 7-8)

**Goals:** Save and restore filter state

**Tasks:**
1. Implement saveFilters/loadFilters functions using localStorage
2. Auto-save filter state on change
3. Restore filters on page load
4. Add "Save Filter Set" feature with custom names
5. Implement "Load Saved Filters" dropdown
6. Add delete saved filter functionality

**Features Added:**
- Automatic filter persistence
- Named filter presets
- Filter sharing via URL parameters (optional)

### Phase 5: Performance Optimization (Day 9-10)

**Goals:** Ensure responsive performance with large datasets

**Tasks:**
1. Add 500ms debounce to search input
2. Optimize filter computation with useMemo
3. Implement virtual scrolling for filter dropdown lists
4. Add progressive filtering for slow queries
5. Benchmark performance with 10K+ log entries
6. Add loading states for expensive filters

**Optimizations:**
- Debounced search input
- Memoized filter computations
- Virtualized long lists
- Progressive results rendering

### Phase 6: Advanced Features (Day 11-14)

**Goals:** Add professional-grade search features

**Tasks:**
1. Implement filter grouping (AND/OR logic between groups)
2. Add visual query builder interface
3. Implement export filtered logs to JSON/CSV
4. Create filter analytics dashboard
5. Add search history and autocomplete
6. Implement fuzzy matching with Fuse.js (optional)

**Features Added:**
- Complex filter combinations
- Filter preset sharing
- Export functionality
- Analytics dashboard
- Search suggestions

## UI/UX Design

### Layout Strategy

**Header Section:**
- Title
- Live/Pause toggle
- Window size control

**Search Bar:**
- Search input with dropdown for text/regex mode
- Field selection checkboxes below input

**Quick Filters Row:**
- All / Success / Errors
- Active filter count badge
- Duration sliders

**Advanced Filters Sidebar (collapsible):**
- Model selection
- Endpoint selection
- Tools available
- Tools used
- Status codes
- Stop reasons
- User ID
- Temperature range
- Token count range
- Tool use count range
- Date range
- Clear all filters button

**Main Content Area:**
- Filtered log entries
- Windowed display preserved
- Search matches highlighted

**Timeline:**
- Minimap preserved
- Filter status indicators

### Color Scheme

- Active filters: Blue accent (#3b82f6)
- Search matches: Yellow highlight (#fef08a)
- Error entries: Red border (#ef4444)
- Success entries: Green border (#22c55e)
- Disabled buttons: Gray (#9ca3af)

### Interactions

- Search: 500ms debounce
- Filter selection: Instant feedback
- Clear filters: Reset to initial state
- Save preset: Prompt for name
- Load preset: Instant application
- Keyboard shortcuts: Escape to clear search, Ctrl+F to focus search

## Dependencies (pnpm)

### New Packages Required

```json
{
  "react-select": "^5.8.0",
  "react-datepicker": "^6.0.0",
  "lodash.debounce": "^4.0.8",
  "fuse.js": "^7.0.0"  // Optional for fuzzy search
}
```

### Installation Commands

```bash
cd viewer
pnpm add react-select react-datepicker lodash.debounce
pnpm add -D fuse.js  # Optional
```

## Technical Considerations

### Performance

- Frontend filtering scales to 10K+ log entries with proper memoization
- Server-side filtering recommended for 100K+ entries (future enhancement)
- Debounce critical for search input
- Virtual scrolling essential for long dropdown lists

### Browser Support

- Modern browsers (Chrome 90+, Firefox 88+, Safari 14+)
- LocalStorage required for filter persistence
- No external API dependencies for MVP

### Accessibility

- Keyboard navigation for all form elements
- ARIA labels on all interactive elements
- Screen reader support for filter selection
- High contrast mode support

### Error Handling

- Invalid regex patterns caught and displayed
- Missing fields handled gracefully
- Empty filter sets default to "show all"
- Malformed log entries skipped during filtering

## Future Enhancements

### Backend Integration

- Add query parameters to `/api/logs` endpoint
- Implement server-side filtering for large datasets
- Add pagination for filtered results
- Support Elasticsearch/MeiliSearch integration

### Advanced Search

- Full-text search with stemming and synonyms
- Vector similarity search for semantic matching
- Natural language query parser
- Search across multiple log files simultaneously

### Analytics

- Filter usage statistics
- Common filter combinations
- Search query analytics
- Temporal pattern analysis

### Collaboration

- Share filter configurations via URL
- Team-wide filter presets
- Filter configuration versioning
- Export/import filter sets

## Success Criteria

### Functional Requirements

1. Text search works across selected fields
2. Regex search properly validates patterns
3. All identified filters function correctly
4. Filter options populate dynamically
5. Filter state persists across sessions
6. Performance acceptable with 10K entries (<100ms filter time)
7. Search matches highlighted in UI
8. Multi-file support for historical logs

### User Experience

1. Intuitive interface with clear visual hierarchy
2. Responsive feedback on all interactions
3. Empty states handled gracefully
4. Error messages are clear and actionable
5. Keyboard shortcuts available

### Code Quality

1. Componentized architecture
2. Proper state management
3. Comprehensive error handling
4. Reusable helper components
5. Type safety (consider TypeScript migration)

## Risks and Mitigations

### Performance Risk

**Risk:** Large log files (>10K entries) cause slow filtering

**Mitigation:**
- Benchmark with production data
- Implement progressive filtering
- Add loading states
- Consider virtualization early

### Complexity Risk

**Risk:** Too many filter options overwhelm users

**Mitigation:**
- Start with essential filters only
- Collapse advanced filters by default
- Provide helpful presets
- Document filter combinations

### Data Quality Risk

**Risk:** Inconsistent or missing log fields

**Mitigation:**
- Graceful handling of missing fields
- Defensive coding in filter logic
- Filter options exclude invalid values
- Clear error messages for malformed data

## Timeline Estimate

- Total: 14 days
- Phase 1-2: MVP (4 days)
- Phase 3-4: Production-ready (4 days)
- Phase 5-6: Advanced features (6 days)

## Related Documentation

- DevLog-001: Workflow Visualization Planning
- README.md: Viewer documentation
- proxy_server.py: Data structure reference
- parse_tools.py: Tool data extraction reference
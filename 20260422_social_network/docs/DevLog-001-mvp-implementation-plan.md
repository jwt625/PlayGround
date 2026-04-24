# DevLog 001: MVP Implementation Plan

Date: 2026-04-22
Author: Wentao & Codex

## Implementation Status

Last updated: 2026-04-23

Current state:

- Backend core CRUD is live for people, organizations, reminders, and events.
- Relationship scoring now updates from event history and is exposed on person detail views with a simple explanation string.
- Person and organization detail now include structured metadata, linked pipeline items, and relationship context.
- Frontend now has a dense dashboard plus working people, organizations, events, reminders, pipelines, search, imports, and exports screens wired to the API.
- The UI direction has been intentionally minimized: low color, sharp corners, tighter spacing, denser tables, and utility-first layouts.
- Backend list endpoints now emit pagination metadata through response headers.
- A local demo-seed endpoint and smoke-test documentation now exist for faster manual verification.

Highest-priority remaining MVP gaps:

- Remaining gaps are mostly hardening-level rather than missing major surfaces.
- API error payloads are now normalized for validation and HTTP errors.
- Enum-style validation now covers the main event, reminder, and organization categorical fields.
- Search/filter coverage now has a Postgres full-text path with portable fallback behavior for local SQLite tests.

## Summary

DevLog 000 defined the broad vision: a local-first, private relationship management system for people, shared experiences, reminders, and contextual relationship maintenance.

DevLog 001 narrows that vision into an implementation-ready MVP.

The MVP should focus on one daily loop:

- Capture a person.
- Record shared context and interactions.
- Find the person later.
- Understand relationship history.
- Maintain the relationship through events, reminders, and progress tracking.

Investor, partner, and supplier workflows remain important, but they should be modeled as secondary use cases. The core product is not a sales CRM or SCM tool. The core product is a personal relationship system centered on genuine human connection, shared experience, and continuity over time.

## Working Name

Working product name: Kizuna.

Kizuna is a strong fit because it means bond, connection, or tie. It matches the product's human-centered direction better than names that sound like sales tooling.

Practical notes:

- Kizuna is memorable and emotionally aligned with the product.
- The spelling may need occasional explanation, but it is short enough to work.
- Internal package names can use `kizuna` safely.
- If productized later, trademark and domain checks should be done before public launch.

Alternative names considered:

- Kinship
- Common Thread
- Ties
- Continuum
- Relay
- Inner Circle
- Context
- Remembered

Recommendation:

- Use `Kizuna` as the project name for now.
- Use `kizuna` for package, database, and service naming.

## MVP Product Scope

### In Scope

The MVP should include:

- People
- Organizations
- Locations
- Contact methods
- External profiles and source links
- Tags
- Interactions and events
- Reminders
- Relationship scoring
- Generic pipelines
- Search and filtering
- CSV import/export
- Simple user model
- Local Postgres through Docker Compose
- Python FastAPI backend
- SvelteKit frontend

### Out Of Scope

The MVP should not include:

- LLM features
- Google auth
- Multi-user sharing
- Real-time sync
- Gmail sync
- Calendar sync
- LinkedIn import
- WhatsApp import
- WeChat import
- Live geolocation
- Map view
- Graph view
- Browser extension
- Native mobile app
- Full supplier management
- Full investor CRM
- Uploaded binary attachments

The MVP may support source links and local file path references, but not full media upload/download workflows.

## Product Model

### People Are Primary

The primary entity is `Person`.

A person can have many roles:

- Friend
- Investor
- Founder
- Supplier contact
- Partner
- Collaborator
- Customer
- Mentor
- Advisor
- Classmate
- Former colleague

These roles should be metadata on top of the person, not separate identities. The same person may be both a friend and an investor, or both a supplier contact and a personal connection.

### Relationships Are Built Through Events

The application should treat relationships as something built through shared experience.

The most important data is not just a static profile. The most important data is what happened:

- We met at an event.
- We had a long one-on-one dinner.
- They introduced me to someone.
- We worked on a supplier quote together.
- We exchanged useful personal context.
- They helped me with something.
- I promised to follow up.

Therefore, interactions and events should be first-class entities.

### Business Workflows Are Secondary

Pipeline workflows are useful, but they should not dominate the information architecture.

Instead of building separate investor CRM, partner CRM, and supplier SCM systems in the MVP, build one generic pipeline model that can support templates:

- Relationship nurture
- Investor conversation
- Partner collaboration
- Supplier qualification
- Hiring or recruiting
- Community relationship

The default pipeline should be personal relationship-oriented.

## Core MVP Features

### 1. People

A person profile should support fixed MVP fields:

- Display name
- Given name
- Family name
- Nickname
- Pronouns
- Short bio
- Relationship summary
- How we met
- First met date
- Primary location
- Relationship score
- Relationship category
- Last interaction date
- Next reminder date
- Notes
- Created by user
- Created at
- Updated at

Associated records:

- Contact methods
- External profiles
- Locations
- Organizations
- Tags
- Interactions
- Reminders
- Pipeline items

### 2. Contact Methods

Contact methods should be structured records, not a single text blob.

Types:

- Email
- Phone
- WeChat
- WhatsApp
- Telegram
- Signal
- LinkedIn
- X/Twitter
- Website
- Other

Fields:

- Type
- Value
- Label
- Is primary
- Notes

### 3. External Profiles And Source Links

External profiles should point back to existing platforms.

Fields:

- Platform
- URL or handle
- Label
- Notes
- Last checked at

Examples:

- LinkedIn profile
- X/Twitter profile
- Personal website
- Company profile
- GitHub profile
- WeChat ID
- WhatsApp number

### 4. Organizations

Organizations represent companies, funds, schools, communities, suppliers, partners, and other groups.

Fields:

- Name
- Type
- Website
- Description
- Industry
- Location
- Notes

Organization types:

- Company
- Fund
- Supplier
- School
- Community
- Partner
- Nonprofit
- Government
- Other

People can be associated with organizations through roles.

Example relationship fields:

- Person
- Organization
- Title
- Role type
- Start date
- End date
- Is current
- Notes

### 5. Locations

People and organizations should support multiple locations with metadata.

Location fields:

- Label
- City
- Region
- Country
- Address line
- Latitude
- Longitude
- Location type
- Is primary
- Notes

Location types:

- Home
- Work
- Frequent visit
- Supplier site
- Event venue
- School
- Other

MVP behavior:

- Store multiple locations.
- Show one primary location prominently.
- Support filtering by city, region, and country.
- Defer live geolocation and map view.

### 6. Interactions And Events

Interactions and events are core to the product.

For MVP, use one entity called `Event` or `InteractionEvent`. The name should be chosen during implementation, but the concept should include meetings, calls, messages, intros, and relationship milestones.

Event types:

- Meeting
- One-on-one
- Group meeting
- Call
- Email
- Message
- Intro
- Meal
- Event attendance
- Work session
- Supplier discussion
- Personal milestone
- Note
- Other

Fields:

- Title
- Type
- Started at
- Ended at
- Duration minutes
- Context
- Summary
- Notes
- Sentiment
- Location
- Created by user
- Created at
- Updated at

Event associations:

- People involved
- Organizations involved
- Locations involved
- Reminders
- Pipeline items
- Source links
- Local file path references

MVP behavior:

- Events can involve multiple people.
- Events can involve multiple organizations.
- Events can create or update reminders.
- Events should affect relationship score.

### 7. Reminders

Reminders should be attachable to relationship objects.

Attachable entities:

- Person
- Organization
- Event
- Pipeline item

Reminder fields:

- Title
- Notes
- Due at
- Status
- Priority
- Snoozed until
- Completed at
- Created by user

Statuses:

- Open
- Snoozed
- Done
- Canceled

MVP behavior:

- Show overdue reminders.
- Show upcoming reminders.
- Allow creating a reminder from a person profile or event.
- Allow snooze and complete.

### 8. Relationship Score

Relationship strength should be computed from interactions and events, then categorized into a human-readable category.

MVP score inputs:

- Number of interactions.
- Recency of interactions.
- Duration of interactions.
- Interaction type.
- One-on-one versus group setting.
- Manual importance override, if needed.

Heuristic examples:

- A long one-on-one meeting contributes more than a short group event.
- A recent meaningful meeting contributes more than an old brief interaction.
- A personal meal or deep work session contributes more than a quick message.
- Relationship score should decay slowly over time if there are no new interactions.

Suggested categories:

- New
- Dormant
- Light
- Warm
- Strong
- Close

Important caveat:

- The score should be treated as a helpful heuristic, not as a judgment of the relationship.
- The UI should explain why a category was assigned.
- Manual correction should be possible later, but is not required in the first implementation.

### 9. Generic Pipelines

The MVP should include one generic pipeline system.

Pipeline fields:

- Name
- Description
- Template type
- Created by user

Pipeline stage fields:

- Pipeline
- Name
- Sort order
- Color
- Is terminal

Pipeline item fields:

- Pipeline
- Stage
- Title
- Description
- Primary person
- Primary organization
- Status
- Priority
- Expected date
- Notes

Default pipeline template:

- Relationship nurture

Default relationship nurture stages:

- Noted
- Need follow-up
- Conversation started
- Building trust
- Active relationship
- Dormant

Secondary templates:

- Investor conversation
- Partner collaboration
- Supplier qualification

These secondary templates can be created as seed data, but they should not force the app into a business-first CRM shape.

### 10. Search And Filtering

MVP search should support:

- Person name
- Organization name
- Notes
- Event summaries
- Tags
- City
- Industry
- School
- Company

MVP filters:

- Tag
- Location
- Organization
- Industry
- School
- Relationship category
- Last contacted
- Reminder status
- Pipeline
- Pipeline stage

Implementation:

- Use Postgres full-text search where practical.
- Start with useful simple search before advanced ranking.

### 11. CSV Import And Export

MVP import should support:

- Custom CSV template for people.
- Basic contacts import.
- Tags in a delimited column.
- Contact methods in simple columns.
- Organization name in a simple column.

MVP export should support:

- People CSV.
- Organizations CSV.
- Events CSV.
- Reminders CSV.

Defer:

- Google Contacts export compatibility.
- LinkedIn export compatibility.
- Deduplication workflows beyond simple exact matching.

## Technical Architecture

### Backend

Stack:

- Python
- uv
- FastAPI
- SQLAlchemy 2.0
- Alembic
- Pydantic
- Postgres
- pytest

Backend structure:

- `backend/app/main.py`
- `backend/app/api/`
- `backend/app/core/`
- `backend/app/db/`
- `backend/app/models/`
- `backend/app/schemas/`
- `backend/app/services/`
- `backend/app/repositories/`
- `backend/tests/`

API style:

- REST JSON API.
- Resource-oriented routes.
- Simple pagination for list endpoints.
- Filter query parameters for search/list pages.

### Frontend

Stack:

- SvelteKit
- TypeScript
- pnpm

Frontend priorities:

- Desktop-first power-user interface.
- Mobile viewing experience should be good enough for profiles, reminders, and search.
- Data-entry workflows can be desktop-optimized initially.

Core pages:

- Dashboard
- People list
- Person profile
- New/edit person
- Organizations list
- Organization profile
- Events list
- New/edit event
- Reminders dashboard
- Pipeline board
- Import/export page

Deferred pages:

- Map view
- Graph view
- Admin sharing
- Auth settings
- LLM interface

### Local Development

Use Docker Compose for Postgres from the beginning.

Expected services:

- Postgres database
- Backend API
- Frontend dev server

The backend should run through `uv`.

The frontend should run through `pnpm`.

## Initial API Surface

People:

- `GET /people`
- `POST /people`
- `GET /people/{person_id}`
- `PATCH /people/{person_id}`
- `DELETE /people/{person_id}`

Organizations:

- `GET /organizations`
- `POST /organizations`
- `GET /organizations/{organization_id}`
- `PATCH /organizations/{organization_id}`
- `DELETE /organizations/{organization_id}`

Events:

- `GET /events`
- `POST /events`
- `GET /events/{event_id}`
- `PATCH /events/{event_id}`
- `DELETE /events/{event_id}`

Reminders:

- `GET /reminders`
- `POST /reminders`
- `PATCH /reminders/{reminder_id}`
- `POST /reminders/{reminder_id}/snooze`
- `POST /reminders/{reminder_id}/complete`

Pipelines:

- `GET /pipelines`
- `POST /pipelines`
- `GET /pipelines/{pipeline_id}`
- `POST /pipelines/{pipeline_id}/items`
- `PATCH /pipeline-items/{item_id}`
- `POST /pipeline-items/{item_id}/move`

Search:

- `GET /search`

Import/export:

- `POST /imports/people-csv`
- `GET /exports/people-csv`
- `GET /exports/organizations-csv`
- `GET /exports/events-csv`
- `GET /exports/reminders-csv`

## Initial Data Model

Core tables:

- `users`
- `people`
- `organizations`
- `person_organizations`
- `contact_methods`
- `external_profiles`
- `locations`
- `entity_locations`
- `tags`
- `entity_tags`
- `interaction_events`
- `event_people`
- `event_organizations`
- `event_locations`
- `source_links`
- `reminders`
- `pipelines`
- `pipeline_stages`
- `pipeline_items`

Implementation notes:

- Use UUID primary keys.
- Use `created_at` and `updated_at` consistently.
- Include `created_by_user_id` where useful.
- Keep schema fixed for MVP.
- Defer NetBox-style flexible object/custom field system until after MVP.

## Implementation Phases

### Phase 0: Project Scaffold

Deliverables:

- Backend `uv` project.
- Frontend SvelteKit project.
- Docker Compose with Postgres.
- Basic README commands.
- Health check API.

Acceptance criteria:

- `docker compose up` starts Postgres.
- Backend starts successfully.
- Frontend starts successfully.
- Health endpoint returns OK.

### Phase 1: Core Data Model And API

Deliverables:

- SQLAlchemy models.
- Alembic migrations.
- Pydantic schemas.
- CRUD APIs for users, people, organizations, contact methods, external profiles, locations, and tags.

Acceptance criteria:

- Can create, edit, list, and view people.
- Can associate people with organizations.
- Can add contact methods and external profiles.
- Can add multiple locations.

### Phase 2: Events And Relationship Score

Deliverables:

- Interaction event model and APIs.
- Event participants.
- Event organization links.
- Basic relationship score calculation.
- Last interaction date updates.

Acceptance criteria:

- Can log a one-on-one meeting.
- Can log a group event with multiple people.
- Person profile shows event timeline.
- Relationship category updates from event history.

### Phase 3: Reminders

Deliverables:

- Reminder model and APIs.
- Reminder dashboard.
- Person-level and event-level reminder creation.
- Snooze and complete actions.

Acceptance criteria:

- Can create reminders from people and events.
- Dashboard shows overdue and upcoming reminders.
- Snoozed reminders are hidden until due again.

### Phase 4: Frontend Core Experience

Deliverables:

- Dashboard.
- People list with filters.
- Person profile with contact details, locations, tags, timeline, reminders, and organizations.
- Event creation flow.
- Reminder dashboard.

Acceptance criteria:

- User can manage the core relationship loop end to end from the UI.
- Mobile profile viewing is usable.
- Desktop workflow is efficient.

### Phase 5: Generic Pipelines

Deliverables:

- Pipeline models and APIs.
- Seeded relationship nurture pipeline.
- Pipeline board UI.
- Pipeline item links to people and organizations.

Acceptance criteria:

- Can move a relationship through default nurture stages.
- Can create secondary pipeline items for investor, partner, or supplier contexts.
- Pipeline does not dominate the person profile.

### Phase 6: Search And Import/Export

Deliverables:

- Global search.
- Key filters.
- People CSV import.
- CSV exports.

Acceptance criteria:

- Can find people by name, tag, organization, city, school, industry, and notes.
- Can import a small contact spreadsheet.
- Can export core data.

## Implementation Decisions

Resolved decisions:

- Use `backend/` and `frontend/` as top-level application directories.
- Use `InteractionEvent` in code and "Event" in the UI.
- Include source links in the MVP, but defer local file path references until after core CRUD works.
- Use custom Svelte components rather than starting with a UI component library.
- Include soft delete for people, organizations, events, and reminders.
- Use Docker only for local Postgres during MVP development; run the backend and frontend directly with hot reload.

## Detailed Execution TODOs

This section converts the MVP plan into an implementation checklist. The intent is to build the smallest useful vertical slice first, then widen it without changing the product direction.

### Phase 0: Local Scaffold And Run Loop

Goal:

- Make the repository runnable locally with separate backend, frontend, and database services.

TODO:

- [x] Create `docker-compose.yml` with a Postgres service named `postgres`.
- [x] Create backend `uv` project under `backend/`.
- [x] Add FastAPI, SQLAlchemy 2.0, Alembic, Pydantic settings, psycopg, pytest, and developer tooling.
- [x] Create backend package layout: `app/api`, `app/core`, `app/db`, `app/models`, `app/schemas`, `app/services`, and `app/repositories`.
- [x] Add `GET /health` endpoint returning app and database status.
- [x] Create frontend SvelteKit project under `frontend/` using TypeScript and `pnpm`.
- [x] Add README commands for database, backend, frontend, migrations, and tests.
- [x] Verify Postgres starts, backend imports cleanly, and frontend builds or type-checks.

### Phase 1: Identity, People, Organizations, And Metadata

Goal:

- Support manually capturing a person with enough context to be useful.

TODO:

- [x] Add SQLAlchemy base model helpers with UUID primary keys and timestamps.
- [x] Add `users` table and seed or create a default local user.
- [x] Add `people` table with fixed MVP profile fields and soft delete support.
- [x] Add `organizations` table with type, website, industry, location summary, and soft delete support.
- [x] Add `person_organizations` join table with title, role type, dates, current flag, and notes.
- [x] Add `contact_methods` table for structured contact records.
- [x] Add `external_profiles` table for platform links and handles.
- [x] Add `locations` and `entity_locations` tables for reusable person and organization locations.
- [x] Add `tags` and `entity_tags` tables.
- [x] Add Pydantic create, update, list, and detail schemas.
- [x] Add CRUD APIs for people and organizations.
- [x] Add nested or related APIs for contact methods, external profiles, locations, tags, and organization roles.
- [x] Add exact-match duplicate guardrails where simple and low-risk.
- [x] Add initial pytest coverage for people and organizations endpoints.

### Phase 2: Events And Relationship Timeline

Goal:

- Make relationships history-based, not just profile-based.

TODO:

- [x] Add `interaction_events` table with title, type, time range, context, summary, notes, sentiment, and soft delete support.
- [x] Add `event_people`, `event_organizations`, and `event_locations` association tables.
- [x] Add `source_links` table for event and entity references back to external systems.
- [x] Add event CRUD APIs.
- [x] Add person profile timeline query.
- [x] Update person `last_interaction_date` after event changes.
- [x] Implement first-pass relationship score service using interaction count, recency, duration, and event type weights.
- [x] Store or expose relationship category explanation.
- [x] Add tests for one-on-one events, group events, and timeline ordering.

### Phase 3: Reminders And Follow-Up Loop

Goal:

- Show what needs attention and allow quick follow-up management.

TODO:

- [x] Add `reminders` table attachable to person, organization, event, or pipeline item.
- [x] Add reminder status, priority, due date, snooze date, completed date, notes, and soft delete support.
- [x] Add reminder CRUD APIs.
- [x] Add `POST /reminders/{reminder_id}/snooze`.
- [x] Add `POST /reminders/{reminder_id}/complete`.
- [x] Add dashboard query for overdue, due today, and upcoming reminders.
- [x] Update person `next_reminder_date` from open reminders.
- [x] Add tests for open, snoozed, completed, canceled, overdue, and upcoming reminder behavior.

### Phase 4: Frontend Core Relationship Experience

Goal:

- Enable the core daily loop from the UI: capture, review, log, and follow up.

TODO:

- [x] Add frontend API client with typed request helpers.
- [x] Build app shell with dashboard, people, organizations, events, reminders, and pipelines navigation.
- [x] Build dashboard with search entry, recent people, overdue reminders, and upcoming reminders.
- [x] Build people list with search, relationship category, location, tags, and last interaction columns.
- [x] Build person create/edit form optimized for under-one-minute capture.
- [x] Build person profile with overview, contact methods, external profiles, organizations, locations, tags, notes, timeline, and reminders.
- [x] Build organization list and organization profile.
- [x] Build event list and event create/edit flow with multiple participants.
- [x] Build reminder dashboard with complete and snooze actions.
- [x] Make profile and reminder views usable on mobile.

### Phase 5: Generic Pipelines

Goal:

- Support investor, partner, supplier, and personal relationship progress without turning the app into a sales CRM.

TODO:

- [x] Add `pipelines`, `pipeline_stages`, and `pipeline_items` tables.
- [x] Add pipeline CRUD APIs.
- [x] Add pipeline item create, update, and move APIs.
- [x] Seed relationship nurture pipeline stages: Noted, Need follow-up, Conversation started, Building trust, Active relationship, Dormant.
- [x] Optionally seed secondary investor, partner, and supplier templates.
- [x] Build pipeline board UI.
- [x] Link pipeline items to people and organizations.
- [x] Surface linked pipeline items lightly on person and organization profiles.
- [x] Add tests for moving pipeline items and preserving stage order.

### Phase 6: Search, Filters, Import, And Export

Goal:

- Make the data easy to retrieve, segment, move in, and move out.

TODO:

- [x] Add global search endpoint covering people, organizations, event summaries, notes, tags, city, industry, school, and company.
- [x] Add list filters for tag, location, organization, industry, school, relationship category, reminder status, pipeline, and pipeline stage.
- [x] Use Postgres full-text search where practical; keep simple `ilike` fallback behavior for early development.
- [x] Add people CSV import with tags, contact methods, and organization name columns.
- [x] Add exact-match import dedupe by email and display name where safe.
- [x] Add CSV exports for people, organizations, events, and reminders.
- [x] Build import/export frontend page.
- [x] Add tests for CSV import validation and export shape.

### Phase 7: MVP Hardening

Goal:

- Make the first private daily-use version reliable enough to trust.

TODO:

- [x] Add consistent API error responses.
- [x] Add pagination metadata to list endpoints.
- [x] Add backend validation for enum-like fields.
- [x] Add empty states and loading states in the frontend.
- [x] Add seed data for local demo and manual testing.
- [x] Add smoke test documentation.
- [x] Confirm soft-deleted records are excluded from normal list and search results.
- [x] Confirm export keeps user data portable.
- [x] Review docs against actual commands and update any drift.

## Success Criteria

The MVP succeeds if it becomes useful before imports, automation, or AI.

Specifically, it should let the user:

- Add a new person in under one minute.
- Record how they met someone.
- Log a meaningful interaction with multiple participants.
- See the history of a relationship on one page.
- Find people by memory fragments like city, school, company, industry, tag, or note.
- Know who needs follow-up.
- Track light progress without turning the app into a sales CRM.

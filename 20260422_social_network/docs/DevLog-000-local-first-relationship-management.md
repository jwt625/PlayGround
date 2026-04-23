# DevLog 000: Local-First Relationship Management System

Date: 2026-04-22
Author: Wentao & Codex

## Summary

This project is a personal, local-first relationship management system for managing investors, partners, suppliers, collaborators, friends, and other important contacts across fragmented communication channels.

The goal is not to build another public social network. The goal is to build a private relationship intelligence tool: a unified, searchable, context-rich memory system for people, organizations, interactions, and opportunities.

The system should help answer questions like:

- Who do I know in a specific city, industry, school, company, or profession?
- How did I meet this person?
- When did we last speak, and what did we discuss?
- Who introduced us?
- What personal context should I remember before meeting them again?
- Which investors, partners, or suppliers need follow-up?
- Which suppliers are reliable, risky, expensive, slow, or strategically important?

## Product Positioning

Existing platforms each own only part of the relationship graph:

- Email owns conversations.
- Phone contacts own basic contact details.
- LinkedIn owns professional identity.
- X/Twitter owns public thoughts and social graph.
- WhatsApp, WeChat, and SMS own private messaging history.
- Calendar owns meetings.
- CRM tools own sales workflows.
- SCM tools own supplier and operational workflows.

This project should sit above those systems as a private, local-first index and memory layer. It should reference external platforms rather than trying to replace them.

## User And Scope

Primary user:

- One individual managing a growing network of investors, partners, suppliers, collaborators, and other important contacts.

Initial scope:

- Highly customized private tool for personal use.
- Local-first data ownership.
- Rich relationship context and search.
- No LLM features in the MVP.

Future scope:

- Optional browser-based access across multiple devices.
- Optional Google authentication.
- Access-controlled sharing with a small team.
- Possible productization later if the tool proves useful to friends or collaborators.

## Design Principles

### Local First

The application should work with local data by default. Data should remain private and portable.

Expected properties:

- User owns the database.
- App works without cloud dependency.
- Import/export should be easy.
- Multimedia files should be stored locally.
- Future sync should be optional, not foundational.

### Relationship Context Over Vanity Metrics

The product should not optimize for likes, follower counts, or feed consumption. It should optimize for memory, follow-through, trust, and useful context.

### Reference External Platforms

External platforms remain the source of truth for some data. The system should store links and references back to those platforms:

- LinkedIn profile URL
- X/Twitter profile URL
- Email address and thread references
- Phone number
- WeChat ID
- WhatsApp number or export reference
- Calendar event reference
- Website URL
- File attachments
- Imported source metadata

### Structured And Flexible

The product should support both structured fields and free-form notes. Relationships are messy, so the model needs custom metadata, tags, notes, and timelines.

### Human-Centered

The core workflow should feel more like maintaining meaningful relationships than operating a sales machine. CRM and SCM features should be useful, but not overbearing.

## Research-Informed Feature Areas

### CRM-Inspired Features

Useful patterns from typical CRM SaaS:

- Contact management
- Company/account management
- Lead and opportunity pipelines
- Interaction history
- Tasks and reminders
- Follow-up cadences
- Deal stages
- Notes and attachments
- Custom fields
- Segmentation and filtering
- Dashboards and reporting
- Email and calendar integration
- Duplicate detection and contact merging

Adaptation for this project:

- Pipelines should be optional and lightweight.
- Contacts should be more personal and context-rich than standard CRM records.
- The system should support investors, partners, suppliers, collaborators, and friends, not only customers.

### SCM-Inspired Features

Useful patterns from typical SCM SaaS:

- Supplier profiles
- Supplier categorization
- Supplier performance tracking
- Procurement requests
- Quotes and pricing notes
- Lead times
- Order and shipment visibility
- Quality issues
- Certifications and compliance
- Supplier risk
- Contracts and terms
- Exception tracking
- Collaboration history

Adaptation for this project:

- The MVP should not try to become a full ERP or supply chain execution system.
- Supplier relationship management should start as structured profiles, notes, files, follow-ups, and lightweight status tracking.

### Personal CRM-Inspired Features

Useful patterns from personal CRM tools:

- Rich contact cards
- Relationship timelines
- Reminders to stay in touch
- Important dates
- Personal notes
- Family, hobbies, preferences, and life context
- How we met
- Last interaction
- Relationship strength or warmth
- Tags and groups
- Map and location-based views
- Social profile references

Adaptation for this project:

- Personal context should be first-class, not an afterthought.
- Professional and personal details should coexist in the same profile.

## High-Level Feature Proposal

### 1. Unified People Graph

Each person should have a canonical profile that can merge many identifiers:

- Name
- Aliases
- Email addresses
- Phone numbers
- Social profile links
- Messaging handles
- Companies
- Schools
- Locations
- Roles
- Industries
- Tags
- Custom fields
- Source records

The system should support relationships between people:

- Introduced by
- Works with
- Invests with
- Friend of
- Former colleague
- Classmate
- Supplier contact
- Partner contact
- Family relation
- Custom relationship type

### 2. Rich Contact Profiles

Each contact profile should display:

- Basic identity
- Contact methods
- Social links
- Current and past companies
- Education
- Location
- Industry and profession
- Relationship status
- Relationship strength
- How we met
- First meeting date
- Last interaction date
- Preferred communication channel
- Notes
- Personal context
- Important dates
- Related people
- Related organizations
- Related opportunities
- Related supplier records
- Attachments and media

### 3. Relationship Timeline

Each contact should have a chronological timeline of interactions:

- Meetings
- Calls
- Emails
- Messages
- Introductions
- Events
- Meals
- Shared files
- Photos
- Voice notes
- Tasks
- Follow-ups
- Deals
- Supplier discussions
- Personal milestones

Each timeline item should support:

- Date and time
- Interaction type
- Participants
- Summary
- Notes
- Attachments
- Source link
- Tags
- Follow-up task

### 4. Search And Views

The frontend should support multiple ways to explore the network:

- Global search
- Contact list
- Contact cards
- Profile detail page
- Timeline view
- Map view
- Graph view
- Pipeline board
- Supplier directory
- Organization directory
- Tag and filter views
- Recently contacted
- Overdue follow-ups
- Upcoming meetings

Important filters:

- Location
- Industry
- Profession
- School
- Company
- Investor type
- Supplier category
- Relationship strength
- Last contacted date
- Follow-up status
- Tags
- Source platform
- Custom metadata

### 5. Follow-Up And Cadence

The system should make it easy to avoid losing touch.

Potential features:

- Follow-up reminders
- Relationship cadence per contact
- Overdue relationship dashboard
- Snooze
- Mark as contacted
- Next action
- Important dates
- Recurring reminders
- Event-based reminders
- Contextual suggestions

Example cadences:

- Follow up every 2 weeks
- Follow up every month
- Follow up every quarter
- Follow up after next trip to a location
- Follow up after a specific event

### 6. Contextual Trigger Engine

The system should eventually support personal events that trigger reminders or suggestions. This should feel like a private assistant that notices useful relationship opportunities, not like a notification spam machine.

Trigger types:

- Location proximity trigger
- Trip or travel trigger
- Calendar meeting trigger
- Common friend trigger
- Casual invite trigger
- Event attendance trigger
- Time-since-last-contact trigger
- Important date trigger
- Pipeline status trigger
- Supplier status trigger

Location examples:

- "I am visiting New York next week. Show people I know in New York."
- "I landed in Shenzhen. Remind me which suppliers and factory contacts are nearby."
- "I will be in San Francisco for three days. Suggest investors I should casually reconnect with."

Calendar examples:

- "I am meeting Alice tomorrow. Show our last interaction, related people, open follow-ups, and personal context."
- "I am meeting Bob, who is close to Carol. Suggest whether I should ask Bob for an intro or mention Carol."
- "I have dinner with a founder. Show investors, suppliers, or partners connected to that founder."

Casual invite examples:

- "I am hosting coffee in Palo Alto. Suggest 5 nearby people who are overdue for a catch-up."
- "I have a free evening in Shanghai. Suggest close contacts or high-value dormant relationships nearby."
- "I am going to an industry event. Show people in my network who are in that sector or city."

Common friend examples:

- "I am seeing a common friend. Remind me who else they know in my network."
- "This person is connected to an investor I want to meet. Suggest whether this is a good intro path."
- "A close friend is visiting. Suggest mutual contacts we may want to invite."

Supplier and partner examples:

- "I am near a supplier location. Remind me about open samples, quality issues, and next actions."
- "I am meeting a partner who knows several suppliers. Surface relevant supplier questions."
- "A quote has been stale for two weeks. Suggest a follow-up."

Important design choices:

- Suggestions should be explainable, with the trigger reason shown clearly.
- Users should be able to dismiss, snooze, or mark suggestions as useful.
- Trigger rules should start deterministic and configurable before any LLM features are introduced.
- Privacy matters: location and calendar data should be local or explicitly imported by the user.
- The system should avoid surprise behavior. It should not contact anyone automatically.

MVP stance:

- Include the data model foundation for triggers and suggestions.
- Implement simple date-based and manual event-based reminders first.
- Defer live geolocation, automatic calendar sync, and complex graph suggestions until after the core product works.

### 7. Investor And Partner CRM

The system should support lightweight investor and partner workflows.

Possible investor fields:

- Investor type
- Fund or angel
- Check size
- Stage focus
- Sector thesis
- Geography
- Portfolio companies
- Intro path
- Relationship owner
- Last meeting
- Last ask
- Next action
- Materials shared
- Sentiment
- Objections
- Pipeline stage

Example stages:

- Identified
- Intro needed
- Intro requested
- Contacted
- Meeting scheduled
- Met
- Follow-up sent
- Diligence
- Soft commit
- Closed
- Passed
- Nurture

Possible partner fields:

- Partner type
- Strategic value
- Collaboration area
- Decision makers
- Mutual goals
- Open questions
- Next steps
- Status

### 8. Supplier Relationship Management

The system should support lightweight supplier management without trying to become full SCM software.

Possible supplier fields:

- Supplier name
- Supplier category
- Products or services
- Capabilities
- Region
- Key contacts
- Minimum order quantity
- Pricing notes
- Lead time
- Payment terms
- Certifications
- Contracts
- Sample status
- Reliability
- Quality notes
- Risk level
- Alternatives
- Last contacted
- Next action

Possible supplier workflows:

- Discovery
- Initial contact
- Qualification
- Quote requested
- Sample requested
- Sample received
- Negotiating
- Approved
- Active
- Paused
- Rejected

### 9. Import Roadmap

All listed sources are valuable, but imports should be staged to reduce complexity.

Initial import targets:

- CSV
- vCard
- Manual entry

Early import targets:

- Gmail
- Google Calendar
- iPhone contacts export
- LinkedIn export
- WhatsApp export

Later import targets:

- WeChat export or manual archive support
- X/Twitter profile references
- Outlook
- Google Drive file references
- Browser extension capture

Important note:

- Some platforms have restrictive APIs or terms of service. The app should prefer user-provided exports, manual references, and official APIs where available.

### 10. Multimedia And Attachments

Contacts and timeline items should support rich local media:

- Images
- PDFs
- Decks
- Voice notes
- Meeting recordings
- Screenshots
- Business cards
- Contracts
- Quotes
- Invoices
- Chat exports

MVP storage model:

- Store files in a local media directory.
- Store metadata and references in the database.
- Avoid duplicating large files unless explicitly imported.

### 11. Privacy And Access Control

MVP:

- Single-user local application.
- No cloud dependency.
- No LLM.

Future:

- Local user account.
- Google authentication for hosted access.
- Role-based access control.
- Per-contact or per-project sharing.
- Audit log for shared access.
- Optional encryption at rest.
- Optional remote backup.

Potential sharing scopes:

- Private
- Shared with specific user
- Shared with team
- Shared read-only
- Shared editable
- Redacted personal context

## MVP Definition

The MVP should focus on durable foundations and high daily utility.

Included:

- Contact database
- Organization database
- Rich contact profiles
- Social/contact links
- Tags
- Custom fields
- Relationship notes
- Interaction timeline
- File attachments
- Follow-up reminders
- Basic contextual reminder foundation
- Search and filtering
- Investor/partner pipeline
- Supplier profiles
- Local Postgres database
- Local media directory
- CSV import/export
- Basic Svelte frontend
- Python backend API

Excluded from MVP:

- LLM features
- Multi-user collaboration
- Real-time sync
- Browser extension
- Native mobile app
- Deep social network automation
- Full SCM execution
- Full ERP-style inventory/order management

## Proposed Technical Architecture

### Backend

Preferred stack:

- Python
- uv
- FastAPI or Litestar
- Postgres
- SQLAlchemy or SQLModel
- Alembic migrations
- Pydantic models
- Local filesystem media storage

Possible backend capabilities:

- REST API
- Full-text search through Postgres
- Background import jobs
- File upload and metadata extraction
- Export endpoints
- Future auth middleware

### Frontend

Preferred stack:

- SvelteKit
- pnpm
- TypeScript
- Component-based UI
- Local development proxy to backend API

Important frontend experiences:

- Command palette
- Fast global search
- Filterable contact table
- Rich contact profile page
- Timeline editor
- Pipeline board
- Supplier directory
- Dashboard
- Responsive desktop-first layout

### Data Storage

MVP:

- Postgres for structured data.
- Postgres full-text search for text search.
- Local media folder for attachments.

Future:

- Encrypted backup.
- Optional vector index if LLM features are added later.

## Initial Data Model Sketch

Core entities:

- Person
- Organization
- ContactMethod
- ExternalProfile
- Location
- Tag
- CustomField
- Relationship
- Interaction
- Attachment
- Reminder
- TriggerRule
- Suggestion
- PersonalEvent
- Opportunity
- Pipeline
- PipelineStage
- SupplierProfile
- SupplierCapability
- ImportSource

Important relationships:

- Person to Organization
- Person to Person
- Person to Interaction
- Person to Reminder
- PersonalEvent to Suggestion
- TriggerRule to Suggestion
- Person to Opportunity
- Organization to SupplierProfile
- Interaction to Attachment
- Interaction to ExternalProfile or ImportSource

## Open Questions

Questions to resolve before implementation:

- Should the backend be FastAPI or Litestar?
- Should the first UI be desktop-web only, or responsive from day one?
- Should the app run as separate backend/frontend processes, or eventually ship as a local desktop app?
- Should authentication be omitted entirely in MVP local mode, or included early to prepare for hosted access?
- Should custom fields be fully dynamic from the start, or limited to tags and notes initially?
- Should supplier and investor workflows share one generic pipeline system, or have separate purpose-built models?

## Recommended Build Order

1. Create backend and frontend project scaffolds.
2. Define core schema for people, organizations, contact methods, external profiles, tags, interactions, reminders, opportunities, suppliers, and attachments.
3. Build CRUD APIs for core entities.
4. Build frontend shell, dashboard, contact list, contact profile, and timeline.
5. Add search and filtering.
6. Add reminders and follow-up dashboard.
7. Add investor/partner pipeline.
8. Add supplier directory and supplier profile pages.
9. Add CSV import/export.
10. Add media attachments.

## Success Criteria

The first useful version should make it meaningfully easier to:

- Find any important person quickly.
- See all known contact methods and social links in one place.
- Remember how and when a relationship started.
- Review the full context before a meeting.
- Track follow-ups without relying on memory.
- Segment contacts by useful metadata.
- Manage investor, partner, and supplier relationships in one private system.

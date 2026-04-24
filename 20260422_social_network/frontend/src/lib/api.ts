const API_BASE = (import.meta.env.PUBLIC_API_BASE_URL || 'http://localhost:8000/api').replace(/\/$/, '');

export type ContactMethod = {
	id: string;
	type: string;
	value: string;
	label: string | null;
	is_primary: boolean;
	notes: string | null;
};

export type Location = {
	id: string;
	label: string | null;
	city: string | null;
	region: string | null;
	country: string | null;
	address_line: string | null;
	latitude: number | null;
	longitude: number | null;
	location_type: string;
	notes: string | null;
};

export type EntityLocation = {
	id: string;
	location_id: string;
	is_primary: boolean;
	notes: string | null;
	location: Location;
};

export type Tag = {
	id: string;
	name: string;
	color: string | null;
};

export type EntityTag = {
	id: string;
	tag_id: string;
	tag: Tag;
};

export type OrganizationRole = {
	id: string;
	organization_id: string;
	organization_name: string | null;
	title: string | null;
	role_type: string | null;
	start_date: string | null;
	end_date: string | null;
	is_current: boolean;
	notes: string | null;
};

export type ExternalProfile = {
	id: string;
	platform: string;
	url_or_handle: string;
	label: string | null;
	notes: string | null;
	last_checked_at: string | null;
};

export type Person = {
	id: string;
	display_name: string;
	given_name: string | null;
	family_name: string | null;
	nickname: string | null;
	pronouns: string | null;
	short_bio: string | null;
	relationship_summary: string | null;
	how_we_met: string | null;
	first_met_date: string | null;
	primary_location: string | null;
	relationship_score: number;
	relationship_category: string;
	last_interaction_date: string | null;
	next_reminder_date: string | null;
	notes: string | null;
	contact_methods: ContactMethod[];
	external_profiles: ExternalProfile[];
};

export type Event = {
	id: string;
	title: string;
	type: string;
	started_at: string;
	ended_at: string | null;
	duration_minutes: number | null;
	context: string | null;
	summary: string | null;
	notes: string | null;
	sentiment: string | null;
	person_ids: string[];
	organization_ids: string[];
	location_ids: string[];
	deleted_at: string | null;
};

export type PersonDetail = Person & {
	organization_roles: OrganizationRole[];
	locations: EntityLocation[];
	tags: EntityTag[];
	recent_events: Event[];
	active_reminders: Reminder[];
	pipeline_items: PipelineItem[];
	relationship_score_reason: string | null;
};

export type Organization = {
	id: string;
	name: string;
	type: string;
	website: string | null;
	description: string | null;
	industry: string | null;
	location: string | null;
	notes: string | null;
};

export type OrganizationDetail = Organization & {
	locations: EntityLocation[];
	tags: EntityTag[];
	people: OrganizationRole[];
	pipeline_items: PipelineItem[];
};

export type Reminder = {
	id: string;
	title: string;
	notes: string | null;
	due_at: string;
	status: string;
	priority: string;
	snoozed_until: string | null;
	completed_at: string | null;
	entity_type: string | null;
	entity_id: string | null;
};

export type PipelineStage = {
	id: string;
	pipeline_id: string;
	name: string;
	sort_order: number;
	color: string | null;
	is_terminal: boolean;
};

export type PipelineItem = {
	id: string;
	pipeline_id: string;
	stage_id: string;
	title: string;
	description: string | null;
	primary_person_id: string | null;
	primary_organization_id: string | null;
	status: string;
	priority: string;
	expected_date: string | null;
	notes: string | null;
};

export type Pipeline = {
	id: string;
	name: string;
	description: string | null;
	template_type: string;
	stages: PipelineStage[];
};

export type PipelineDetail = Pipeline & {
	items: PipelineItem[];
};

export type SearchResult = {
	entity_type: string;
	id: string;
	title: string;
	subtitle: string | null;
};

export type SearchResponse = {
	people: SearchResult[];
	organizations: SearchResult[];
	events: SearchResult[];
	reminders: SearchResult[];
};

type QueryValue = string | number | undefined | null;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
	const response = await fetch(`${API_BASE}${path}`, {
		headers: {
			'Content-Type': 'application/json',
			...(init?.headers || {})
		},
		...init
	});

	if (!response.ok) {
		const detail = await response.text();
		throw new Error(detail || `Request failed with status ${response.status}`);
	}

	if (response.status === 204) {
		return undefined as T;
	}

	return response.json() as Promise<T>;
}

function withQuery(path: string, params: Record<string, QueryValue>): string {
	const search = new URLSearchParams();

	for (const [key, value] of Object.entries(params)) {
		if (value !== undefined && value !== null && `${value}`.trim() !== '') {
			search.set(key, `${value}`);
		}
	}

	const query = search.toString();
	return query ? `${path}?${query}` : path;
}

export function listPeople(
	params: { q?: string; relationship_category?: string; city?: string; limit?: number } = {}
) {
	return request<Person[]>(withQuery('/people', params));
}

export function getPerson(personId: string) {
	return request<PersonDetail>(`/people/${personId}`);
}

export function createPerson(payload: {
	display_name: string;
	given_name?: string;
	family_name?: string;
	primary_location?: string;
	relationship_summary?: string;
	how_we_met?: string;
	notes?: string;
	contact_methods: Array<{ type: string; value: string; label?: string; is_primary?: boolean }>;
	external_profiles: Array<{ platform: string; url_or_handle: string; label?: string }>;
}) {
	return request<Person>('/people', {
		method: 'POST',
		body: JSON.stringify(payload)
	});
}

export function addPersonTag(personId: string, payload: { name: string; color?: string }) {
	return request<PersonDetail>(`/people/${personId}/tags`, {
		method: 'POST',
		body: JSON.stringify(payload)
	});
}

export function addPersonLocation(
	personId: string,
	payload: { location: Omit<Location, 'id'>; is_primary?: boolean; notes?: string }
) {
	return request<PersonDetail>(`/people/${personId}/locations`, {
		method: 'POST',
		body: JSON.stringify(payload)
	});
}

export function addPersonOrganizationRole(
	personId: string,
	payload: {
		organization_id: string;
		title?: string;
		role_type?: string;
		start_date?: string;
		end_date?: string;
		is_current?: boolean;
		notes?: string;
	}
) {
	return request<PersonDetail>(`/people/${personId}/organization-roles`, {
		method: 'POST',
		body: JSON.stringify(payload)
	});
}

export function listOrganizations(params: { q?: string; industry?: string; limit?: number } = {}) {
	return request<Organization[]>(withQuery('/organizations', params));
}

export function getOrganization(organizationId: string) {
	return request<OrganizationDetail>(`/organizations/${organizationId}`);
}

export function addOrganizationTag(organizationId: string, payload: { name: string; color?: string }) {
	return request<OrganizationDetail>(`/organizations/${organizationId}/tags`, {
		method: 'POST',
		body: JSON.stringify(payload)
	});
}

export function addOrganizationLocation(
	organizationId: string,
	payload: { location: Omit<Location, 'id'>; is_primary?: boolean; notes?: string }
) {
	return request<OrganizationDetail>(`/organizations/${organizationId}/locations`, {
		method: 'POST',
		body: JSON.stringify(payload)
	});
}

export function listEvents(
	params: { q?: string; person_id?: string; organization_id?: string; limit?: number } = {}
) {
	return request<Event[]>(withQuery('/events', params));
}

export function createEvent(payload: {
	title: string;
	type: string;
	started_at: string;
	ended_at?: string;
	duration_minutes?: number;
	context?: string;
	summary?: string;
	notes?: string;
	sentiment?: string;
	person_ids: string[];
	organization_ids?: string[];
	location_ids?: string[];
}) {
	return request<Event>('/events', {
		method: 'POST',
		body: JSON.stringify(payload)
	});
}

export function listReminders(params: { q?: string; status?: string; limit?: number } = {}) {
	return request<Reminder[]>(withQuery('/reminders', params));
}

export function createReminder(payload: {
	title: string;
	due_at: string;
	notes?: string;
	status?: string;
	priority?: string;
	entity_type?: string;
	entity_id?: string;
}) {
	return request<Reminder>('/reminders', {
		method: 'POST',
		body: JSON.stringify(payload)
	});
}

export function updateReminder(reminderId: string, payload: Partial<Reminder>) {
	return request<Reminder>(`/reminders/${reminderId}`, {
		method: 'PATCH',
		body: JSON.stringify(payload)
	});
}

export function snoozeReminder(reminderId: string, snoozedUntil: string) {
	return request<Reminder>(`/reminders/${reminderId}/snooze`, {
		method: 'POST',
		body: JSON.stringify({ snoozed_until: snoozedUntil })
	});
}

export function completeReminder(reminderId: string) {
	return request<Reminder>(`/reminders/${reminderId}/complete`, {
		method: 'POST'
	});
}

export function searchAll(q: string, limit = 10) {
	return request<SearchResponse>(withQuery('/search', { q, limit }));
}

export function listPipelines() {
	return request<Pipeline[]>('/pipelines');
}

export function getPipeline(pipelineId: string) {
	return request<PipelineDetail>(`/pipelines/${pipelineId}`);
}

export function createPipelineItem(
	pipelineId: string,
	payload: {
		title: string;
		stage_id: string;
		description?: string;
		primary_person_id?: string;
		primary_organization_id?: string;
		status?: string;
		priority?: string;
		expected_date?: string;
		notes?: string;
	}
) {
	return request<PipelineItem>(`/pipelines/${pipelineId}/items`, {
		method: 'POST',
		body: JSON.stringify(payload)
	});
}

export function movePipelineItem(itemId: string, stageId: string) {
	return request<PipelineItem>(`/pipeline-items/${itemId}/move`, {
		method: 'POST',
		body: JSON.stringify({ stage_id: stageId })
	});
}

export async function importPeopleCsv(file: File) {
	const formData = new FormData();
	formData.append('file', file);
	const response = await fetch(`${API_BASE}/imports/people-csv`, {
		method: 'POST',
		body: formData
	});
	if (!response.ok) {
		throw new Error((await response.text()) || `Request failed with status ${response.status}`);
	}
	return response.json() as Promise<{ created: number; skipped: number; errors: string[] }>;
}

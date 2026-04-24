from fastapi.testclient import TestClient

from app.main import app


def test_create_and_list_organization() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/organizations",
        json={
            "name": "Kizuna Labs",
            "type": "Company",
            "industry": "Relationship software",
            "location": "San Francisco",
        },
    )

    assert response.status_code == 201
    created = response.json()
    assert created["name"] == "Kizuna Labs"

    list_response = client.get("/api/organizations", params={"q": "Kizuna"})
    assert list_response.status_code == 200
    assert any(item["id"] == created["id"] for item in list_response.json())


def test_create_person_with_contact_methods() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/people",
        json={
            "display_name": "Ada Lovelace",
            "given_name": "Ada",
            "family_name": "Lovelace",
            "primary_location": "London",
            "how_we_met": "Met through a computing history reading group.",
            "contact_methods": [
                {
                    "type": "Email",
                    "value": "ada@example.com",
                    "label": "Personal",
                    "is_primary": True,
                }
            ],
            "external_profiles": [
                {
                    "platform": "Website",
                    "url_or_handle": "https://example.com/ada",
                }
            ],
        },
    )

    assert response.status_code == 201
    created = response.json()
    assert created["display_name"] == "Ada Lovelace"
    assert created["relationship_category"] == "New"
    assert created["contact_methods"][0]["value"] == "ada@example.com"

    detail_response = client.get(f"/api/people/{created['id']}")
    assert detail_response.status_code == 200
    assert detail_response.json()["external_profiles"][0]["platform"] == "Website"


def test_create_and_filter_reminders() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/reminders",
        json={
            "title": "Follow up with Ada",
            "notes": "Send the draft and propose next week.",
            "due_at": "2026-04-24T15:30:00Z",
            "status": "Open",
            "priority": "High",
            "entity_type": "Person",
        },
    )

    assert response.status_code == 201
    created = response.json()
    assert created["title"] == "Follow up with Ada"

    list_response = client.get("/api/reminders", params={"status": "Open", "q": "Ada"})
    assert list_response.status_code == 200
    assert any(item["id"] == created["id"] for item in list_response.json())


def test_event_updates_person_relationship_timeline_and_score() -> None:
    client = TestClient(app)

    person_response = client.post(
        "/api/people",
        json={
            "display_name": "Grace Hopper",
            "primary_location": "New York",
            "contact_methods": [],
            "external_profiles": [],
        },
    )
    assert person_response.status_code == 201
    person_id = person_response.json()["id"]

    event_response = client.post(
        "/api/events",
        json={
            "title": "Coffee with Grace",
            "type": "One-on-one",
            "started_at": "2026-04-20T17:00:00Z",
            "duration_minutes": 90,
            "summary": "Discussed distributed systems and hiring.",
            "person_ids": [person_id],
        },
    )
    assert event_response.status_code == 201

    detail_response = client.get(f"/api/people/{person_id}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["recent_events"][0]["title"] == "Coffee with Grace"
    assert detail["relationship_score"] > 0
    assert detail["relationship_category"] in {"Dormant", "Light", "Warm", "Strong", "Close"}
    assert detail["last_interaction_date"].startswith("2026-04-20T17:00:00")


def test_reminder_snooze_and_complete_actions() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/reminders",
        json={
            "title": "Follow up next month",
            "due_at": "2026-04-24T15:30:00Z",
            "status": "Open",
            "priority": "Normal",
        },
    )
    assert response.status_code == 201
    reminder_id = response.json()["id"]

    snooze_response = client.post(
        f"/api/reminders/{reminder_id}/snooze",
        json={"snoozed_until": "2026-05-24T15:30:00Z"},
    )
    assert snooze_response.status_code == 200
    assert snooze_response.json()["status"] == "Snoozed"

    hidden_list_response = client.get("/api/reminders")
    assert hidden_list_response.status_code == 200
    assert all(item["id"] != reminder_id for item in hidden_list_response.json())

    complete_response = client.post(f"/api/reminders/{reminder_id}/complete")
    assert complete_response.status_code == 200
    assert complete_response.json()["status"] == "Done"


def test_person_next_reminder_date_tracks_open_reminders() -> None:
    client = TestClient(app)

    person_response = client.post(
        "/api/people",
        json={"display_name": "Katherine Johnson", "contact_methods": [], "external_profiles": []},
    )
    assert person_response.status_code == 201
    person_id = person_response.json()["id"]

    create_response = client.post(
        "/api/reminders",
        json={
            "title": "Send follow-up note",
            "due_at": "2026-04-24T09:00:00Z",
            "entity_type": "Person",
            "entity_id": person_id,
        },
    )
    assert create_response.status_code == 201

    detail_response = client.get(f"/api/people/{person_id}")
    assert detail_response.status_code == 200
    assert detail_response.json()["next_reminder_date"].startswith("2026-04-24T09:00:00")


def test_global_search_and_csv_exports() -> None:
    client = TestClient(app)

    person_response = client.post(
        "/api/people",
        json={"display_name": "Radia Perlman", "notes": "Network architecture", "contact_methods": [], "external_profiles": []},
    )
    organization_response = client.post(
        "/api/organizations",
        json={"name": "Bridge Systems", "industry": "Networking"},
    )
    assert person_response.status_code == 201
    assert organization_response.status_code == 201
    person_id = person_response.json()["id"]
    organization_id = organization_response.json()["id"]
    tag_response = client.post(f"/api/people/{person_id}/tags", json={"name": "network"})
    assert tag_response.status_code == 201
    role_response = client.post(
        f"/api/people/{person_id}/organization-roles",
        json={"organization_id": organization_id, "title": "Advisor"},
    )
    assert role_response.status_code == 201

    search_response = client.get("/api/search", params={"q": "Network"})
    assert search_response.status_code == 200
    payload = search_response.json()
    assert any(item["title"] == "Radia Perlman" for item in payload["people"])
    assert any(item["title"] == "Bridge Systems" for item in payload["organizations"])

    tag_search_response = client.get("/api/search", params={"q": "network"})
    assert tag_search_response.status_code == 200
    assert any(item["title"] == "Radia Perlman" for item in tag_search_response.json()["people"])

    export_response = client.get("/api/exports/people-csv")
    assert export_response.status_code == 200
    assert "display_name" in export_response.text
    assert "Radia Perlman" in export_response.text
    assert "Bridge Systems" in export_response.text


def test_person_metadata_and_pipeline_workflows() -> None:
    client = TestClient(app)

    organization_response = client.post("/api/organizations", json={"name": "Bell Labs", "industry": "Research"})
    person_response = client.post(
        "/api/people",
        json={"display_name": "Claude Shannon", "contact_methods": [], "external_profiles": []},
    )
    assert organization_response.status_code == 201
    assert person_response.status_code == 201
    organization_id = organization_response.json()["id"]
    person_id = person_response.json()["id"]

    role_response = client.post(
        f"/api/people/{person_id}/organization-roles",
        json={"organization_id": organization_id, "title": "Researcher", "role_type": "Colleague"},
    )
    assert role_response.status_code == 201
    assert role_response.json()["organization_roles"][0]["organization_name"] == "Bell Labs"

    tag_response = client.post(f"/api/people/{person_id}/tags", json={"name": "legend"})
    assert tag_response.status_code == 201
    assert tag_response.json()["tags"][0]["tag"]["name"] == "legend"

    location_response = client.post(
        f"/api/people/{person_id}/locations",
        json={"location": {"city": "New York", "country": "USA"}, "is_primary": True},
    )
    assert location_response.status_code == 201
    assert location_response.json()["primary_location"] == "New York, USA"

    pipelines_response = client.get("/api/pipelines")
    assert pipelines_response.status_code == 200
    nurture_pipeline = next(item for item in pipelines_response.json() if item["template_type"] == "Relationship nurture")
    stage_id = nurture_pipeline["stages"][0]["id"]

    item_response = client.post(
        f"/api/pipelines/{nurture_pipeline['id']}/items",
        json={"title": "Stay in touch with Claude", "stage_id": stage_id, "primary_person_id": person_id},
    )
    assert item_response.status_code == 201
    item_id = item_response.json()["id"]

    move_response = client.post(
        f"/api/pipeline-items/{item_id}/move",
        json={"stage_id": nurture_pipeline["stages"][1]["id"]},
    )
    assert move_response.status_code == 200
    assert move_response.json()["stage_id"] == nurture_pipeline["stages"][1]["id"]


def test_people_csv_import_with_dedupe() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/imports/people-csv",
        files={
            "file": (
                "people.csv",
                "display_name,email,primary_location\nAlan Turing,alan@example.com,London\nAlan Turing,alan@example.com,London\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["created"] == 1
    assert payload["skipped"] == 1


def test_pagination_headers_demo_seed_and_soft_delete_exclusion() -> None:
    client = TestClient(app)

    seed_response = client.post("/api/demo/seed")
    assert seed_response.status_code == 204

    people_response = client.get("/api/people", params={"limit": 10, "offset": 0})
    assert people_response.status_code == 200
    assert people_response.headers["x-total-count"] == "1"
    person_id = people_response.json()[0]["id"]

    delete_response = client.delete(f"/api/people/{person_id}")
    assert delete_response.status_code == 204

    list_response = client.get("/api/people")
    assert list_response.status_code == 200
    assert list_response.json() == []

    search_response = client.get("/api/search", params={"q": "Ada"})
    assert search_response.status_code == 200
    assert search_response.json()["people"] == []


def test_validation_errors_use_consistent_shape() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/reminders",
        json={"title": "Bad reminder", "due_at": "2026-04-24T09:00:00Z", "status": "InvalidStatus"},
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"] == "validation_error"
    assert isinstance(payload["detail"], list)


def test_nested_contact_profile_apis_and_duplicate_guardrails() -> None:
    client = TestClient(app)

    person_response = client.post(
        "/api/people",
        json={"display_name": "Donald Knuth", "contact_methods": [], "external_profiles": []},
    )
    assert person_response.status_code == 201
    person_id = person_response.json()["id"]

    contact_response = client.post(
        f"/api/people/{person_id}/contact-methods",
        json={"type": "Email", "value": "donald@example.com", "is_primary": True},
    )
    assert contact_response.status_code == 201
    assert any(item["value"] == "donald@example.com" for item in contact_response.json()["contact_methods"])

    profile_response = client.post(
        f"/api/people/{person_id}/external-profiles",
        json={"platform": "Website", "url_or_handle": "https://example.com/donald"},
    )
    assert profile_response.status_code == 201
    assert any(item["platform"] == "Website" for item in profile_response.json()["external_profiles"])

    duplicate_response = client.post(
        "/api/people",
        json={"display_name": "Donald Knuth", "contact_methods": [], "external_profiles": []},
    )
    assert duplicate_response.status_code == 409

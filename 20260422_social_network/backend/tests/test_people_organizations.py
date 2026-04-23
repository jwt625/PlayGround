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

"""
Template service for managing academic website templates.
"""

import logging

from models.github import TemplateInfo, TemplateRepository, TemplateType

logger = logging.getLogger(__name__)


class TemplateService:
    """Service for managing academic website templates."""

    def __init__(self) -> None:
        """Initialize template service with predefined templates."""
        self._templates = self._initialize_templates()

    def _initialize_templates(self) -> dict[TemplateType, TemplateRepository]:
        """Initialize the available templates with real GitHub repositories."""
        return {
            TemplateType.ACADEMIC_PAGES: TemplateRepository(
                id=TemplateType.ACADEMIC_PAGES,
                name="Academic Pages",
                description=(
                    "A comprehensive academic website template with publications, talks, "
                    "CV, and portfolio sections. Perfect for researchers and academics."
                ),
                repository_owner="academicpages",
                repository_name="academicpages.github.io",
                branch="master",  # Academic Pages uses master branch
                features=[
                    "Publications management",
                    "CV/Resume section",
                    "Teaching portfolio",
                    "Blog posts",
                    "Talks and presentations",
                    "Google Analytics",
                    "SEO optimized",
                    "Mobile responsive"
                ],
                preview_url="https://academicpages.github.io/"
            ),

            TemplateType.AL_FOLIO: TemplateRepository(
                id=TemplateType.AL_FOLIO,
                name="Al-folio",
                description=(
                    "A beautiful, simple, clean, and responsive Jekyll theme for "
                    "academics. Features a clean design with support for publications, "
                    "projects, and blog posts."
                ),
                repository_owner="alshedivat",
                repository_name="al-folio",
                branch="master",
                features=[
                    "Clean minimal design",
                    "Publications with BibTeX",
                    "Project showcases",
                    "News and announcements",
                    "Math support (MathJax)",
                    "Code highlighting",
                    "Dark mode",
                    "Multi-language support"
                ],
                preview_url="https://alshedivat.github.io/al-folio/"
            ),

            TemplateType.MINIMAL_ACADEMIC: TemplateRepository(
                id=TemplateType.MINIMAL_ACADEMIC,
                name="Minimal Academic",
                description=(
                    "A lightweight, fast-loading academic template focused on content. "
                    "Ideal for single papers or simple academic pages."
                ),
                repository_owner="pages-themes",
                repository_name="minimal",
                branch="master",
                features=[
                    "Lightweight and fast",
                    "Clean typography",
                    "Math support",
                    "Mobile responsive",
                    "Easy customization",
                    "GitHub Pages compatible"
                ],
                preview_url="https://pages-themes.github.io/minimal/"
            )
        }

    def get_all_templates(self) -> list[TemplateInfo]:
        """Get all available templates as TemplateInfo objects."""
        templates = []

        for template_repo in self._templates.values():
            template_info = TemplateInfo(
                id=template_repo.id.value,
                name=template_repo.name,
                description=template_repo.description,
                preview_url=template_repo.preview_url,
                features=template_repo.features,
                repository_url=f"https://github.com/{template_repo.full_name}",
                repository_owner=template_repo.repository_owner,
                repository_name=template_repo.repository_name
            )
            templates.append(template_info)

        return templates

    def get_template_repository(self, template_id: TemplateType) -> TemplateRepository:
        """Get template repository configuration by ID."""
        if template_id not in self._templates:
            raise ValueError(f"Unknown template: {template_id}")

        return self._templates[template_id]

    def get_template_by_string_id(self, template_id: str) -> TemplateRepository:
        """Get template repository by string ID (for frontend compatibility)."""
        try:
            template_enum = TemplateType(template_id)
            return self.get_template_repository(template_enum)
        except ValueError:
            logger.warning(
                f"Unknown template ID: {template_id}, falling back to minimal-academic"
            )
            return self.get_template_repository(TemplateType.MINIMAL_ACADEMIC)

    def validate_template_id(self, template_id: str) -> bool:
        """Validate if a template ID is supported."""
        try:
            TemplateType(template_id)
            return True
        except ValueError:
            return False


# Global template service instance
template_service = TemplateService()

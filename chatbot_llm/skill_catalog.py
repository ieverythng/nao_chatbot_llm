"""Skill-catalog extraction from exported ROS package manifests."""

from __future__ import annotations

from dataclasses import dataclass

from planner_common import ExportedSkillManifest
from planner_common import load_exported_skill_manifests

try:  # pragma: no cover - optional shared registry dependency
    from skill_common import load_default_registry as load_default_skill_registry
except ImportError:  # pragma: no cover - keep exported-manifest fallback
    load_default_skill_registry = None


@dataclass(frozen=True)
class SkillDescriptor:
    """Compact skill descriptor injected into prompts."""

    package: str
    skill_id: str
    interface_path: str
    datatype: str
    description: str
    input_names: list[str]
    functional_domains: list[str]


def parse_package_list(value: str) -> list[str]:
    """Parse CSV package lists into normalized names."""
    return [token.strip() for token in str(value or '').split(',') if token.strip()]


def build_skill_catalog_text(
    package_names: list[str],
    max_entries: int,
    max_chars: int,
    logger=None,
) -> tuple[str, list[SkillDescriptor]]:
    """Build compact catalog text from allow-listed packages."""
    descriptors = [
        _descriptor_from_manifest(manifest)
        for manifest in load_exported_skill_manifests(package_names, logger=logger)
    ]
    if not descriptors:
        return '', []

    if max_entries > 0:
        descriptors = descriptors[:max_entries]

    lines = ['Available skills:']
    for item in descriptors:
        inputs = ', '.join(item.input_names) if item.input_names else 'none'
        domains = ', '.join(item.functional_domains) if item.functional_domains else 'unspecified'
        description = _shorten(item.description, 180)
        lines.append(
            '- [{package}] {skill} -> {path} ({datatype}) | domains: {domains} | inputs: {inputs} | {desc}'.format(
                package=item.package,
                skill=item.skill_id,
                path=item.interface_path or '<unspecified>',
                datatype=item.datatype or '<unspecified>',
                domains=domains,
                inputs=inputs,
                desc=description or 'no description',
            )
        )

    rendered = '\n'.join(lines)
    if max_chars > 0 and len(rendered) > max_chars:
        rendered = rendered[: max_chars - 3].rstrip() + '...'
    return rendered, descriptors


def build_skill_catalog_text_from_shared_registry(
    max_entries: int,
    max_chars: int,
) -> tuple[str, list[SkillDescriptor]]:
    """Build catalog text from skill_common when available."""
    if load_default_skill_registry is None:
        return '', []
    try:
        registry = load_default_skill_registry()
        skills = list(getattr(registry, 'prompt_manifest', lambda: [])())
    except Exception:
        return '', []

    descriptors: list[SkillDescriptor] = []
    for item in skills:
        if not isinstance(item, dict):
            continue
        descriptors.append(
            SkillDescriptor(
                package='skill_common',
                skill_id=str(item.get('name', '')).strip(),
                interface_path=str(item.get('robot_adapter_mapping', '')).strip(),
                datatype='skill_common/contracts.SkillSpec',
                description=_shorten(
                    ' '.join(str(text).strip() for text in item.get('planner_guidance', []) if str(text).strip()),
                    180,
                ),
                input_names=list(item.get('params', []) or []),
                functional_domains=[str(item.get('category', '')).strip()],
            )
        )
    if not descriptors:
        return '', []

    if max_entries > 0:
        descriptors = descriptors[:max_entries]

    lines = ['Available skills (skill_common):']
    for item in descriptors:
        inputs = ', '.join(item.input_names) if item.input_names else 'none'
        domains = ', '.join(item.functional_domains) if item.functional_domains else 'unspecified'
        description = item.description or 'no description'
        lines.append(
            '- [{package}] {skill} -> {path} ({datatype}) | domains: {domains} | inputs: {inputs} | {desc}'.format(
                package=item.package,
                skill=item.skill_id,
                path=item.interface_path or '<unspecified>',
                datatype=item.datatype,
                domains=domains,
                inputs=inputs,
                desc=description,
            )
        )
    rendered = '\n'.join(lines)
    if max_chars > 0 and len(rendered) > max_chars:
        rendered = rendered[: max_chars - 3].rstrip() + '...'
    return rendered, descriptors


def _descriptor_from_manifest(manifest: ExportedSkillManifest) -> SkillDescriptor:
    return SkillDescriptor(
        package=manifest.package,
        skill_id=manifest.skill_id,
        interface_path=manifest.interface_path,
        datatype=manifest.datatype,
        description=manifest.description,
        input_names=list(manifest.input_names),
        functional_domains=list(manifest.functional_domains),
    )


def _shorten(value: str, limit: int) -> str:
    if limit <= 0:
        return ''
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + '...'

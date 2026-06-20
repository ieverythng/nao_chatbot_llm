"""Skill-catalog extraction from exported ROS package manifests."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from planner_common import ExportedSkillManifest
from planner_common import load_exported_skill_manifests

try:  # pragma: no cover - optional shared registry dependency
    from skill_common import load_default_ab_registry
    from skill_common import load_default_registry as load_default_skill_registry
except ImportError:  # pragma: no cover - keep exported-manifest fallback
    load_default_ab_registry = None
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
    required_params: list[str]
    aliases: list[str]
    functional_domains: list[str]

    def turn_state_entry(self) -> dict:
        """Project registry metadata needed for deterministic IRR admission."""
        return {
            'name': self.skill_id,
            'aliases': list(self.aliases),
            'params': list(self.input_names),
            'required_params': list(self.required_params),
        }


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
    registry_path: str = '',
) -> tuple[str, list[SkillDescriptor]]:
    """Build catalog text from skill_common when available."""
    skills = _shared_skill_payloads(registry_path)

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
                required_params=list(item.get('required_params', []) or []),
                aliases=list(item.get('aliases', []) or []),
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
        required = ', '.join(item.required_params) if item.required_params else 'none'
        domains = ', '.join(item.functional_domains) if item.functional_domains else 'unspecified'
        description = item.description or 'no description'
        lines.append(
            '- [{package}] {skill} -> {path} ({datatype}) | domains: {domains} | inputs: {inputs} | required: {required} | {desc}'.format(
                package=item.package,
                skill=item.skill_id,
                path=item.interface_path or '<unspecified>',
                datatype=item.datatype,
                domains=domains,
                inputs=inputs,
                required=required,
                desc=description,
            )
        )
    rendered = '\n'.join(lines)
    if max_chars > 0 and len(rendered) > max_chars:
        rendered = rendered[: max_chars - 3].rstrip() + '...'
    return rendered, descriptors


def build_turn_state_skill_manifest(registry_path: str = '') -> list[dict]:
    """Return the complete canonical manifest, independent of prompt truncation."""
    return [
        {
            'name': str(item.get('name', '')).strip(),
            'aliases': list(item.get('aliases', []) or []),
            'params': list(item.get('params', []) or []),
            'required_params': list(item.get('required_params', []) or []),
        }
        for item in _shared_skill_payloads(registry_path)
        if str(item.get('name', '')).strip()
    ]


def _shared_skill_payloads(registry_path: str = '') -> list[dict]:
    merged: dict[str, dict] = {}
    if load_default_skill_registry is not None:
        try:
            registry = load_default_skill_registry()
            for item in getattr(registry, 'prompt_manifest', lambda: [])():
                if isinstance(item, dict) and str(item.get('name', '')).strip():
                    merged[str(item['name']).strip()] = dict(item)
        except Exception:
            pass
    if load_default_ab_registry is not None:
        try:
            planner_view = load_default_ab_registry().planner_view()
            for item in planner_view.get('skills', []):
                if not isinstance(item, dict) or not str(item.get('name', '')).strip():
                    continue
                name = str(item['name']).strip()
                merged[name] = {**merged.get(name, {}), **item}
        except Exception:
            pass
    overlay_path = Path(str(registry_path or '').strip()).expanduser()
    if str(registry_path or '').strip() and overlay_path.exists():
        try:
            payload = json.loads(overlay_path.read_text(encoding='utf-8'))
            for item in payload.get('skills', []):
                if not isinstance(item, dict) or not str(item.get('name', '')).strip():
                    continue
                name = str(item['name']).strip()
                merged[name] = {**merged.get(name, {}), **item}
        except (OSError, ValueError, TypeError):
            pass
    return [merged[name] for name in sorted(merged)]


def _descriptor_from_manifest(manifest: ExportedSkillManifest) -> SkillDescriptor:
    return SkillDescriptor(
        package=manifest.package,
        skill_id=manifest.skill_id,
        interface_path=manifest.interface_path,
        datatype=manifest.datatype,
        description=manifest.description,
        input_names=list(manifest.input_names),
        required_params=list(getattr(manifest, 'required_params', []) or []),
        aliases=list(getattr(manifest, 'aliases', []) or []),
        functional_domains=list(manifest.functional_domains),
    )


def _shorten(value: str, limit: int) -> str:
    if limit <= 0:
        return ''
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + '...'

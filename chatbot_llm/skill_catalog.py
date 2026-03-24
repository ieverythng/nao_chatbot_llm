"""Skill-catalog extraction from ROS package manifests."""

from __future__ import annotations

import re
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:  # pragma: no cover - runtime dependency
    get_package_share_directory = None

try:
    import yaml
except ImportError:  # pragma: no cover - runtime dependency
    yaml = None


# ---------------------------------------------------------------------------
# Skill descriptor model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SkillDescriptor:
    """Compact skill descriptor injected into prompts."""

    package: str
    skill_id: str
    interface_path: str
    datatype: str
    description: str
    input_names: list[str]


# ---------------------------------------------------------------------------
# Public catalog assembly
# ---------------------------------------------------------------------------

def parse_package_list(value: str) -> list[str]:
    """Parse CSV package lists into normalized names."""
    tokens = []
    for raw in str(value or '').split(','):
        token = raw.strip()
        if token:
            tokens.append(token)
    return tokens


def build_skill_catalog_text(
    package_names: list[str],
    max_entries: int,
    max_chars: int,
    logger=None,
) -> tuple[str, list[SkillDescriptor]]:
    """Build compact catalog text from allow-listed packages."""
    descriptors: list[SkillDescriptor] = []

    for package_name in package_names:
        package_xml = _resolve_package_xml(package_name)
        if package_xml is None:
            _warn(logger, f'Skill catalog package not found: {package_name}')
            continue
        descriptors.extend(
            _load_package_skill_descriptors(
                package_name=package_name,
                package_xml=package_xml,
                logger=logger,
            )
        )

    if not descriptors:
        return '', []

    if max_entries > 0:
        descriptors = descriptors[:max_entries]

    lines = ['Available skills:']
    for item in descriptors:
        inputs = ', '.join(item.input_names) if item.input_names else 'none'
        description = _shorten(item.description, 180)
        lines.append(
            '- [{package}] {skill} -> {path} ({datatype}) | inputs: {inputs} | {desc}'.format(
                package=item.package,
                skill=item.skill_id,
                path=item.interface_path or '<unspecified>',
                datatype=item.datatype or '<unspecified>',
                inputs=inputs,
                desc=description or 'no description',
            )
        )

    rendered = '\n'.join(lines)
    if max_chars > 0 and len(rendered) > max_chars:
        rendered = rendered[: max_chars - 3].rstrip() + '...'
    return rendered, descriptors


# ---------------------------------------------------------------------------
# Package discovery and manifest parsing
# ---------------------------------------------------------------------------

def _resolve_package_xml(package_name: str) -> Path | None:
    candidates: list[Path] = []

    if get_package_share_directory is not None:
        try:
            share_dir = Path(get_package_share_directory(package_name))
            candidates.append(share_dir / 'package.xml')
        except Exception:
            pass

    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / 'src' / package_name / 'package.xml',
            Path(__file__).resolve().parents[2] / package_name / 'package.xml',
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_package_skill_descriptors(
    package_name: str,
    package_xml: Path,
    logger=None,
) -> list[SkillDescriptor]:
    try:
        root = ET.parse(package_xml).getroot()
    except Exception as err:
        _warn(logger, f'Could not parse {package_xml}: {err}')
        return []

    items = []
    for skill_elem in root.findall('.//skill'):
        content_type = str(skill_elem.attrib.get('content-type', '')).strip().lower()
        if content_type != 'yaml':
            continue

        manifest_text = textwrap.dedent(skill_elem.text or '').strip()
        if not manifest_text:
            continue

        manifest = _parse_skill_manifest_yaml(manifest_text, logger=logger)
        if not manifest:
            continue

        skill_id = str(manifest.get('id', '')).strip()
        if not skill_id:
            continue

        interface_path = str(manifest.get('default_interface_path', '')).strip()
        datatype = str(manifest.get('datatype', '')).strip()
        description = _normalize_spaces(str(manifest.get('description', '')).strip())

        input_names = []
        params = manifest.get('parameters', {})
        if isinstance(params, dict):
            in_params = params.get('in', [])
            if isinstance(in_params, list):
                for entry in in_params:
                    if isinstance(entry, dict):
                        name = str(entry.get('name', '')).strip()
                        if name:
                            input_names.append(name)

        items.append(
            SkillDescriptor(
                package=package_name,
                skill_id=skill_id,
                interface_path=interface_path,
                datatype=datatype,
                description=description,
                input_names=input_names,
            )
        )

    return items


# ---------------------------------------------------------------------------
# YAML fallback parsing helpers
# ---------------------------------------------------------------------------

def _parse_skill_manifest_yaml(payload: str, logger=None) -> dict[str, Any]:
    if yaml is not None:
        try:
            parsed = yaml.safe_load(payload)
            if isinstance(parsed, dict):
                return parsed
        except Exception as err:
            _warn(logger, f'Skill manifest YAML parse failed: {err}')
            return {}

    manifest = {}
    id_match = re.search('^\\s*id:\\s*(.+?)\\s*$', payload, flags=re.MULTILINE)
    if id_match:
        manifest['id'] = id_match.group(1).strip()

    path_match = re.search(
        '^\\s*default_interface_path:\\s*(.+?)\\s*$',
        payload,
        flags=re.MULTILINE,
    )
    if path_match:
        manifest['default_interface_path'] = path_match.group(1).strip()

    datatype_match = re.search(
        '^\\s*datatype:\\s*(.+?)\\s*$',
        payload,
        flags=re.MULTILINE,
    )
    if datatype_match:
        manifest['datatype'] = datatype_match.group(1).strip()

    description_match = re.search(
        '^\\s*description:\\s*\\|\\s*$([\\s\\S]*?)^\\s*[a-zA-Z_]+:\\s*',
        payload,
        flags=re.MULTILINE,
    )
    if description_match:
        manifest['description'] = _normalize_spaces(description_match.group(1).strip())

    names = re.findall('^\\s*-\\s*name:\\s*(.+?)\\s*$', payload, flags=re.MULTILINE)
    if names:
        manifest['parameters'] = {'in': [{'name': x.strip()} for x in names]}

    return manifest


def _normalize_spaces(value: str) -> str:
    return re.sub('\\s+', ' ', value).strip()


def _shorten(value: str, limit: int) -> str:
    if limit <= 0:
        return ''
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + '...'


def _warn(logger, message: str) -> None:
    if logger is not None:
        logger.warn(message)

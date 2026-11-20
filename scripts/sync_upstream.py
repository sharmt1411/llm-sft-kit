"""Sync upstream sanitized skills (e.g. eval-platform) and package a clean distributable plugin bundle."""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def sync_eval_platform_from_file(src_skill: Path, dest_dir: Path) -> None:
    if not src_skill.exists():
        raise FileNotFoundError(f"Source SKILL.md not found: {src_skill}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Copy SKILL.md verbatim to preserve byte-for-byte parity with upstream.
    shutil.copy2(src_skill, dest_dir / "SKILL.md")

    # Sync optional sibling directories (references/, scripts/) if present
    for sub in ("references", "scripts", "examples", "resources"):
        src_sub = src_skill.parent / sub
        dst_sub = dest_dir / sub
        if src_sub.exists() and src_sub.is_dir():
            if dst_sub.exists():
                shutil.rmtree(dst_sub)
            shutil.copytree(src_sub, dst_sub)


def sync_eval_platform_from_git(repo_url: str, subpath: str, branch: str | None, dest_dir: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cmd = ["git", "clone", "--depth", "1"]
        if branch:
            cmd.extend(["--branch", branch])
        cmd.extend([repo_url, tmp])
        subprocess.run(cmd, check=True)
        src_skill = Path(tmp) / subpath
        sync_eval_platform_from_file(src_skill, dest_dir)


def validate_plugin_bundle(root: Path) -> list[str]:
    errors: list[str] = []
    for json_file in ("plugin.json", "mcp_config.json", "hooks.json"):
        p = root / json_file
        if p.exists():
            try:
                json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"Invalid JSON in {p}: {exc}")
        elif json_file == "plugin.json":
            errors.append("Missing required plugin.json")

    skills_dir = root / "skills"
    if skills_dir.exists():
        for skill_folder in sorted(d for d in skills_dir.iterdir() if d.is_dir()):
            skill_md = skill_folder / "SKILL.md"
            if not skill_md.exists():
                errors.append(f"Missing SKILL.md in {skill_folder}")
                continue
            text = skill_md.read_text(encoding="utf-8")
            m = re.match(r"^---\r?\n([\s\S]*?)\r?\n---", text)
            if not m:
                errors.append(f"Missing YAML frontmatter in {skill_md}")
                continue
            fm = m.group(1)
            name_m = re.search(r"^name\s*:\s*(.+)$", fm, flags=re.MULTILINE)
            if not name_m or not name_m.group(1).strip().strip("\"'"):
                errors.append(f"Missing `name` in {skill_md}")
            if not re.search(r"^description\s*:", fm, flags=re.MULTILINE):
                errors.append(f"Missing `description` in {skill_md}")
    return errors


def export_dist(src_root: Path, dist_dir: Path) -> None:
    ignore_names = {".git", "__pycache__", ".pytest_cache", "dist", ".venv"}
    dist_dir.mkdir(parents=True, exist_ok=True)
    expected_rel_paths: set[Path] = set()

    for item in src_root.rglob("*"):
        rel = item.relative_to(src_root)
        if any(part in ignore_names for part in rel.parts) or item.suffix == ".pyc":
            continue
        expected_rel_paths.add(rel)
        target = dist_dir / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)

    # Prune stale files no longer in src_root
    for existing in sorted(dist_dir.rglob("*"), reverse=True):
        rel = existing.relative_to(dist_dir)
        if rel not in expected_rel_paths:
            try:
                if existing.is_dir():
                    existing.rmdir()
                else:
                    existing.unlink()
            except OSError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync upstream skills and package llm-sft-kit.")
    parser.add_argument("--eval-skill-path", type=str, default=None, help="Local path to upstream eval-platform SKILL.md")
    parser.add_argument("--eval-repo-url", type=str, default=None, help="Git URL of upstream eval-platform repository")
    parser.add_argument(
        "--eval-skill-subpath",
        type=str,
        default=".agents/skills/eval-platform/SKILL.md",
        help="Path to SKILL.md inside --eval-repo-url",
    )
    parser.add_argument("--branch", type=str, default=None, help="Optional Git branch or tag")
    parser.add_argument("--dist-dir", type=str, default=None, help="Optional output directory for clean distributable copy")
    args = parser.parse_args()

    eval_dest = PLUGIN_ROOT / "skills" / "eval-platform"
    if args.eval_skill_path:
        sync_eval_platform_from_file(Path(args.eval_skill_path), eval_dest)
        print(f"Synced eval-platform skill from local file: {args.eval_skill_path}")
    elif args.eval_repo_url:
        sync_eval_platform_from_git(args.eval_repo_url, args.eval_skill_subpath, args.branch, eval_dest)
        print(f"Synced eval-platform skill from git: {args.eval_repo_url}")

    errors = validate_plugin_bundle(PLUGIN_ROOT)
    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        return 1

    print("Plugin validation passed.")
    if args.dist_dir:
        dist_path = Path(args.dist_dir)
        export_dist(PLUGIN_ROOT, dist_path)
        print(f"Exported clean distributable plugin to: {dist_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

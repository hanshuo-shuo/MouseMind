from pathlib import Path

from mouse_llm import privacy_guard


def _report(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "mouse_llm/reports/aggregate.json"
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_report_content_guard_rejects_absolute_cluster_paths(tmp_path, monkeypatch):
    path = _report(tmp_path, '{"source":"/shares/private/run"}\n')
    monkeypatch.setattr(
        privacy_guard,
        "tracked_files",
        lambda _root: [str(path.relative_to(tmp_path))],
    )
    assert privacy_guard.blocked_report_contents(tmp_path) == [
        "mouse_llm/reports/aggregate.json: contains '/shares/'"
    ]


def test_report_content_guard_accepts_portable_aggregate(tmp_path, monkeypatch):
    path = _report(tmp_path, '{"run":"frozen-final-language-v1"}\n')
    monkeypatch.setattr(
        privacy_guard,
        "tracked_files",
        lambda _root: [str(path.relative_to(tmp_path))],
    )
    assert privacy_guard.blocked_report_contents(tmp_path) == []

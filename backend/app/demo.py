"""End-to-end demo through the HTTP API, in-process, no server or key needed.

    python -m app.demo [--out ../data/demo_run] [--variant late_braking] [--seed 42]

Generates the synthetic scenario, uploads it through the API, runs the ingestion job,
lists events, asks the diagnostic question when a real provider is configured (skipped
with LLM_PROVIDER=fake), creates the report and prints where it landed.
"""

import argparse
import io
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.app import create_app
from app.core.config import get_settings
from app.rag.index import ChunkIndex, IndexError_, build_index
from app.synthetic.aeb_generator import generate_aeb_scenario
from app.synthetic.io import CSV_NAME, METADATA_NAME, write_scenario
from app.synthetic.schemas import AebScenarioConfig, ScenarioVariant


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aip-demo", description=__doc__)
    p.add_argument("--out", type=Path, default=Path("../data/demo_run"))
    p.add_argument(
        "--variant",
        choices=[v.value for v in ScenarioVariant],
        default=ScenarioVariant.LATE_BRAKING.value,
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--docs", type=Path, default=Path("../data/demo_docs"))
    p.add_argument("--access", default="internal")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors="replace")  # report text may contain non-cp1252 glyphs
    settings = get_settings().model_copy(
        update={
            "database_url": f"sqlite:///{(args.out / 'aip_demo.sqlite').as_posix()}",
            "workspace_dir": args.out / "workspace",
        }
    )
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Synthetic scenario on disk, exactly as an engineer would upload it.
    scenario = generate_aeb_scenario(
        AebScenarioConfig(variant=ScenarioVariant(args.variant), seed=args.seed)
    )
    scenario_dir = args.out / f"aeb_{args.variant}_seed{args.seed}"
    write_scenario(scenario, scenario_dir)
    print(f"[1] scenario written: {scenario_dir}  collision={scenario.ground_truth.collision}")

    # 2. RAG index: reuse the configured one, else build from the demo docs.
    app_index: ChunkIndex | None
    try:
        app_index = ChunkIndex.load(settings.index_dir)
        print(f"[2] index loaded from {settings.index_dir} ({len(app_index)} chunks)")
    except IndexError_:
        from app.llm.factory import build_provider

        app_index = build_index(args.docs, build_provider(settings))
        app_index.save(args.out / "index")
        print(f"[2] index built from {args.docs} ({len(app_index)} chunks)")

    app = create_app(settings, index=app_index)
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        print(f"[3] api up: provider={health['llm_provider']} index={health['rag_index_loaded']}")

        project = client.post("/api/projects", json={"name": "AEB demo"}).json()
        print(f"[4] project {project['id']}")

        with (
            (scenario_dir / CSV_NAME).open("rb") as csv,
            (scenario_dir / METADATA_NAME).open("rb") as sc,
        ):
            files = {
                "telemetry": (CSV_NAME, csv, "text/csv"),
                "sidecar": (METADATA_NAME, sc, "application/json"),
            }
            up = client.post(f"/api/projects/{project['id']}/files", files=files)
        up.raise_for_status()
        f = up.json()
        print(f"[5] uploaded {f['id']}  quality={f['quality_verdict']}  origin={f['data_origin']}")

        job = client.post("/api/ingestion/jobs", json={"file_id": f["id"]}).json()
        print(
            f"[6] ingestion job {job['status']}: events={job['events']} "
            f"metrics={job['metrics_available']} missing={job['metrics_missing']}"
        )
        for e in client.get("/api/events", params={"file_id": f["id"]}).json():
            print(f"     {e['t_s']:6.2f}s  {e['event_type']:<24} {e['id']}")

        run_id: str | None = None
        if health["llm_provider"] == "fake":
            print(
                "[7] query skipped: LLM_PROVIDER=fake has no answers (metrics-only report follows)"
            )
        else:
            q = client.post(
                "/api/query",
                json={"project_id": project["id"], "file_id": f["id"], "access_level": args.access},
            )
            if q.status_code != 200:
                print(f"[7] query failed: {q.status_code} {q.text}")
            else:
                qa = q.json()
                run_id = qa["run_id"]
                print(f"[7] answer ({qa['confidence']}): {qa['answer']}")
                print(f"     evidence: {', '.join(qa['evidence_ids'])}")
                if qa["unsupported_claims"]:
                    print(f"     stripped: {qa['unsupported_claims']}")

        rep = client.post(
            "/api/reports",
            json={
                "project_id": project["id"],
                "file_id": f["id"],
                "run_id": run_id,
                "access_level": args.access,
            },
        )
        rep.raise_for_status()
        r = rep.json()
        print(
            f"[8] report {r['report_id']}  confidence={r['report_confidence']}  "
            f"approval={r['approval_id']}"
        )
        md = client.get(f"/api/reports/{r['report_id']}", params={"format": "md"}).text
        print("     " + "\n     ".join(md.splitlines()[:14]))

        summary = client.get("/api/dashboard/summary").json()
        print(f"[9] dashboard: {summary}")
    print(f"done. workspace: {settings.workspace_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

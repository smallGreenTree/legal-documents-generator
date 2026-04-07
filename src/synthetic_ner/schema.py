"""Case schema and document-id helpers."""

import json
from pathlib import Path


def make_schema_nodes(
    defendants: list,
    collateral: list,
    charged_orgs: list,
    associated_orgs: list,
) -> tuple[list[dict], list[dict]]:
    all_orgs = charged_orgs + associated_orgs
    persons = defendants + collateral

    person_nodes = [
        {
            "id": f"p{index}",
            "name": person["name_plain"],
            "display": person["name"],
            "type": "defendant" if person["is_defendant"] else "collateral",
        }
        for index, person in enumerate(persons)
    ]
    org_nodes = [
        {
            "id": f"o{index}",
            "name": org["name"],
            "type": "charged" if index < len(charged_orgs) else "associated",
        }
        for index, org in enumerate(all_orgs)
    ]
    return person_nodes, org_nodes


def make_case_schema(
    doc_id: str,
    fraud_type: str,
    defendants: list,
    collateral: list,
    charged_orgs: list,
    associated_orgs: list,
) -> dict:
    """Auto-generate a plausible relationship graph from the cast."""
    all_orgs = charged_orgs + associated_orgs
    person_nodes, org_nodes = make_schema_nodes(
        defendants,
        collateral,
        charged_orgs,
        associated_orgs,
    )

    edges = []

    for person_index, person in enumerate(defendants):
        org_index = person_index % len(all_orgs)
        edges.append({
            "from": f"p{person_index}",
            "to": f"o{org_index}",
            "type": "controlled",
            "label": f"{person['name_plain']} controlled {all_orgs[org_index]['name']}",
        })

    if defendants and collateral:
        main_defendant_index = 0
        for collateral_index, collateral_person in enumerate(collateral):
            person_index = len(defendants) + collateral_index
            edges.append({
                "from": f"p{main_defendant_index}",
                "to": f"p{person_index}",
                "type": "instructed",
                "label": (
                    f"{defendants[main_defendant_index]['name_plain']} instructed "
                    f"{collateral_person['name_plain']}"
                ),
            })

    for index in range(len(defendants) - 1):
        edges.append({
            "from": f"p{index}",
            "to": f"p{index + 1}",
            "type": "conspired_with",
            "label": (
                f"{defendants[index]['name_plain']} conspired with "
                f"{defendants[index + 1]['name_plain']}"
            ),
        })

    if charged_orgs and associated_orgs:
        for charged_index, charged_org in enumerate(charged_orgs):
            associated_index = charged_index % len(associated_orgs)
            associated_org = associated_orgs[associated_index]
            edges.append({
                "from": f"o{charged_index}",
                "to": f"o{len(charged_orgs) + associated_index}",
                "type": "received_funds_from",
                "label": f"{associated_org['name']} received funds from {charged_org['name']}",
            })

    return {
        "doc_id": doc_id,
        "fraud_type": fraud_type,
        "persons": person_nodes,
        "orgs": org_nodes,
        "edges": edges,
    }


def schema_to_context(schema: dict) -> str:
    lines = ["Established facts for this case (use these relationships throughout):"]
    for edge in schema["edges"]:
        lines.append(f"  - {edge['label'].rstrip('.')}.")
    return "\n".join(lines)


def write_case_schema(path: Path, schema: dict) -> None:
    path.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")


def load_case_schema(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_schema(
    schema_cfg: dict,
    doc_id: str,
    fraud_type: str,
    defendants: list,
    collateral: list,
    charged_orgs: list,
    associated_orgs: list,
) -> dict:
    if not isinstance(schema_cfg, dict):
        raise ValueError("case.schema must be a mapping")

    derived_persons, derived_orgs = make_schema_nodes(
        defendants,
        collateral,
        charged_orgs,
        associated_orgs,
    )
    edges = schema_cfg.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError("case.schema.edges must be a list")

    return {
        "doc_id": doc_id,
        "fraud_type": fraud_type,
        "persons": schema_cfg.get("persons", derived_persons),
        "orgs": schema_cfg.get("orgs", derived_orgs),
        "edges": edges,
    }


def doc_id_prefix(doc_type: str, fraud_type: str) -> str:
    return f"en_{doc_type}_{fraud_type}_"


def make_doc_id(doc_type: str, fraud_type: str, counter: int) -> str:
    return f"{doc_id_prefix(doc_type, fraud_type)}{counter:03d}"


def counter_from_doc_id(doc_id: str, doc_type: str, fraud_type: str) -> int:
    prefix = doc_id_prefix(doc_type, fraud_type)
    if not isinstance(doc_id, str) or not doc_id.startswith(prefix):
        raise ValueError(f"Schema doc_id must start with '{prefix}', got {doc_id!r}")

    suffix = doc_id[len(prefix):]
    if not suffix.isdigit():
        raise ValueError(f"Schema doc_id must end with digits, got {doc_id!r}")
    return int(suffix)


def next_counter(output_dir: Path, doc_type: str, fraud_type: str) -> int:
    prefix = doc_id_prefix(doc_type, fraud_type)
    numbers = [
        int(directory.name.replace(prefix, ""))
        for directory in output_dir.iterdir()
        if directory.is_dir()
        and directory.name.startswith(prefix)
        and directory.name.replace(prefix, "").isdigit()
    ] if output_dir.exists() else []
    return (max(numbers) + 1) if numbers else 1

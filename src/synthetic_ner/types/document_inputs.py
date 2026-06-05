from dataclasses import dataclass, field


@dataclass
class DocumentInputs:
    defendants: list[dict]
    collateral: list[dict]
    charged_orgs: list[dict]
    associated_orgs: list[dict]
    metadata: dict
    amounts: dict
    counts_list: list[dict]
    evidence_categories: list[str] = field(default_factory=list)

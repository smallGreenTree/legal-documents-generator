from dataclasses import dataclass


@dataclass
class DocumentInputs:
    defendants: list[dict]
    collateral: list[dict]
    charged_orgs: list[dict]
    associated_orgs: list[dict]
    metadata: dict
    counts_list: list[dict]
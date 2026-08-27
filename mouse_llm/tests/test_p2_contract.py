from pathlib import Path

from mouse_llm.evaluation.contracts import contract_seeds, load_p2_contract


def test_p2_seed_sets_are_frozen_and_disjoint():
    contract = load_p2_contract()
    collection = set(contract_seeds("data_collection"))
    development = set(contract_seeds("development"))
    final = set(contract_seeds("final_id_test"))
    assert len(collection) == contract["seed_pools"]["data_collection"]["count"]
    assert not collection & development
    assert not collection & final
    assert not development & final
    assert Path("mouse_llm/evaluation/contracts/p2_eval_v1.json").is_file()

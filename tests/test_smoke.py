from src.config import PROJECT_CONFIG


def test_fixed_project_contract_smoke() -> None:
    assert PROJECT_CONFIG.dataset == "fd001"
    assert PROJECT_CONFIG.random_seed == 42
    assert PROJECT_CONFIG.rul_cap == 125
    assert PROJECT_CONFIG.minimum_history_cycles == 20
    assert PROJECT_CONFIG.feature_schema_version == "v1"

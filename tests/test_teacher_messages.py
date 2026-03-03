from __future__ import annotations

from prompts.teacher_messages import build_teacher_output_contract_block


def test_contract_current_turn_has_no_next_turn_language() -> None:
    contract = build_teacher_output_contract_block(supervision_mode="current_turn")

    lower_contract = contract.lower()
    assert "current turn" in lower_contract
    assert "should have done differently" in lower_contract
    assert "next turn" not in lower_contract


def test_contract_next_turn_keeps_legacy_language() -> None:
    contract = build_teacher_output_contract_block(supervision_mode="next_turn")

    lower_contract = contract.lower()
    assert "next turn" in lower_contract
    assert "current-turn reflection" not in lower_contract

"""Tests for audit rule."""

import datetime

from bank_projections.scenarios.audit import AuditRule
from bank_projections.utils.time import TimeIncrement
from examples.synthetic_data import create_synthetic_balance_sheet


class TestAuditRule:
    """Test AuditRule class."""

    def test_audit_rule_instantiation(self):
        """Test that AuditRule can be instantiated."""
        rule = AuditRule()
        assert rule is not None

    def test_apply_no_audit_in_increment(self, minimal_scenario, minimal_scenario_snapshot):
        """Test applying audit rule when no audit date falls in increment."""
        rule = AuditRule()

        bs = create_synthetic_balance_sheet(datetime.date(2024, 1, 1), scenario=minimal_scenario)
        # January increment - audit month is March (in minimal_scenario)
        increment = TimeIncrement(datetime.date(2024, 1, 1), datetime.date(2024, 1, 31))

        result = rule.apply(bs, increment, minimal_scenario_snapshot)

        assert result is not None

    def test_apply_with_audit_in_increment(self, minimal_scenario, minimal_scenario_snapshot):
        """Test applying audit rule when audit date falls in increment."""
        rule = AuditRule()

        bs = create_synthetic_balance_sheet(datetime.date(2024, 1, 1), scenario=minimal_scenario)
        # March increment - audit month is March (in minimal_scenario)
        increment = TimeIncrement(datetime.date(2024, 3, 1), datetime.date(2024, 3, 31))

        result = rule.apply(bs, increment, minimal_scenario_snapshot)

        assert result is not None

"""Tests for combine utility module."""

import pytest

from bank_projections.utils.combine import Combinable


class CombinableImpl(Combinable):
    """Test implementation of Combinable for testing."""

    def __init__(self, value: int):
        self.value = value

    def combine(self, other: "CombinableImpl") -> "CombinableImpl":
        return CombinableImpl(self.value + other.value)


class NonCombinableClass:
    """Test class that doesn't implement Combinable."""

    def __init__(self, value: int):
        self.value = value


class TestCombinableProtocol:
    """Test Combinable protocol functionality."""

    def test_combinable_implementation(self):
        """Test that a class can implement Combinable."""
        obj1 = CombinableImpl(5)
        obj2 = CombinableImpl(10)

        result = obj1.combine(obj2)

        assert isinstance(result, CombinableImpl)
        assert result.value == 15

    def test_combinable_is_abstract(self):
        """Test that Combinable cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Combinable()

    def test_type_variable_usage(self):
        """Test that the type variable T works correctly."""
        obj1 = CombinableImpl(3)
        obj2 = CombinableImpl(7)

        # The combine method should return the same type
        result = obj1.combine(obj2)
        assert type(result) is CombinableImpl
        assert result.value == 10

    def test_non_combinable_class(self):
        """Test that non-combinable classes don't have combine method."""
        obj = NonCombinableClass(5)

        # Non-combinable class shouldn't have combine method
        assert not hasattr(obj, "combine")

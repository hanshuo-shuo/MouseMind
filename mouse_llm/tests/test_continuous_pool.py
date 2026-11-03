"""Environment ownership tests without simulator construction."""

import pytest

from mouse_llm.continuous.pool import environment_pool


class Resource:
    def __init__(self):
        self.closes = 0

    def close(self):
        self.closes += 1


def test_pool_owns_distinct_instances_and_closes_once_after_use():
    created = []
    def factory():
        env = Resource()
        created.append(env)
        return env
    with environment_pool(factory, 2) as environments:
        assert len(environments) == 2 and environments[0] is not environments[1]
        assert all(env.closes == 0 for env in environments)
    assert [env.closes for env in created] == [1, 1]


def test_pool_closes_successfully_constructed_instances_on_factory_failure():
    first = Resource()
    def factory():
        if not hasattr(factory, "called"):
            factory.called = True
            return first
        raise RuntimeError("second constructor failed")
    with pytest.raises(RuntimeError, match="constructor failed"):
        with environment_pool(factory, 2):
            pytest.fail("Partial construction must not enter the body")
    assert first.closes == 1


def test_pool_rejects_aliasing_without_double_closing_one_resource():
    env = Resource()
    with pytest.raises(ValueError, match="independently"):
        with environment_pool(lambda: env, 2):
            pytest.fail("Same instance must not count as two replay environments")
    assert env.closes == 1


def test_pool_closes_both_instances_when_rollout_raises():
    created = []
    def factory():
        env = Resource()
        created.append(env)
        return env
    with pytest.raises(RuntimeError, match="rollout"):
        with environment_pool(factory, 2):
            raise RuntimeError("rollout failed")
    assert [env.closes for env in created] == [1, 1]

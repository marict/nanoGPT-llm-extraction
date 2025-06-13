import types
import os
import sys
import pytest

runpod = pytest.importorskip("runpod", reason="runpod package required")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import runpod_service as rp


class DummyEndpoint:
    def __init__(self, endpoint_id):
        self.endpoint_id = endpoint_id
        self.called = False

    def run_sync(self, data):
        self.called = True
        return {"output": "ok"}


def test_run_inference(monkeypatch):
    dummy = DummyEndpoint("123")
    monkeypatch.setattr(
        rp, "runpod", types.SimpleNamespace(Endpoint=lambda x: dummy, api_key=None)
    )
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    result = rp.run_inference("hi", endpoint_id="123")
    assert result == "ok"
    assert dummy.called


def test_start_cloud_training(monkeypatch):
    created = {}

    def fake_create_pod(**kwargs):
        created.update(kwargs)
        return {"id": "pod123"}

    statuses = [{"state": "STARTING"}, {"state": "STOPPED"}]

    def fake_get_pod(pod_id):
        return statuses.pop(0)

    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.setattr(
        rp,
        "runpod",
        types.SimpleNamespace(
            create_pod=fake_create_pod,
            get_pod=fake_get_pod,
            Endpoint=None,
            api_key=None,
        ),
    )
    monkeypatch.setattr(rp.time, "sleep", lambda x: None)
    pod_id = rp.start_cloud_training("config.py")
    assert pod_id == "pod123"
    assert "docker_args" in created and "config.py" in created["docker_args"]

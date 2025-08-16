import pytest
from pathlib import Path
import torch
from features.resource_manager import ResourceManager

@pytest.fixture
def resource_manager():
    cache_dir = Path("/tmp/test_cache")
    return ResourceManager(cache_dir)

def test_resource_manager_init(resource_manager):
    assert resource_manager.cache == {}
    assert resource_manager.memory_limit == 0.8

def test_available_memory(resource_manager):
    memory = resource_manager.get_available_memory()
    assert isinstance(memory, float)
    assert memory > 0

def test_cleanup(resource_manager):
    resource_manager.cache["test"] = torch.ones(100, 100)
    assert len(resource_manager.cache) == 1
    resource_manager.cleanup()
    assert len(resource_manager.cache) == 0

def test_predictor_caching(resource_manager):
    model_path = Path("test_model.pth")
    with pytest.raises(FileNotFoundError):
        resource_manager.get_predictor(model_path)
"""
Model Registry API Routes.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import List, Optional
from pydantic import BaseModel

from src.training.model_registry import ModelRegistry, ModelVersion
from src.api.dependencies import get_registry, get_orchestrator
from src.framework.mcts.llm_guided.integration import UnifiedSearchOrchestrator
import shutil
import tempfile
from pathlib import Path

router = APIRouter(prefix="/models", tags=["models"])

class ModelVersionResponse(BaseModel):
    version_id: str
    model_type: str
    filepath: str
    created_at: str
    metrics: dict
    tags: List[str]

@router.post("/register", response_model=ModelVersionResponse)
async def register_model(
    model_type: str = Form(...),
    file: UploadFile = File(...),
    registry: ModelRegistry = Depends(get_registry)
):
    """Uploaded a new model file."""
    # Save upload to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    
    try:
        # Register in registry
        # Minimal metrics/tags for now
        v_id = registry.register_model(
            source_path=tmp_path,
            model_type=model_type,
            metrics={},
            tags=["api-upload"]
        )
        
        # Cleanup
        tmp_path.unlink()
        
        # Get details
        version = registry.repo.get_version(v_id)
        if not version:
             raise HTTPException(status_code=500, detail="Registration failed")
             
        return ModelVersion.from_db(version).to_dict()
        
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{version_id}/promote")
async def promote_model(
    version_id: str,
    registry: ModelRegistry = Depends(get_registry),
    orchestrator: UnifiedSearchOrchestrator = Depends(get_orchestrator)
):
    """Promote a model to 'best' and hot-reload."""
    try:
        registry.promote_to_best(version_id)
        
        # Get model path
        path = registry.get_model_path(version_id)
        version = registry.repo.get_version(version_id)
        
        # Hot reload orchestrator
        if version.model_type == "hrm":
            orchestrator.hot_reload(hrm_path=str(path))
        elif version.model_type == "trm":
            orchestrator.hot_reload(trm_path=str(path))
        elif version.model_type == "meta_controller":
             orchestrator.hot_reload(meta_path=str(path))
             
        return {"status": "promoted", "version_id": version_id, "hot_reloaded": True}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

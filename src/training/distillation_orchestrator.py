"""
Distillation Orchestrator.

Coordinates the distillation process from LLM-generated data to neural models.
Manages:
- Data loading via DistillationDataset
- Model initialization including Text Encoder
- Training loops for Policy-Value, HRM, TRM, and Meta-Controller
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.distillation_dataset import DistillationDataset, DistillationTask
from src.training.system_config import SystemConfig, HRMConfig, TRMConfig
from src.observability.logging import get_structured_logger

# Import Agents being distilled
from src.agents.hrm_agent import HRMAgent, HRMLoss, create_hrm_agent
from src.agents.trm_agent import TRMAgent, TRMLoss, create_trm_agent
from src.agents.common.text_decoder import SystemDecoder
from src.agents.common.system_encoder import SystemEncoder
from src.agents.policy_value_network import PolicyValueNetwork
from src.agents.meta_controller.bert_controller_v2 import BERTMetaController

# Import Encoder
try:
    from transformers import AutoTokenizer, AutoModel
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModel = None

logger = get_structured_logger(__name__)




class DistillationOrchestrator:
    """
    Coordinates distillation of knowledge from LLM data to neural agents.
    """

    def __init__(
        self,
        config: SystemConfig,
        data_dir: str | Path,
        output_dir: str | Path,
    ):
        """
        Initialize the orchestrator.

        Args:
            config: System configuration.
            data_dir: Directory containing distillation source data.
            output_dir: Directory to save distilled models.
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = config.device
        
        # Initialize Shared Encoder
        if _TRANSFORMERS_AVAILABLE:
            self.encoder = SystemEncoder(device=self.device)
            # Freeze encoder by default? Or fine-tune?
            # For phase 2 start, we freeze to save memory/speed
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            self.encoder = None
            logger.warning("Transformers not found, encoder disabled")

        if _TRANSFORMERS_AVAILABLE and self.encoder:
            self.decoder = SystemDecoder(device=self.device, latent_dim=self.encoder.hidden_size)
        elif _TRANSFORMERS_AVAILABLE:
             # Fallback if no encoder (unlikely with _TRANSFORMERS_AVAILABLE check above)
             self.decoder = SystemDecoder(device=self.device)
        else:
            self.decoder = None

        # Models (Lazy init)
        self.hrm_agent: HRMAgent | None = None
        self.trm_agent: TRMAgent | None = None
        self.value_net: PolicyValueNetwork | None = None
        self.meta_controller: BERTMetaController | None = None

    def distill_hrm(
        self,
        batch_size: int = 32,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
    ) -> dict[str, float]:
        """
        Distill HRM (Decomposition) agent.
        """
        logger.info("Starting HRM distillation")
        
        if not self.encoder or not self.decoder:
            logger.error("Encoder/Decoder not available for HRM")
            return {}

        dataset = DistillationDataset(
            data_dir=self.data_dir,
            task_type=DistillationTask.HRM_DECOMPOSITION
        )
        
        if len(dataset) == 0:
             logger.warning("No data found for HRM distillation")
             return {}

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=None)
        
        # Init HRM
        hrm_config = HRMConfig(h_dim=self.encoder.hidden_size) # Match encoder dim
        self.hrm_agent = create_hrm_agent(hrm_config, device=self.device)
        self.hrm_agent.train()
        self.decoder.train()
        
        # Optimize both HRM and Decoder (or just HRM if decoder pretrained? Usually fine-tune both)
        optimizer = torch.optim.AdamW(
            list(self.hrm_agent.parameters()) + list(self.decoder.parameters()), 
            lr=learning_rate
        )
        
        loss_fn = HRMLoss()
        
        # Latent to Text Loss (Decoder handles it internally via CausalLM loss)
        # But we need to define the "sub_task_loss" interface expected by HRMLoss
        # HRMLoss expects (output, pred, target, criterion)
        # We will wrap the decoding loss computation
        
        def decoding_loss_wrapper(latent_state, target_text_list):
            # latent_state: [B, H]
            # target_text: list of strings
            
            # Tokenize targets
            target_tokens = self.decoder.tokenizer(
                target_text_list,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=128
            ).to(self.device)
            
            target_ids = target_tokens.input_ids
            
            # Forward pass decoder
            outputs = self.decoder(
                latent_state=latent_state,
                target_ids=target_ids
            )
            return outputs.loss

        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for batch in dataloader:
                problems = batch["problem"] # list of str
                targets = batch["target_decomposition"] # list of list of str
                
                # Flatten the targets? HRM predicts a sequence of subproblems.
                # Current HRM implementation produces `final_state` [B, S, H] or similar.
                # Let's assume for now we train it to predict the *first* subproblem or *next* step.
                # Or simplify: The 'target' for the planner is a text description of the plan.
                
                # Strategy: Concatenate subproblems into one text string -> "Step 1... Step 2..."
                target_texts = [" ".join(str(t) for t in sublist) for sublist in targets]
                
                # Encode inputs
                with torch.no_grad():
                    input_embeddings = self.encoder(problems) # [B, S, H]
                
                optimizer.zero_grad()
                
                # HRM Forward
                output = self.hrm_agent(input_embeddings)
                
                # We want to match output.final_state (the plan embedding) to the text description
                # output.final_state is [B, 1, H] (global plan state)
                plan_embedding = output.final_state.squeeze(1) # [B, H]
                
                # Compute reconstruction/generation loss
                loss = decoding_loss_wrapper(plan_embedding, target_texts)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            logger.info(f"HRM Epoch {epoch+1}/{num_epochs} Loss: {total_loss/len(dataloader):.4f}")
            
        # Save
        torch.save(self.hrm_agent.state_dict(), self.output_dir / "hrm_agent.pt")
        torch.save(self.decoder.state_dict(), self.output_dir / "system_decoder.pt")
        return {"avg_loss": total_loss/len(dataloader)}

    def distill_trm(
        self,
        batch_size: int = 32,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
    ) -> dict[str, float]:
        """
        Distill TRM (Refinement) agent.
        """
        logger.info("Starting TRM distillation")
        
        if not self.encoder or not self.decoder:
            logger.error("Encoder/Decoder not available for TRM")
            return {}
            
        dataset = DistillationDataset(
            data_dir=self.data_dir,
            task_type=DistillationTask.TRM_REFINEMENT
        )
        
        if len(dataset) == 0:
             logger.warning("No data found for TRM distillation")
             return {}
             
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Init TRM
        trm_config = TRMConfig(latent_dim=self.encoder.hidden_size)
        self.trm_agent = create_trm_agent(trm_config, device=self.device)
        self.trm_agent.train()
        self.decoder.train()
        
        optimizer = torch.optim.AdamW(
            list(self.trm_agent.parameters()) + list(self.decoder.parameters()), 
            lr=learning_rate
        )
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for batch in dataloader:
                initial_codes = batch["initial_code"]
                target_codes = batch["target_code"]
                
                with torch.no_grad():
                    input_emb = self.encoder(initial_codes) # [B, S, H]
                    # Mean pool for simple vector input to TRM
                    input_vec = input_emb.mean(dim=1)
                
                optimizer.zero_grad()
                
                # TRM Forward
                output = self.trm_agent(input_vec)
                
                # Decoder Loss (Generation)
                # target_codes is list of str
                target_tokens = self.decoder.tokenizer(
                    target_codes,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=256 # Longer for code
                ).to(self.device)
                
                # We want the refined state to decode to the target code
                # output.final_prediction is [B, latent_dim]
                
                dec_outputs = self.decoder(
                    latent_state=output.final_prediction,
                    target_ids=target_tokens.input_ids
                )
                
                loss = dec_outputs.loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            logger.info(f"TRM Epoch {epoch+1}/{num_epochs} Loss: {total_loss/len(dataloader):.4f}")
            
        torch.save(self.trm_agent.state_dict(), self.output_dir / "trm_agent.pt")
        # Decoder state dict already saved in HRM but we overwrite/update it here
        torch.save(self.decoder.state_dict(), self.output_dir / "system_decoder.pt")
        return {"avg_loss": total_loss/len(dataloader)}

    def distill_policy_value(
        self, 
        batch_size: int = 32, 
        num_epochs: int = 10, 
        learning_rate: float = 1e-4
    ) -> dict[str, float]:
        """
        Distill Policy-Value Network (Critic).
        """
        logger.info("Starting Policy-Value distillation")
        
        if not self.encoder:
            logger.error("Encoder not available for Policy-Value")
            return {}
            
        dataset = DistillationDataset(
            data_dir=self.data_dir,
            task_type=DistillationTask.POLICY_VALUE
        )
        
        if len(dataset) == 0:
             logger.warning("No data found for Policy-Value distillation")
             return {}
             
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Init Value Net
        self.value_net = PolicyValueNetwork(input_dim=self.encoder.hidden_size).to(self.device)
        self.value_net.train()
        
        optimizer = torch.optim.AdamW(self.value_net.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in dataloader:
                state_code_list = batch["state_code"] # list of str
                target_value = batch["target_value"].float().to(self.device).unsqueeze(1) # [B, 1]
                
                with torch.no_grad():
                    embeddings = self.encoder(state_code_list) # [B, S, H]
                    # Pool
                    state_vec = embeddings.mean(dim=1) # [B, H]
                    
                optimizer.zero_grad()
                pred_value = self.value_net(state_vec)
                loss = loss_fn(pred_value, target_value)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            logger.info(f"PV Epoch {epoch+1}/{num_epochs} Loss: {total_loss/len(dataloader):.4f}")
            
        torch.save(self.value_net.state_dict(), self.output_dir / "value_net.pt")
        return {"avg_loss": total_loss/len(dataloader)}

    def distill_meta_controller(
        self, 
        batch_size: int = 32, 
        num_epochs: int = 10,
        learning_rate: float = 1e-5 # Fine-tuning BERT
    ) -> dict[str, float]:
        """
        Distill Meta-Controller (Router).
        Target: Imitate successful routing (Contextual Bandit-like).
        """
        logger.info("Starting Meta-Controller distillation")
        
        dataset = DistillationDataset(
            data_dir=self.data_dir,
            task_type=DistillationTask.META_CONTROLLER
        )
        
        if len(dataset) == 0:
             logger.warning("No data found for Meta-Controller distillation")
             return {}
             
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Init Meta Controller (BERT-based)
        # Note: BERTMetaController manages its own tokenizer/model internally
        self.meta_controller = BERTMetaController(device=self.device)
        self.meta_controller.model.train() # Set internal model to train
        
        # Optimize internal model parameters
        optimizer = torch.optim.AdamW(self.meta_controller.model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        
        agent_map = {name: i for i, name in enumerate(self.meta_controller.AGENT_NAMES)}
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            valid_batches = 0
            
            for batch in dataloader:
                outcomes = batch["outcome"]
                strategies = batch["agent_strategy"]
                
                # Filter for successful episodes only (Imitation Learning)
                mask = outcomes > 0
                if not mask.any():
                    continue
                    
                valid_batches += 1
                
                # Prepare Inputs
                # Note: BERTMetaController.predict takes MetaControllerFeatures (dataclass)
                # But here we are batch training the underlying model directly.
                # We need to construct text from features batch-wise.
                # Assuming batch["features"] is dict of lists/tensors
                
                features_dict = batch["features"]
                # Convert to text strings
                # This is inefficient to do inside loop, ideally dataset does it or collate_fn
                # For Phase 3 MVP, we do it here.
                
                batch_text = []
                batch_labels = []
                
                from src.agents.meta_controller.utils import features_to_text
                from src.agents.meta_controller.base import MetaControllerFeatures
                
                # Iterate over masked (successful) samples
                indices = torch.nonzero(mask).squeeze(1).tolist()
                for i in indices:
                    strat = strategies[i]
                    if strat not in agent_map:
                         continue
                         
                    f = MetaControllerFeatures(
                        hrm_confidence=features_dict["llm_confidence"][i] if strat == "hrm" else 0.0,
                        trm_confidence=features_dict["llm_confidence"][i] if strat == "trm" else 0.0,
                        mcts_value=features_dict["q_value"][i] if "q_value" in features_dict else 0.0,
                        consensus_score=0.0,
                        last_agent="none",
                        iteration=int(features_dict["depth"][i]),
                        query_length=0,
                        has_rag_context=False
                    )
                    text = features_to_text(f)
                    batch_text.append(text)
                    batch_labels.append(agent_map[strat])
                    
                if not batch_text:
                    continue
                    
                # Tokenize
                inputs = self.meta_controller.tokenizer(
                    batch_text,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=128
                ).to(self.device)
                
                labels = torch.tensor(batch_labels, device=self.device)
                
                optimizer.zero_grad()
                outputs = self.meta_controller.model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / valid_batches if valid_batches > 0 else 0.0
            logger.info(f"MetaController Epoch {epoch+1}/{num_epochs} Loss: {avg_loss:.4f}")
            
        self.meta_controller.save_model(str(self.output_dir / "meta_controller"))
        return {"avg_loss": avg_loss}

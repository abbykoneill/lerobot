"""
VLA (Vision-Language-Action) Educational Module
================================================

This module provides a simplified, educational implementation of a VLA-like system.
Students will learn how vision, language, and robot state combine to produce actions.

The real VLAs like SmolVLA, RT-2, and OpenVLA are much more sophisticated, but this
module captures the core concepts in an accessible way.

Architecture Overview:
----------------------
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   Image     │     │  Language   │     │ Robot State │
    │  Encoder    │     │   Encoder   │     │   Encoder   │
    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
           │                   │                   │
           └───────────────────┼───────────────────┘
                               │
                        ┌──────▼──────┐
                        │   Fusion    │
                        │   Module    │
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │   Action    │
                        │   Decoder   │
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │   Action    │
                        │   Tokens    │
                        └─────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import numpy as np


# =============================================================================
# PART 1: Individual Encoders (Students will implement these)
# =============================================================================

class VisionEncoder(nn.Module):
    """
    Encodes visual input (images) into a feature representation.
    
    In real VLAs, this might be:
    - A pretrained ViT (Vision Transformer)
    - A CLIP visual encoder
    - A CNN backbone like ResNet
    
    For educational purposes, we use a simple CNN.
    """
    
    def __init__(self, input_channels: int = 3, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Simple CNN for educational purposes
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc = nn.Linear(128 * 4 * 4, embedding_dim)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Tensor of shape (batch, channels, height, width)
        
        Returns:
            Visual embedding of shape (batch, embedding_dim)
        """
        # EXERCISE 1: Trace through what happens to the image
        # Print the shape after each layer to understand the transformations
        x = self.conv_layers(image)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class LanguageEncoder(nn.Module):
    """
    Encodes natural language instructions into a feature representation.
    
    In real VLAs, this might be:
    - A pretrained language model (like from SmolLM or Llama)
    - A CLIP text encoder
    - A sentence transformer
    
    For educational purposes, we use a simple embedding + LSTM approach.
    """
    
    def __init__(self, vocab_size: int = 1000, embedding_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Simple word embeddings
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM to process the sequence
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Project back to embedding_dim
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: Tensor of shape (batch, sequence_length) containing token IDs
        
        Returns:
            Language embedding of shape (batch, embedding_dim)
        """
        # EXERCISE 2: Why do we use bidirectional LSTM?
        # What information does reading left-to-right vs right-to-left capture?
        x = self.word_embedding(tokens)
        output, (hidden, cell) = self.lstm(x)
        
        # Concatenate forward and backward final hidden states
        combined = torch.cat([hidden[0], hidden[1]], dim=1)
        x = self.fc(combined)
        return x


class StateEncoder(nn.Module):
    """
    Encodes the robot's proprioceptive state (joint positions, velocities, etc.)
    
    In real robots, state might include:
    - Joint angles (e.g., 6 for a 6-DOF arm)
    - Joint velocities
    - Gripper state (open/closed)
    - End-effector position and orientation
    """
    
    def __init__(self, state_dim: int = 7, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: Tensor of shape (batch, state_dim)
        
        Returns:
            State embedding of shape (batch, embedding_dim)
        """
        return self.encoder(state)


# =============================================================================
# PART 2: Fusion Module (Key concept for students to understand)
# =============================================================================

class MultimodalFusion(nn.Module):
    """
    Fuses vision, language, and state embeddings into a unified representation.
    
    This is a KEY CONCEPT in VLAs:
    - How do we combine information from different modalities?
    - What information from vision helps with the language understanding?
    - How does current robot state affect what action to take?
    
    Common fusion strategies:
    1. Concatenation (simple, but doesn't capture interactions)
    2. Attention (allows modalities to "attend" to each other)
    3. Cross-attention (like in transformers)
    4. Multiplicative fusion
    """
    
    def __init__(self, embedding_dim: int = 256, fusion_dim: int = 512, fusion_type: str = "attention"):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == "concat":
            # Simple concatenation
            self.fusion = nn.Linear(embedding_dim * 3, fusion_dim)
        
        elif fusion_type == "attention":
            # Attention-based fusion
            self.query = nn.Linear(embedding_dim, embedding_dim)
            self.key = nn.Linear(embedding_dim, embedding_dim)
            self.value = nn.Linear(embedding_dim, embedding_dim)
            self.attention_proj = nn.Linear(embedding_dim, fusion_dim)
            self.output_proj = nn.Linear(embedding_dim * 3, fusion_dim)
        
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(
        self, 
        vision_emb: torch.Tensor, 
        language_emb: torch.Tensor, 
        state_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            vision_emb: Visual embedding (batch, embedding_dim)
            language_emb: Language embedding (batch, embedding_dim)
            state_emb: State embedding (batch, embedding_dim)
        
        Returns:
            Fused representation (batch, fusion_dim)
        """
        if self.fusion_type == "concat":
            # EXERCISE 3: This is the simplest fusion - just concatenate
            # What are the limitations of this approach?
            combined = torch.cat([vision_emb, language_emb, state_emb], dim=1)
            return self.fusion(combined)
        
        elif self.fusion_type == "attention":
            # Stack embeddings for attention: (batch, 3, embedding_dim)
            stacked = torch.stack([vision_emb, language_emb, state_emb], dim=1)
            
            # Self-attention across modalities
            Q = self.query(stacked)
            K = self.key(stacked)
            V = self.value(stacked)
            
            # Compute attention weights
            d_k = Q.size(-1)
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            # Apply attention
            attended = torch.matmul(attention_weights, V)
            
            # Flatten and project
            flat = attended.view(attended.size(0), -1)
            return self.output_proj(flat)


# =============================================================================
# PART 3: Action Decoder (Output action tokens)
# =============================================================================

class ActionDecoder(nn.Module):
    """
    Decodes the fused representation into robot actions.
    
    Actions can be represented as:
    1. Continuous values (e.g., joint angles as floats)
    2. Discrete tokens (e.g., discretized into 256 bins per dimension)
    
    Many modern VLAs use ACTION TOKENS - discretized action representations
    that can be generated autoregressively like language tokens.
    
    For a robot arm, typical action dimensions are:
    - 6 for joint positions (6-DOF arm)
    - 1 for gripper (open/close)
    - Total: 7 action dimensions
    """
    
    def __init__(
        self, 
        fusion_dim: int = 512, 
        action_dim: int = 7,
        num_action_bins: int = 256,
        use_action_tokens: bool = True
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_action_bins = num_action_bins
        self.use_action_tokens = use_action_tokens
        
        if use_action_tokens:
            # Discrete action tokens (like in RT-2)
            # Each action dimension gets its own discrete bin
            self.action_heads = nn.ModuleList([
                nn.Linear(fusion_dim, num_action_bins) for _ in range(action_dim)
            ])
        else:
            # Continuous action output
            self.action_predictor = nn.Sequential(
                nn.Linear(fusion_dim, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
                nn.Tanh()  # Actions typically normalized to [-1, 1]
            )
    
    def forward(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused: Fused representation (batch, fusion_dim)
        
        Returns:
            Dictionary containing:
            - 'actions': Continuous action values (batch, action_dim)
            - 'action_logits': If using tokens, logits for each dimension
            - 'action_tokens': If using tokens, the selected token indices
        """
        output = {}
        
        if self.use_action_tokens:
            # EXERCISE 4: Why might we want discrete action tokens instead of continuous?
            # Hint: Think about language models and how they generate text...
            
            action_logits = []
            action_tokens = []
            
            for head in self.action_heads:
                logits = head(fused)  # (batch, num_bins)
                action_logits.append(logits)
                tokens = torch.argmax(logits, dim=-1)  # (batch,)
                action_tokens.append(tokens)
            
            # Stack: (batch, action_dim, num_bins)
            output['action_logits'] = torch.stack(action_logits, dim=1)
            # Stack: (batch, action_dim)
            output['action_tokens'] = torch.stack(action_tokens, dim=1)
            
            # Convert tokens back to continuous values in [-1, 1]
            # Token 0 -> -1, Token 127 -> 0, Token 255 -> 1
            continuous = (output['action_tokens'].float() / (self.num_action_bins - 1)) * 2 - 1
            output['actions'] = continuous
        
        else:
            output['actions'] = self.action_predictor(fused)
        
        return output


# =============================================================================
# PART 4: Complete VLA Model
# =============================================================================

class SimpleVLA(nn.Module):
    """
    A simplified Vision-Language-Action model for educational purposes.
    
    This demonstrates the core VLA concept:
        (Image, Instruction, State) -> Action
    
    Real VLAs like SmolVLA, RT-2, and OpenVLA use:
    - Much larger pretrained vision encoders (ViT, SigLIP)
    - Large language models (Llama, PaLI)
    - More sophisticated fusion (cross-attention, transformer layers)
    - Trained on millions of robot demonstrations
    """
    
    def __init__(
        self,
        image_channels: int = 3,
        vocab_size: int = 1000,
        state_dim: int = 7,
        action_dim: int = 7,
        embedding_dim: int = 256,
        fusion_dim: int = 512,
        num_action_bins: int = 256,
        fusion_type: str = "attention",
        use_action_tokens: bool = True
    ):
        super().__init__()
        
        # Encoders
        self.vision_encoder = VisionEncoder(image_channels, embedding_dim)
        self.language_encoder = LanguageEncoder(vocab_size, embedding_dim, embedding_dim)
        self.state_encoder = StateEncoder(state_dim, embedding_dim)
        
        # Fusion
        self.fusion = MultimodalFusion(embedding_dim, fusion_dim, fusion_type)
        
        # Action decoder
        self.action_decoder = ActionDecoder(
            fusion_dim, action_dim, num_action_bins, use_action_tokens
        )
        
        # Store config
        self.config = {
            'image_channels': image_channels,
            'vocab_size': vocab_size,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'embedding_dim': embedding_dim,
            'fusion_dim': fusion_dim,
            'num_action_bins': num_action_bins,
            'fusion_type': fusion_type,
            'use_action_tokens': use_action_tokens
        }
    
    def forward(
        self,
        image: torch.Tensor,
        language_tokens: torch.Tensor,
        state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass of the VLA.
        
        Args:
            image: (batch, channels, height, width)
            language_tokens: (batch, sequence_length)
            state: (batch, state_dim)
        
        Returns:
            Dictionary with action outputs
        """
        # Encode each modality
        vision_emb = self.vision_encoder(image)
        language_emb = self.language_encoder(language_tokens)
        state_emb = self.state_encoder(state)
        
        # Fuse modalities
        fused = self.fusion(vision_emb, language_emb, state_emb)
        
        # Decode to actions
        action_output = self.action_decoder(fused)
        
        # Also return intermediate representations for analysis
        action_output['vision_embedding'] = vision_emb
        action_output['language_embedding'] = language_emb
        action_output['state_embedding'] = state_emb
        action_output['fused_embedding'] = fused
        
        return action_output
    
    def predict_action(
        self,
        image: torch.Tensor,
        language_tokens: torch.Tensor,
        state: torch.Tensor
    ) -> torch.Tensor:
        """
        Simplified inference method that returns just the action.
        
        Args:
            image: (batch, channels, height, width)
            language_tokens: (batch, sequence_length)
            state: (batch, state_dim)
        
        Returns:
            actions: (batch, action_dim)
        """
        with torch.no_grad():
            output = self.forward(image, language_tokens, state)
        return output['actions']


# =============================================================================
# PART 5: Simple Tokenizer for Educational Purposes
# =============================================================================

class SimpleTokenizer:
    """
    A very simple tokenizer for educational purposes.
    Real VLAs use sophisticated tokenizers from language models.
    """
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.word_to_id = {'<PAD>': 0, '<UNK>': 1}
        self.id_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.next_id = 2
    
    def _add_word(self, word: str) -> int:
        """Add a word to vocabulary if not present."""
        word = word.lower()
        if word not in self.word_to_id and self.next_id < self.vocab_size:
            self.word_to_id[word] = self.next_id
            self.id_to_word[self.next_id] = word
            self.next_id += 1
        return self.word_to_id.get(word, 1)  # Return <UNK> if not in vocab
    
    def encode(self, text: str, max_length: int = 32) -> torch.Tensor:
        """
        Convert text to token IDs.
        
        Args:
            text: Input string
            max_length: Maximum sequence length
        
        Returns:
            Token tensor of shape (max_length,)
        """
        words = text.lower().split()
        tokens = [self._add_word(word) for word in words]
        
        # Pad or truncate
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens = tokens + [0] * (max_length - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode(self, tokens: torch.Tensor) -> str:
        """Convert token IDs back to text."""
        words = []
        for token_id in tokens.tolist():
            if token_id == 0:  # PAD
                break
            words.append(self.id_to_word.get(token_id, '<UNK>'))
        return ' '.join(words)


# =============================================================================
# PART 6: Demo and Exercises
# =============================================================================

def create_demo_inputs(batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """
    Create synthetic demo inputs for testing.
    
    Returns:
        image: Random image tensor
        tokens: Tokenized instruction
        state: Random robot state
        instruction: Original instruction text
    """
    # Random image (batch, 3, 224, 224)
    image = torch.randn(batch_size, 3, 224, 224)
    
    # Tokenize an instruction
    tokenizer = SimpleTokenizer()
    instruction = "pick up the red block"
    tokens = tokenizer.encode(instruction).unsqueeze(0).expand(batch_size, -1)
    
    # Random robot state (batch, 7) - typical for 6-DOF arm + gripper
    state = torch.randn(batch_size, 7)
    
    return image, tokens, state, instruction


def run_demo():
    """
    Run a demonstration of the SimpleVLA model.
    """
    print("=" * 70)
    print("SIMPLE VLA DEMONSTRATION")
    print("=" * 70)
    
    # Create model
    model = SimpleVLA(
        image_channels=3,
        vocab_size=1000,
        state_dim=7,
        action_dim=7,
        embedding_dim=256,
        fusion_dim=512,
        use_action_tokens=True
    )
    model.eval()
    
    # Create demo inputs
    image, tokens, state, instruction = create_demo_inputs()
    
    print(f"\nInput Shapes:")
    print(f"  Image: {image.shape}")
    print(f"  Language tokens: {tokens.shape}")
    print(f"  Robot state: {state.shape}")
    print(f"\nInstruction: \"{instruction}\"")
    
    # Forward pass
    output = model(image, tokens, state)
    
    print(f"\nOutput Shapes:")
    print(f"  Vision embedding: {output['vision_embedding'].shape}")
    print(f"  Language embedding: {output['language_embedding'].shape}")
    print(f"  State embedding: {output['state_embedding'].shape}")
    print(f"  Fused embedding: {output['fused_embedding'].shape}")
    print(f"  Actions: {output['actions'].shape}")
    
    if 'action_tokens' in output:
        print(f"  Action tokens: {output['action_tokens'].shape}")
    
    print(f"\nPredicted Actions:")
    actions = output['actions'][0].detach().numpy()
    action_names = [
        "joint_1", "joint_2", "joint_3", 
        "joint_4", "joint_5", "joint_6", "gripper"
    ]
    for name, value in zip(action_names, actions):
        print(f"  {name}: {value:.4f}")
    
    print("\n" + "=" * 70)
    print("Demo complete! See the exercises in the notebook.")
    print("=" * 70)
    
    return model, output


# =============================================================================
# EXERCISE TEMPLATES (Students will implement these)
# =============================================================================

def exercise_1_trace_dimensions():
    """
    EXERCISE 1: Trace Dimensions Through the Network
    
    Goal: Understand how data flows through the VLA by printing shapes
    at each stage.
    
    TODO: Add print statements in VisionEncoder.forward() to trace the
    image dimensions through each convolutional layer.
    """
    # Your code here
    pass


def exercise_2_modify_fusion():
    """
    EXERCISE 2: Implement a New Fusion Strategy
    
    Goal: Implement multiplicative fusion and compare results.
    
    Multiplicative fusion: element-wise product of embeddings
    This forces the model to learn features that are useful for
    ALL modalities simultaneously.
    
    TODO: Add a new fusion_type="multiplicative" option to MultimodalFusion
    """
    # Your code here
    pass


def exercise_3_visualize_attention():
    """
    EXERCISE 3: Visualize Attention Weights
    
    Goal: Extract and visualize the attention weights in the fusion module.
    
    Questions to answer:
    - Does the model attend more to vision or language?
    - How does attention change with different instructions?
    """
    # Your code here
    pass


def exercise_4_action_space():
    """
    EXERCISE 4: Experiment with Action Representation
    
    Goal: Compare discrete tokens vs continuous actions.
    
    Questions to answer:
    - What are the tradeoffs between discrete and continuous?
    - How does the number of bins affect precision?
    - Why might language models prefer discrete tokens?
    """
    # Your code here
    pass


def exercise_5_instruction_sensitivity():
    """
    EXERCISE 5: Test Instruction Sensitivity
    
    Goal: See how the model's output changes with different instructions.
    
    Test with:
    - "pick up the red block"
    - "pick up the blue block"  
    - "put down the object"
    - "move left"
    
    Does the untrained model show any meaningful differences?
    (It shouldn't - but a trained model would!)
    """
    # Your code here
    pass


if __name__ == "__main__":
    run_demo()


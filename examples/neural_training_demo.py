"""
Neural Network Training Demo.

Demonstrates training policy and value networks on synthetic data,
including data collection, training, and evaluation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.models.policy_network import create_policy_network
from src.models.value_network import create_value_network
from src.training.data_collector import Experience, ExperienceBuffer
from src.training.neural_trainer import (
    PolicyDataset,
    TrainingConfig,
    ValueDataset,
    train_policy_network,
    train_value_network,
)


class TicTacToeSimulator:
    """Simple Tic-Tac-Toe simulator for demo."""

    def __init__(self):
        self.state_dim = 9  # 3x3 board
        self.action_dim = 9  # 9 possible moves

    def encode_state(self, board):
        """Encode board as tensor."""
        # board: list of 9 elements (-1, 0, 1) for (O, empty, X)
        return torch.tensor(board, dtype=torch.float32)

    def generate_random_game(self):
        """Generate random game for demonstration."""
        board = [0] * 9
        experiences = []
        player = 1  # X starts

        for move in range(9):
            # Available moves
            available = [i for i, cell in enumerate(board) if cell == 0]
            if not available:
                break

            # Random action
            action = available[torch.randint(len(available), (1,)).item()]

            # Record experience
            state = self.encode_state(board)
            experiences.append(Experience(state=state, action=action, value=0.0))

            # Apply action
            board[action] = player

            # Check win
            if self.check_win(board, player):
                outcome = player
                break

            # Switch player
            player = -player
        else:
            outcome = 0  # Draw

        # Update values with outcome
        for exp in experiences:
            exp.value = float(outcome)

        return experiences

    def check_win(self, board, player):
        """Check if player has won."""
        # Check rows
        for i in range(0, 9, 3):
            if all(board[i + j] == player for j in range(3)):
                return True

        # Check columns
        for i in range(3):
            if all(board[i + j * 3] == player for j in range(3)):
                return True

        # Check diagonals
        if all(board[i] == player for i in [0, 4, 8]):
            return True
        if all(board[i] == player for i in [2, 4, 6]):
            return True

        return False


def generate_training_data(num_games=1000):
    """Generate training data from random games."""
    print(f"Generating {num_games} games...")

    simulator = TicTacToeSimulator()
    buffer = ExperienceBuffer(max_size=num_games * 9)

    for i in range(num_games):
        experiences = simulator.generate_random_game()
        buffer.add_batch(experiences)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_games} games")

    print(f"Collected {len(buffer)} experiences")
    return buffer


def train_policy_demo():
    """Demonstrate policy network training."""
    print("\n" + "=" * 60)
    print("POLICY NETWORK TRAINING DEMO")
    print("=" * 60)

    # Generate training data
    buffer = generate_training_data(num_games=1000)

    # Create datasets
    experiences = buffer.get_all()
    states = torch.stack([exp.state for exp in experiences])
    actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long)
    values = torch.tensor([exp.value for exp in experiences], dtype=torch.float32)

    # Split train/val
    train_size = int(0.8 * len(states))
    train_dataset = PolicyDataset(states[:train_size], actions[:train_size], values[:train_size])
    val_dataset = PolicyDataset(states[train_size:], actions[train_size:], values[train_size:])

    print("\nDataset sizes:")
    print(f"  Training: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")

    # Create policy network
    policy_net = create_policy_network(state_dim=9, action_dim=9, config={"hidden_dims": [128, 64]})

    print("\nPolicy Network:")
    print(f"  Parameters: {policy_net.get_parameter_count():,}")

    # Training config
    config = TrainingConfig(
        learning_rate=0.001,
        batch_size=32,
        num_epochs=50,
        checkpoint_dir="checkpoints/policy",
        save_every=10,
        early_stopping_patience=10,
    )

    # Train
    print("\nTraining policy network...")
    trainer = train_policy_network(policy_net, train_dataset, val_dataset, config)

    # Plot training curves
    print("\nPlotting training curves...")
    plot_training_curves(trainer.training_history, "Policy Network Training", "policy_training.png")

    print("\nPolicy network training complete!")
    return policy_net, trainer


def train_value_demo():
    """Demonstrate value network training."""
    print("\n" + "=" * 60)
    print("VALUE NETWORK TRAINING DEMO")
    print("=" * 60)

    # Generate training data
    buffer = generate_training_data(num_games=1000)

    # Create datasets
    experiences = buffer.get_all()
    states = torch.stack([exp.state for exp in experiences])
    values = torch.tensor([exp.value for exp in experiences], dtype=torch.float32)

    # Split train/val
    train_size = int(0.8 * len(states))
    train_dataset = ValueDataset(states[:train_size], values[:train_size])
    val_dataset = ValueDataset(states[train_size:], values[train_size:])

    print("\nDataset sizes:")
    print(f"  Training: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")

    # Create value network
    value_net = create_value_network(state_dim=9, config={"hidden_dims": [128, 64, 32], "output_activation": "tanh"})

    print("\nValue Network:")
    print(f"  Parameters: {value_net.get_parameter_count():,}")

    # Training config
    config = TrainingConfig(
        learning_rate=0.001,
        batch_size=32,
        num_epochs=50,
        checkpoint_dir="checkpoints/value",
        save_every=10,
        early_stopping_patience=10,
    )

    # Train
    print("\nTraining value network...")
    trainer = train_value_network(value_net, train_dataset, val_dataset, config)

    # Plot training curves
    print("\nPlotting training curves...")
    plot_training_curves(trainer.training_history, "Value Network Training", "value_training.png")

    print("\nValue network training complete!")
    return value_net, trainer


def plot_training_curves(history, title, filename):
    """Plot training and validation loss curves."""
    epochs = [m.epoch for m in history]
    train_losses = [m.train_loss for m in history]
    val_losses = [m.val_loss for m in history if m.val_loss is not None]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss", linewidth=2)
    if val_losses:
        plt.plot(epochs, val_losses, label="Validation Loss", linewidth=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    Path("outputs").mkdir(exist_ok=True)
    plt.savefig(f"outputs/{filename}", dpi=150)
    print(f"  Saved plot to outputs/{filename}")

    plt.close()


def evaluate_networks(policy_net, value_net):
    """Evaluate trained networks."""
    print("\n" + "=" * 60)
    print("NETWORK EVALUATION")
    print("=" * 60)

    # Test on some example states
    test_boards = [
        [1, 0, 0, 0, 1, 0, 0, 0, 1],  # Diagonal winning for X
        [1, 1, 0, -1, -1, 0, 0, 0, 0],  # Close game
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Empty board
    ]

    for i, board in enumerate(test_boards):
        print(f"\nTest Board {i + 1}: {board}")

        state = torch.tensor(board, dtype=torch.float32)

        # Policy prediction
        action_probs = policy_net.get_action_probs(state)
        best_action = torch.argmax(action_probs).item()

        print("  Policy Network:")
        print(f"    Best action: {best_action}")
        print(f"    Action probabilities: {action_probs.squeeze().tolist()}")

        # Value prediction
        value = value_net.evaluate(state)
        confidence = value_net.get_confidence(state)

        print("  Value Network:")
        print(f"    Position value: {value:.3f}")
        print(f"    Confidence: {confidence:.3f}")


def main():
    """Run complete neural training demo."""
    print("\n" + "=" * 60)
    print("NEURAL NETWORK TRAINING DEMONSTRATION")
    print("Tic-Tac-Toe Example")
    print("=" * 60)

    # Train policy network
    policy_net, policy_trainer = train_policy_demo()

    # Train value network
    value_net, value_trainer = train_value_demo()

    # Evaluate networks
    evaluate_networks(policy_net, value_net)

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print("\nPolicy Network:")
    print(f"  Final train loss: {policy_trainer.training_history[-1].train_loss:.4f}")
    print(f"  Final val loss: {policy_trainer.training_history[-1].val_loss:.4f}")
    print(f"  Best val loss: {policy_trainer.best_val_loss:.4f}")

    print("\nValue Network:")
    print(f"  Final train loss: {value_trainer.training_history[-1].train_loss:.4f}")
    print(f"  Final val loss: {value_trainer.training_history[-1].val_loss:.4f}")
    print(f"  Best val loss: {value_trainer.best_val_loss:.4f}")

    print("\n" + "=" * 60)
    print("Demo complete! Check outputs/ for training curves.")
    print("=" * 60)


if __name__ == "__main__":
    main()

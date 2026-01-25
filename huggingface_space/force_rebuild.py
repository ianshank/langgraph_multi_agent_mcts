"""
Force Hugging Face Space to rebuild by updating requirements.txt
"""

import datetime
import subprocess
import sys


def force_rebuild():
    """Force rebuild by adding timestamp to requirements.txt"""

    print("[REBUILD] Forcing Hugging Face Space rebuild...")

    # Read current requirements
    with open("requirements.txt", encoding="utf-8") as f:
        content = f.read()

    # Remove old timestamp comments
    lines = [line for line in content.split("\n") if not line.startswith("# Force rebuild timestamp:")]

    # Add new timestamp
    timestamp = datetime.datetime.now().strftime("%a, %b %d, %Y %I:%M:%S %p")
    lines.append(f"\n# Force rebuild timestamp: {timestamp}")

    # Write back
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Added timestamp: {timestamp}")

    # Git operations
    try:
        subprocess.run(["git", "add", "requirements.txt"], check=True)
        commit_msg = f"force: Rebuild Space - {timestamp}"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)

        # Push to the current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True
        )
        branch = result.stdout.strip()
        subprocess.run(["git", "push", "space", branch], check=True)
        print("[OK] Changes pushed to trigger rebuild")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Git operation failed: {e}")
        return False

    print("\n[INFO] Next steps:")
    print("1. Go to https://huggingface.co/spaces/ianshank/langgraph-mcts-demo")
    print("2. Click 'Settings' tab")
    print("3. Click 'Factory reboot' to force Space to reload")
    print("4. Wait for the Space to rebuild (2-3 minutes)")

    return True


if __name__ == "__main__":
    success = force_rebuild()
    sys.exit(0 if success else 1)

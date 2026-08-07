from pathlib import Path
from v5exec.demo import run_demo

if __name__ == "__main__":
    result = run_demo(Path(__file__).resolve().parent, run_tests=True)
    print("\nGenerated:", result["artifact_root"])
    print("Overall evidence:", result["evidence"]["overall_result"])

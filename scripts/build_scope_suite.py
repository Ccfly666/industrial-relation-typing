"""Plan or build all 20 scopes by calling the existing feature builder."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def plan(task, pool, labels_dir, name_vectors, out, device="cpu", projectors_dir=None):
    """Return five outer and fifteen inner commands; never consume outer truth."""
    commands = []
    builder = Path(__file__).with_name("build_features.py")
    for outer in range(5):
        for inner in [None, 0, 1, 2]:
            scope = f"o{outer}" + ("" if inner is None else f"i{inner}")
            name = f"{task}_{scope}"
            command = [
                sys.executable, str(builder), "--pool", str(pool),
                "--fit-labels", str(Path(labels_dir) / f"{task}__{scope}__fit_labels.npz"),
                "--name-vectors", str(name_vectors), "--outer", str(outer),
                "--out", str(Path(out) / f"{name}.npz"), "--device", device,
            ]
            if inner is not None:
                command += [
                    "--inner", str(inner), "--validation-labels",
                    str(Path(labels_dir) / f"{task}__{scope}__validation_labels.npz"),
                ]
            if projectors_dir is not None:
                command += ["--projector", str(Path(projectors_dir) / f"{name}_projector.npz")]
            commands.append(command)
    return commands


def preflight(commands):
    """Check every input and destination before starting any scope."""
    for command in commands:
        for flag in ["--pool", "--fit-labels", "--name-vectors", "--validation-labels", "--projector"]:
            if flag in command:
                path = Path(command[command.index(flag) + 1])
                if not path.is_file():
                    raise FileNotFoundError(f"Missing {flag} input: {path}")
        output = Path(command[command.index("--out") + 1])
        if output.exists():
            raise FileExistsError(f"Scope output already exists: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Call build_features.py for 5 outer and 15 inner scopes.",
        epilog="Dry-run prints the exact command arrays without reading data or writing features. "
        "Execution checks all inputs and destination collisions first; existing outputs are never overwritten.",
    )
    parser.add_argument("--task", required=True, choices=list("ECUG"))
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--name-vectors", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--projectors-dir", type=Path, help="Optional directory of TASK_oO[iI]_projector.npz files.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    commands = plan(args.task, args.pool, args.labels_dir, args.name_vectors,
                    args.out, args.device, args.projectors_dir)
    if args.dry_run:
        print(json.dumps({"task": args.task, "scopes": len(commands), "commands": commands}, indent=2))
        return
    preflight(commands)
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

"""Compare four baseline result folders only when their tasks and protocols match."""
import argparse
import csv
import math
from pathlib import Path
from statistics import mean


def read_results(folder):
    results = {}
    for path in sorted(Path(folder).glob("*.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                required = ("dataset", "split", "prompt_id", "runseed", "protocol_sha256",
                            "metric", "test_best", "epoch", "lr")
                if any(not row.get(key) for key in required):
                    raise ValueError("{} lacks protocol metadata; rerun with the comparison entry point".format(path))
                key = (row["dataset"], int(row["split"]), int(row["prompt_id"]), int(row["runseed"]))
                if key in results:
                    raise ValueError("Duplicate task/prompt/seed result: {}".format(key))
                if not math.isfinite(float(row["test_best"])):
                    raise ValueError("Nonfinite test result: {}".format(key))
                results[key] = row
    if not results:
        raise ValueError("No results found in {}".format(folder))
    return results


def compare(folders):
    tables = {name: read_results(folder) for name, folder in folders.items()}
    reference = next(iter(tables.values()))
    for name, table in tables.items():
        if set(table) != set(reference):
            raise ValueError("{} has missing/extra tasks; compare the same complete dataset selection".format(name))
        for key, row in table.items():
            other = reference[key]
            if (row["protocol_sha256"], row["metric"], int(row["epoch"]), float(row["lr"])) != (
                    other["protocol_sha256"], other["metric"], int(other["epoch"]), float(other["lr"])):
                raise ValueError("{} protocol/metric/epoch/lr differs for {}".format(name, key))
    output = []
    for dataset, seed in sorted({(key[0], key[3]) for key in reference}):
        keys = [key for key in reference if key[0] == dataset and key[3] == seed]
        tasks = sorted({key[1] for key in keys})
        row = dict(dataset=dataset, runseed=seed, metric=reference[keys[0]]["metric"], n_tasks=len(tasks))
        for name, table in tables.items():
            # Equal weight per property; average its prompts before averaging tasks.
            row[name] = mean(mean(float(table[key]["test_best"]) for key in keys if key[1] == task)
                             for task in tasks)
        output.append(row)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("gift", "gimlet", "adapter", "graph2token"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("cache/baseline_comparison.csv"))
    args = parser.parse_args()
    try:
        rows = compare({"GIFT": args.gift, "GIMLET-original": args.gimlet,
                        "GIMLET-adapter": args.adapter, "Graph2Token": args.graph2token})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("Saved", args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, "Error: {}\n".format(error))


if __name__ == "__main__":
    main()

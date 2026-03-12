"""Quick test: run Seq2Topt on a single known sequence and print result."""

import subprocess
import csv
import os
import sys
import tempfile

PLATFORM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ2TOPT_PATH = os.path.join(PLATFORM_ROOT, "tools", "Seq2Topt", "code")
SEQ2TOPT_ENV  = "lin"

# Carbonic anhydrase II (1CA2 chain A, first 50 AA for speed)
TEST_SEQ = (
    "SHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSTNEHGSEHTVDGVKYSAELHLVHWNTKYGDFGTAAQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"
)

def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv  = os.path.join(tmpdir, "input.csv")
        output_stem = os.path.join(tmpdir, "output")
        output_csv = output_stem + ".csv"

        with open(input_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["sequence"])
            writer.writerow([TEST_SEQ])

        print(f"Input CSV written to {input_csv}")
        print(f"Running Seq2Topt from cwd={SEQ2TOPT_PATH} ...")

        cmd = [
            "conda", "run", "--no-capture-output", "-n", SEQ2TOPT_ENV,
            "python", os.path.join(SEQ2TOPT_PATH, "seq2topt.py"),
            "--input", input_csv,
            "--output", output_stem,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=SEQ2TOPT_PATH
        )

        print("--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)
        print(f"Return code: {result.returncode}")

        if result.returncode != 0:
            print("FAILED")
            sys.exit(1)

        if not os.path.exists(output_csv):
            print(f"Output CSV not found at {output_csv}")
            sys.exit(1)

        print(f"\nOutput CSV ({output_csv}):")
        with open(output_csv) as fh:
            print(fh.read())

        # Parse result
        with open(output_csv, newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        print("Parsed rows:", rows)
        if rows:
            topt = rows[0].get("pred_topt", "N/A")
            print(f"\nPredicted Topt: {topt} °C")
            print("SUCCESS")
        else:
            print("No rows in output. FAILED")
            sys.exit(1)

if __name__ == "__main__":
    main()

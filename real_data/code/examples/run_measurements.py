"""
Generate one synthetic network per era and write metrics CSV.
"""
import json, os
from metrics.measure import write_summary_csv
from synth import hunter_gatherer, agrarian, early_modern, imperial_trade, industrial, digital

HERE = os.path.dirname(__file__)

def main():
    with open(os.path.join(HERE, "era_configs.json"), "r") as f:
        cfg = json.load(f)
    graphs = []
    graphs.append(("hunter_gatherer", hunter_gatherer.generate(**cfg["hunter_gatherer"], seed=42), None))
    graphs.append(("agrarian", agrarian.generate(**cfg["agrarian"], seed=42), None))
    graphs.append(("early_modern", early_modern.generate(**cfg["early_modern"], seed=42), None))
    graphs.append(("imperial_trade", imperial_trade.generate(**cfg["imperial_trade"], seed=42), None))
    graphs.append(("industrial", industrial.generate(**cfg["industrial"]), None))
    graphs.append(("digital", digital.generate(**cfg["digital"], seed=42), None))
    out_csv = os.path.join(HERE, "era_metrics_demo.csv")
    df = write_summary_csv(graphs, out_csv)
    print("Wrote metrics to:", out_csv)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Print row/column sizes for AFL .df pickle files.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing *_data.df files.")
    parser.add_argument(
        "--files",
        nargs="*",
        default=["train_data.df", "Val_data.df", "Test_data.df"],
        help="Pickle dataframe file names to inspect.",
    )
    args = parser.parse_args()

    try:
        import pandas as pd
    except ModuleNotFoundError:
        print("pandas is not installed in this environment. Install pandas, then rerun this script.")
        return

    print(f"Inspecting dataframe sizes in: {args.data_dir}")
    for name in args.files:
        path = os.path.join(args.data_dir, name)
        if not os.path.exists(path):
            print(f"- {name}: MISSING")
            continue
        df = pd.read_pickle(path)
        print(f"- {name}: rows={len(df)}, cols={len(df.columns)}, shape={df.shape}")


if __name__ == "__main__":
    main()

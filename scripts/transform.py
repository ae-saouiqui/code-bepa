from argparse import ArgumentParser
import polars as pl
from pathlib import Path
from tqdm import tqdm



parser = ArgumentParser()


parser.add_argument(
    "-i",
    "--ipath",
    type=str,
    required=True
)

parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=True
)


args = parser.parse_args()


def main():
    path = Path(args.ipath)
    output_path = Path(args.output)

    if not path.exists() or not path.is_file() or path.suffix != ".jsonl" :
        # Make it generic it is not used for production
        raise Exception("Failed to read this file") 

    output_path.mkdir(parents=True, exist_ok=True)

    lazy_df = pl.scan_ndjson(path)

    python_df = lazy_df.filter(
        (pl.col("language") == "PYTHON3") 
        & (pl.col("description").is_not_null()) 
        & (pl.col("solution").is_not_null())
    )

    # removing duplicates 
    python_df = python_df.unique(subset=["description","solution"])

    for i,batch in tqdm(
        enumerate(python_df.collect_batches(chunk_size=10_000)),
        desc="Processing batches"
        ) : 
        df = batch.select(
            pl.col("description").alias("problem"),
            pl.col("solution")
        )

        df.write_parquet(output_path / f"part-{i}.parquet")


if __name__ == "__main__" : 
    main()

    
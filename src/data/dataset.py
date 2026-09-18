from torch.utils.data import IterableDataset,get_worker_info
import polars as pl

from pathlib import Path

class CodeBEPADataSet(IterableDataset):
    """A dataset that reads problem-solution pairs from a Parquet file.

    The dataset reads the data in streaming batches, so the whole
    dataset does not need to be loaded into memory at once.

    Parameters
    ----------
    dataset_path : str or pathlib.Path
        Path to the Parquet file containing the dataset.

    Yields
    ------
    tuple
        A tuple containing the problem and its solution.

    Raises
    ------
    TypeError
        If the dataset path is not a string or a pathlib.Path.
    FileNotFoundError
        If the given dataset path does not exist.
    """

    
    def __init__(self,dataset_path :Path | str):
        if not isinstance(dataset_path,(Path,str)):
            raise TypeError("Unmatched Type  : 'dataset_path' must be either of type 'str' or 'pathlib.Path' ")
        self._path = Path(dataset_path)
        if not self._path.exists():
            raise FileNotFoundError(f" \"{dataset_path}\" does not exist")

    def __iter__(self):
        wroker_info = get_worker_info()

        # In case the number of workers is <=0 
        if wroker_info is None :
            num_workers = 1
            worker_id = 0
        else :
            num_workers = wroker_info.num_workers
            worker_id = wroker_info.id

        index = 0
        lf = pl.scan_parquet(self._path).select(["problem","solution"])
        for batch in lf.collect_batches(engine="streaming"):
            for problem,solution  in batch.iter_rows():
                if (index % num_workers) == worker_id:
                    yield problem, solution
                index +=1            
"""Module to access tableshift BRFSS data using the tableshift package."""

from __future__ import annotations

import logging

# import pickle
from pathlib import Path

import pandas as pd
from tableshift import get_iid_dataset

from ..dataset import Dataset
from .tableshift_tasks import (
    TableshiftBRFSSTaskMetadata,
    passthrough_preprocessor_config,
)

DEFAULT_DATA_DIR = Path("~/data").expanduser().resolve()
DEFAULT_TEST_SIZE = 0.1
DEFAULT_VAL_SIZE = 0.1
DEFAULT_SEED = 42


class TableshiftBRFSSDataset(Dataset):
    """Wrapper for tableshift BRFSS datasets."""

    def __init__(
        self,
        data: pd.DataFrame,
        full_brfss_data: pd.DataFrame,
        task: TableshiftBRFSSTaskMetadata,
        test_size: float = DEFAULT_TEST_SIZE,
        val_size: float = DEFAULT_VAL_SIZE,
        subsampling: float = None,
        seed: int = 42,
    ):
        self.full_brfss_data = full_brfss_data
        super().__init__(
            data=data,
            task=task,
            test_size=test_size,
            val_size=val_size,
            subsampling=subsampling,
            seed=seed,
        )

    @classmethod
    def make_from_task(
        cls,
        task: str | TableshiftBRFSSTaskMetadata,
        cache_dir: str | Path = None,
        survey_year: str = None,
        seed: int = DEFAULT_SEED,
        load_dataset_if_not_cached=True,  # add 'extra control' before downloading dataset
        **kwargs,
    ):
        """Construct an TableshiftBRFSSDataset object from a given Tableshift BRFSS task.

        Can customize survey sample parameters (survey year).

        Parameters
        ----------
        task : str | TableshiftBRFSSTaskMetadata
            The name of the Tableshift BRFSS task or the task object itself.
        cache_dir : str | Path, optional
            The directory where Tableshift BRFSS data is (or will be) saved to, by default
            uses DEFAULT_DATA_DIR.
        survey_year : str, optional
            The year from which to load survey data, by default DEFAULT_SURVEY_YEARS.
        seed : int, optional
            The random seed, by default DEFAULT_SEED.
        **kwargs
            Extra key-word arguments to be passed to the Dataset constructor.
        """
        # Parse task if given a string
        task_obj = TableshiftBRFSSTaskMetadata.get_task(task) if isinstance(task, str) else task
        logging.debug(f"task_obj : {task_obj.tableshift_obj.__dict__}")

        # Create "folktables" sub-folder under the given cache dir
        cache_dir = Path(cache_dir or DEFAULT_DATA_DIR).expanduser().resolve() / "tableshift" / task_obj.name.lower()
        if not cache_dir.exists():
            logging.warning(f"Creating cache directory '{cache_dir}' for TableShift data.")
            cache_dir.mkdir(exist_ok=True, parents=False)

        # if not already available, load data
        csv_file = cache_dir / f"{task_obj.name.lower()}_all.csv"
        # pickle_file = cache_dir / f"{task_obj.name.lower()}.pickle"
        if csv_file.exists():
            logging.info("Loading TableShift task data from cache...")
            logging.warning("Assuming dataset is preprocessed as wanted.")
            df = pd.read_csv(csv_file.as_posix())
        else:
            if not load_dataset_if_not_cached:
                raise ValueError("Could not load dataset from cache, save dataset locally first.")
            else:
                # Load Tableshift data source
                logging.info("Loading TableShift task data (may take a while)...")
                tab_dataset = get_iid_dataset(
                    task_obj.name.lower(),
                    cache_dir=cache_dir,
                    preprocessor_config=passthrough_preprocessor_config,
                )
                X, y, _, _ = tab_dataset.get_pandas("all")
                df = pd.concat([X, y], axis=1)
                for col in df.select_dtypes(include="category").columns:
                    if col not in ["SMOKDAY2", "TOLDHI"]:
                        df[col] = pd.to_numeric(df[col].astype(str), errors="coerce")
                df.to_csv(csv_file, index=0)
                # will convert categorical columns to numeric
                df = pd.read_csv(csv_file.as_posix())

        # Parse data for this task
        parsed_data = cls._parse_task_data(df, task_obj)

        return cls(
            data=parsed_data,
            full_brfss_data=df,
            task=task_obj,
            seed=seed,
            **kwargs,
        )

    @property
    def task(self) -> TableshiftBRFSSTaskMetadata:
        return self._task

    @task.setter
    def task(self, new_task: TableshiftBRFSSTaskMetadata):
        # Parse data rows for new Tableshift BRFSS task
        self._data = self._parse_task_data(self._full_acs_data, new_task)

        # Re-make train/test/val split
        self._train_indices, self._test_indices, self._val_indices = self._make_train_test_val_split(
            self._data, self.test_size, self.val_size, self._rng
        )

        # Check if sub-sampling is necessary (it's applied only to train/test/val indices)
        if self.subsampling is not None:
            self._subsample_train_test_val_indices(self.subsampling)

        self._task = new_task

    @classmethod
    def _parse_task_data(cls, full_df: pd.DataFrame, task: TableshiftBRFSSTaskMetadata) -> pd.DataFrame:
        """Parse a DataFrame for compatibility with the given task object.

        Parameters
        ----------
        full_df : pd.DataFrame
            Full DataFrame. Some rows and/or columns may be discarded for each
            task.
        task : TableshiftBRFSSTaskMetadata
            The task object used to parse the given data.

        Returns
        -------
        parsed_df : pd.DataFrame
            Parsed DataFrame in accordance with the given task.
        """
        # Pre-process the data if necessary (already included in loading dataset)
        # if (
        #     isinstance(task, TableshiftBRFSSTaskMetadata)
        #     and task.tableshift_obj is not None
        # ):
        #     parsed_df = task.tableshift_obj._preprocess(full_df)  ## TODO
        # else:
        #     parsed_df = full_df
        parsed_df = full_df

        # Threshold the target column if necessary
        if task.target is not None and task.target_threshold is not None and task.get_target() not in parsed_df.columns:
            parsed_df[task.get_target()] = task.target_threshold.apply_to_column_data(parsed_df[task.target])

        return parsed_df

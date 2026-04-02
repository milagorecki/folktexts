"""Module to access tableshift BRFSS data using the tableshift package."""

from __future__ import annotations

import logging

# import pickle
from pathlib import Path

import pandas as pd

from ..dataset import Dataset
from .load_sipp import download_sipp, load_sipp, preprocess_sipp
from .sipp_tasks import (
    SIPPTaskMetadata,
)

DEFAULT_DATA_DIR = Path("~/data").expanduser().resolve()
DEFAULT_TEST_SIZE = 0.1
DEFAULT_VAL_SIZE = 0.1
DEFAULT_SEED = 42


class SIPPDataset(Dataset):
    """Wrapper for tableshift BRFSS datasets."""

    def __init__(
        self,
        data: pd.DataFrame,
        full_sipp_data: pd.DataFrame,
        task: SIPPTaskMetadata,
        test_size: float = DEFAULT_TEST_SIZE,
        val_size: float = DEFAULT_VAL_SIZE,
        subsampling: float = None,
        seed: int = 42,
    ):
        self.full_sipp_data = full_sipp_data
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
        task: str | SIPPTaskMetadata,
        cache_dir: str | Path = None,
        survey_year: str = None,
        seed: int = DEFAULT_SEED,
        load_dataset_if_not_cached=True,  # add 'extra control' before downloading dataset
        **kwargs,
    ):
        """Construct an SIPPDataset object from a given Tableshift BRFSS task.

        Can customize survey sample parameters (survey year).

        Parameters
        ----------
        task : str | SIPPTaskMetadata
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
        task_obj = SIPPTaskMetadata.get_task(task) if isinstance(task, str) else task

        # Create "sipp" sub-folder under the given cache dir
        cache_dir = Path(cache_dir or DEFAULT_DATA_DIR).expanduser().resolve() / "sipp"
        if not cache_dir.exists():
            logging.warning(f"Creating cache directory '{cache_dir}' for SIPP data.")
            cache_dir.mkdir(exist_ok=True, parents=False)

        # if not already available, load data
        csv_file = cache_dir / f"{task_obj.name.lower()}_2014.csv"
        if csv_file.exists():
            logging.info("Loading SIPP task data from cache...")
            df = pd.read_csv(csv_file.as_posix())
        else:
            print((cache_dir / "sipp_2014_wave_1.csv").exists(), (cache_dir / "sipp_2014_wave_2.csv").exists())
            if not (cache_dir / "sipp_2014_wave_1.csv").exists() and not (cache_dir / "sipp_2014_wave_2.csv").exists():
                print("Download SIPP data... (May take a while)")
                download_sipp(save_path=cache_dir)
                print("Preprocess SIPP... (May take a while)")
                preprocess_sipp(data_dir=cache_dir)
            X, y = load_sipp(data_dir=cache_dir)
            df = pd.concat([X, y], axis=1)

        # Parse data for this task
        parsed_data = cls._parse_task_data(df, task_obj)

        return cls(
            data=parsed_data,
            full_sipp_data=df,
            task=task_obj,
            seed=seed,
            **kwargs,
        )

    @property
    def task(self) -> SIPPTaskMetadata:
        return self._task

    @task.setter
    def task(self, new_task: SIPPTaskMetadata):
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
    def _parse_task_data(cls, full_df: pd.DataFrame, task: SIPPTaskMetadata) -> pd.DataFrame:
        """Parse a DataFrame for compatibility with the given task object.

        Parameters
        ----------
        full_df : pd.DataFrame
            Full DataFrame. Some rows and/or columns may be discarded for each
            task.
        task : SIPPTaskMetadata
            The task object used to parse the given data.

        Returns
        -------
        parsed_df : pd.DataFrame
            Parsed DataFrame in accordance with the given task.
        """
        parsed_df = full_df

        # Threshold the target column if necessary
        if task.target is not None and task.target_threshold is not None and task.get_target() not in parsed_df.columns:
            parsed_df[task.get_target()] = task.target_threshold.apply_to_column_data(parsed_df[task.target])

        return parsed_df

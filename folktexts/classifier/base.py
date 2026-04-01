"""Module containing the base class for all LLM risk classifiers.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Callable
from os import remove

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm.auto import tqdm

from folktexts.dataset import Dataset
from folktexts.evaluation import compute_best_threshold
from folktexts.prompting import encode_row_prompt as default_encode_row_prompt
from folktexts.qa_interface import DirectNumericQA, MultipleChoiceQA
from folktexts.task import TaskMetadata

from .._utils import hash_dict, hash_function

DEFAULT_CONTEXT_SIZE = 600
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_NEW_TOKENS = 10_000

SCORE_COL_NAME = "risk_score"
LABEL_COL_NAME = "label"


class LLMClassifier(BaseEstimator, ClassifierMixin, ABC):
    """An interface to produce risk scores and class predictions with an LLM."""

    DEFAULT_INFERENCE_KWARGS = {
        "context_size": DEFAULT_CONTEXT_SIZE,
        "batch_size": DEFAULT_BATCH_SIZE,
        "enable_thinking": False,                   ## TODO: wrong place? 
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS    ## TODO: wrong place? 
    } 

    def __init__(
        self,
        model_name: str,
        task: TaskMetadata | str,
        custom_prompt_prefix: str = None,
        encode_row: Callable[[pd.Series], str] = None,
        threshold: float = 0.5,
        correct_order_bias: bool = True,
        seed: int = 42,
        prompt_variation: dict = {},
        **inference_kwargs,
    ):
        """Creates an LLMClassifier object.

        Parameters
        ----------
        model_name : str
            The model name or ID.
        task : TaskMetadata | str
            The task metadata object or name of an already created task.
        custom_prompt_prefix : str, optional
            A custom prompt prefix to supply to the model before the encoded
            row data, by default None.
        encode_row : Callable[[pd.Series], str], optional
            The function used to encode tabular rows into natural text. If not
            provided, will use the default encoding function for the task.
        threshold : float, optional
            The classification threshold to use when outputting binary
            predictions, by default 0.5. Must be between 0 and 1. Will be
            re-calibrated if `fit` is called.
        correct_order_bias : bool, optional
            Whether to correct ordering bias in multiple-choice Q&A questions,
            by default True.
        seed : int, optional
            The random seed - used for reproducibility.
        **inference_kwargs
            Additional keyword arguments to be used at inference time. Options
            include `context_size` and `batch_size`.
        """

        # Set classifier metadata
        self._model_name = model_name
        # self.enable_thinking = False
        # self.max_new_tokens # what is the models native max  

        self._task = TaskMetadata.get_task(task) if isinstance(task, str) else task
        self._custom_prompt_prefix = custom_prompt_prefix
        self._prompt_variation = prompt_variation

        self._encode_row = encode_row or partial(
            default_encode_row_prompt,
            task=self.task,
            custom_prompt_prefix=self.custom_prompt_prefix,
            prompt_variation=self.prompt_variation
        )

        self._threshold = threshold
        self._threshold_fitted_on = 0
        self._threshold_obj = 'balanced_accuracy' ##TODO: remove (but will change benchmark hash)
        self._correct_order_bias = correct_order_bias
        self._seed = seed

        # Default inference kwargs
        self._inference_kwargs = self.DEFAULT_INFERENCE_KWARGS.copy()
        self._inference_kwargs.update(inference_kwargs)

        # Fixed sklearn parameters
        self.classes_ = np.array([0, 1])
        self._is_fitted = False

    def __hash__(self) -> int:
        """Generate a unique hash for this object."""

        # All parameters that affect the model's behavior
        hash_params = dict(
            model_name=self.model_name,
            task_hash=hash(self.task),
            custom_prompt_prefix=self.custom_prompt_prefix,
            prompt_variation=hash_dict(self.prompt_variation),
            correct_order_bias=self.correct_order_bias,
            threshold=self.threshold,
            encode_row_hash=hash_function(self.encode_row),
        )

        return int(hash_dict(hash_params), 16)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def task(self) -> TaskMetadata:
        return self._task

    @property
    def custom_prompt_prefix(self) -> str:
        return self._custom_prompt_prefix

    @property
    def prompt_variation(self) -> dict:
        return self._prompt_variation

    @property
    def encode_row(self) -> Callable[[pd.Series], str]:
        return self._encode_row

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        if not 0 <= value <= 1:
            logging.error(f"Threshold must be between 0 and 1; got {value}.")

        # Clip threshold to valid range
        self._threshold = np.clip(value, 0, 1)
        logging.warning(f"Setting {self.model_name} threshold to {self._threshold}.")

    @property
    def correct_order_bias(self) -> bool:
        return self._correct_order_bias

    @correct_order_bias.setter
    def correct_order_bias(self, value: bool):
        self._correct_order_bias = value
        logging.warning(f"Setting {self.model_name} correct_order_bias to {value}.")

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def inference_kwargs(self) -> dict:
        return self._inference_kwargs

    def set_inference_kwargs(self, **kwargs):
        """Set inference kwargs for the model."""
        self._inference_kwargs.update(kwargs)

    def __sklearn_is_fitted__(self):
        """Check fitted status and return a Boolean value."""
        return hasattr(self, "_is_fitted") and self._is_fitted

    @staticmethod
    def _get_positive_class_scores(risk_scores: np.ndarray) -> np.ndarray:
        """Helper function to get positive class scores from risk scores."""
        if len(risk_scores.shape) > 1:
            return risk_scores[:, -1]
        else:
            return risk_scores

    @staticmethod
    def _make_predictions_multiclass(pos_class_scores: np.ndarray) -> np.ndarray:
        """Converts positive class scores to multiclass scores."""
        return np.column_stack([1 - pos_class_scores, pos_class_scores])

    def fit(self, X, y, *, false_pos_cost=1.0, false_neg_cost=1.0, threshold_obj='balanced_accuracy', **kwargs):
        """Uses the provided data sample to fit the prediction threshold."""

        # Compute risk estimates for the data
        y_pred_scores = self._get_positive_class_scores(
            self.predict_proba(X, **kwargs)
        )

        # Compute the best threshold for the given data
        self.threshold = compute_best_threshold(
            y, y_pred_scores,
            false_pos_cost=false_pos_cost,
            false_neg_cost=false_neg_cost,
            maximize=threshold_obj
        )
        self._threshold_obj = threshold_obj ## TODO: save in llm_clf._threshold_obj before calling fun

        # Update sklearn is_fitted status
        self._is_fitted = True
        return self

    def _load_predictions_from_disk(
        self,
        predictions_save_path: str | Path,
        data: pd.DataFrame,
    ) -> np.ndarray | None:
        """Attempts to load pre-computed predictions from disk."""

        # Load predictions from disk
        predictions_save_path = Path(predictions_save_path).with_suffix(".csv")
        predictions_df = pd.read_csv(predictions_save_path, index_col=0)

        # Check if index matches our current dataframe
        if predictions_df.index.equals(data.index):
            return predictions_df[SCORE_COL_NAME].values
        else:
            logging.error("Saved predictions do not match the current dataframe.")
            return None

    def predict(
        self,
        data: pd.DataFrame,
        predictions_save_path: str | Path = None,
        labels: pd.Series | np.ndarray = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Returns binary predictions for the given data."""
        risk_scores = self.predict_proba(
            data,
            predictions_save_path=predictions_save_path,
            labels=labels,
        )
        return (self._get_positive_class_scores(risk_scores) >= self.threshold).astype(int)

    def predict_proba(
        self,
        data: pd.DataFrame,
        predictions_save_path: str | Path = None,
        labels: pd.Series | np.ndarray = None,
    ) -> np.ndarray:
        """Returns probability estimates for the given data.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to compute risk estimates for.
        predictions_save_path : str | Path, optional
            If provided, will save the computed risk scores to this path in
            disk. If the path exists, will attempt to load pre-computed
            predictions from this path.
        labels : pd.Series | np.ndarray, optional
            The labels corresponding to the provided data. Not required to
            compute predictions. Will only be used to save alongside predictions
            to disk.

        Returns
        -------
        risk_scores : np.ndarray
            The risk scores for the given data.
        """
        # Validate arguments
        if labels is not None and predictions_save_path is None:
            logging.error(
                "** Ignoring `labels` argument as `predictions_save_path` was not provided. **"
                "The `labels` argument is only used in conjunction with "
                "`predictions_save_path` to save alongside predictions to disk. ")

        # Check if `predictions_save_path` exists and load predictions if possible
        logging.info(f"Check if predictions_save_path '{predictions_save_path}' exists:{Path(predictions_save_path).exists()}")
        if predictions_save_path is not None and Path(predictions_save_path).exists():
            result = self._load_predictions_from_disk(predictions_save_path, data=data)
            if result is not None:
                logging.info(f"Loaded predictions from {predictions_save_path}.")
                return self._make_predictions_multiclass(result)
            else:
                logging.error(
                    f"Failed to load predictions from {predictions_save_path}. "
                    f"Re-computing predictions and overwriting local file..."
                )

        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                f"`data` must be a pd.DataFrame, received {type(data)} instead.")

        # Compute risk estimates
        risk_scores = self.compute_risk_estimates_for_dataframe(df=data,
                                                                save_intermed={'path': predictions_save_path,
                                                                               'labels': labels})

        # Save to disk if `predictions_save_path` is provided
        if predictions_save_path is not None:
            predictions_save_path = Path(predictions_save_path).with_suffix(".csv")
            logging.info(f'Saving predictions to {predictions_save_path}')

            predictions_df = pd.DataFrame(risk_scores, index=data.index, columns=[SCORE_COL_NAME])
            predictions_df[LABEL_COL_NAME] = labels
            predictions_df.to_csv(predictions_save_path, index=True, mode="w")

        return self._make_predictions_multiclass(risk_scores)

    @abstractmethod
    def _query_prompt_risk_estimates_batch(
        self,
        prompts_batch: list[str],
        *,
        question: MultipleChoiceQA | DirectNumericQA,
        context_size: int = None,
    ) -> np.ndarray:
        """Query model with a batch of prompts and return risk estimates."""
        raise NotImplementedError("Calling an abstract method :: Use one of the subclasses of LLMClassifier.")

    def compute_risk_estimates_for_dataframe(
        self,
        df: pd.DataFrame,
        save_intermed: dict = {'path': None, 'labels': None}
    ) -> np.ndarray:
        """Compute risk estimates for a specific dataframe (internal helper function).

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to compute risk estimates for.
        save_intermed: dict
            A dictionary containing information for saving intermediate results. Should contain the keys:
                - 'path': str | Path, optional
                - 'labels': pd.Series | np.ndarray, optional
        Returns
        -------
        risk_scores : np.ndarray
            The risk estimates for each row in the dataframe.
        """

        # Initialize risk scores and other constants
        fill_value = -1
        risk_scores = np.empty(len(df))
        risk_scores.fill(fill_value)    # fill with -1's

        batch_size = self._inference_kwargs["batch_size"]
        context_size = self._inference_kwargs["context_size"]
        num_batches = math.ceil(len(df) / batch_size)

        # Get questions to ask
        q = self.task.question
        questions = [q]
        if self.correct_order_bias:
            if isinstance(q, DirectNumericQA):
                logging.info("No need to correct ordering bias for DirectNumericQA prompting.")
            elif isinstance(q, MultipleChoiceQA):
                questions = list(MultipleChoiceQA.create_answer_keys_permutations(q))
            else:
                logging.error(f"Unknown question type '{type(q)}'; cannot correct ordering bias.")

        # Prepare storage for model responses
        model_outputs = [] # either text or tlp

        # initialize path to save intermediate batch results
        batch_path = None

        # Compute risk estimates per batch
        for batch_idx in tqdm(range(num_batches), desc="Computing risk estimates"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            batch_data = df.iloc[start_idx:end_idx]
            batch_row_ids = batch_data.index.values

            batch_risk_scores = np.empty((len(batch_data), len(questions)))
            for q_idx, q in enumerate(questions):

                # Encode batch data into natural text prompts
                # TODO: potential improvement: encode outside of loop with question placeholder, only replace placeholder
                data_texts_batch = [
                    self.encode_row(
                        row,
                        question=q,
                        custom_prompt_prefix=self.custom_prompt_prefix,
                    )
                    for _, row in batch_data.iterrows()
                ]

                # Query the model with the batch of data
                risk_estimates_batch, responses_batch = self._query_prompt_risk_estimates_batch(
                    prompts_batch=data_texts_batch,
                    question=q,
                    context_size=context_size,
                )

                # Store risk estimates for current question
                batch_risk_scores[:, q_idx] = np.clip(risk_estimates_batch, 0, 1)
                if q.get_answer_from_generated_text and isinstance(q, MultipleChoiceQA):
                    for i, resp in enumerate(responses_batch):
                        if resp is not None:
                            extracted_answer = q.get_answer_key_from_value(q.get_answer_from_generated_text(resp.get("response"))) if resp.get("response") is not None else ""
                            model_outputs.append({
                                "row_idx" : batch_row_ids[i],
                                "question_idx" : q_idx,
                                "prompt" : data_texts_batch[i],  # includes row information and question
                                "reasoning": resp.get("reasoning", ""),
                                "response": resp.get("response", ""),
                                "extracted_answer": extracted_answer or "",
                            })
                        else:
                            model_outputs.append({
                                "row_idx" : batch_row_ids[i],
                                "question_idx" : q_idx,
                                "prompt" : data_texts_batch[i],  # includes row information and question
                                "reasoning": "",
                                "response": "",
                                "extracted_answer": "",
                            })
                        
            # Calculate final risk score (mean across questions)
            risk_scores[start_idx: end_idx] = batch_risk_scores.mean(axis=1)

            # if questions[0].get_answer_from_generated_text:
            #     batch_responses_all_questions = zip(*question_responses) if len(questions) > 1 else question_responses
            #     model_outputs.extend(batch_responses_all_questions)
            #     logging.debug(f"length model outputs after batch {batch_idx}: {len(model_outputs)}")
            # else:
            #     logging.debug("Skip storing last token probabilities, update if needed.")

            # log items where one answer could not be extracted or permutation of the questions lead to different outcomes
            tol = 1e-8
            arr = batch_risk_scores.reshape(-1, 1) if batch_risk_scores.ndim == 1 else batch_risk_scores
            undecided_or_disagree = ~(np.isclose(arr, 0, atol=tol) | np.isclose(arr, 1, atol=tol))
            if np.any(undecided_or_disagree):
                idx_unclear = np.nonzero(undecided_or_disagree)[0] #only row indices (1D)
                msg = (f"Risk scores: {batch_risk_scores[idx_unclear]}"
                       f"\nRisk scores mean: {risk_scores[start_idx: end_idx][idx_unclear]}")
                if questions[0].get_answer_from_generated_text:
                    tmp_texts = [f"response {i}\n{item}\n\n" for i, item in enumerate(model_outputs[-len(batch_data)*len(questions):]) if i in idx_unclear]
                logging.debug(msg +  f"\nCorresponding responses: {tmp_texts}")

            # Save intermediate results
            path = save_intermed.get('path')
            
            if batch_idx % 10 == 0 and path is not None:
                # add _batch + suffix to path
                batch_path = path.with_stem(path.stem + "_batch")
                self._save_intermediate_results(
                    path=batch_path, 
                    risk_scores = risk_scores[:end_idx],
                    labels = save_intermed['labels'][:end_idx],
                    responses = model_outputs if questions[0].get_answer_from_generated_text else None,
                    )

        # Check that all risk scores were computed
        assert not np.isclose(risk_scores, fill_value).any()
        # remove intermediate saves after all are computed
        if batch_path is not None and Path(batch_path).exists() and str(batch_path).endswith('_batch.csv'):
            logging.info(f"Removing file '{batch_path}'.")
            remove(batch_path)

        return risk_scores
    
    def _save_intermediate_results(self, path: str | Path, 
                                   risk_scores: np.ndarray, 
                                   labels: pd.Series | np.ndarray, 
                                   responses:list = None, ):
        # save risk_scores as pd.DataFrame
        predictions_df = pd.DataFrame(risk_scores,
                                      index=labels.index,
                                      columns=[SCORE_COL_NAME])
        predictions_df[LABEL_COL_NAME] = labels
        logging.info(f'Saving intermediate results to {path}')
        predictions_df.to_csv(path, index=True, mode="w")

        if responses is not None:
            response_df = pd.DataFrame(responses)
                                    #    index=labels.index,
                                    #    columns=[f'response_{i}' for i in range(num_questions)])
            logging.debug(f"response_df shape: {response_df.shape}, columns: {response_df.columns}")
            # TODO: match by row index 
            response_df[SCORE_COL_NAME] = response_df['row_idx'].map(predictions_df[SCORE_COL_NAME])
            response_df[LABEL_COL_NAME] = response_df['row_idx'].map(predictions_df[LABEL_COL_NAME])
            responses_file_name = path.with_stem(path.stem.replace("_predictions_batch", "_responses"))
            logging.info(f'Saving responses to {responses_file_name}')
            response_df.to_csv(responses_file_name, index=False, encoding='utf-8', mode="w")

    def compute_risk_estimates_for_dataset(
        self,
        dataset: Dataset,
    ) -> dict[str, np.ndarray]:
        """Computes risk estimates for each row in the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to compute risk estimates for.

        Returns
        -------
        results : dict[str, np.ndarray]
            The risk estimates for each data type in the dataset (usually "train",
            "val", "test").
        """
        data_types = {
            "train": dataset.get_train()[0],
            "test": dataset.get_test()[0],
        }
        if dataset.get_val() is not None:
            data_types["val"] = dataset.get_val()[0]

        results = {
            data_type: self.compute_risk_estimates_for_dataframe(
                df=df,
            )
            for data_type, df in data_types.items()
        }

        return results

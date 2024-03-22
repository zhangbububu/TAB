from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy
import typing
import time
import uuid

from d3m import container
from d3m.primitive_interfaces import base, transformer
from d3m.metadata import base as metadata_base, hyperparams

from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
from d3m.base import utils as base_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from ..common.TODSBasePrimitives import TODSTransformerPrimitiveBase


import os.path

import time
import statsmodels.api as sm

__all__ = ('BKFilterPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame
from tods.utils import construct_primitive_metadata

class Hyperparams(hyperparams.Hyperparams):
    # Tuning
    low = hyperparams.UniformInt(
        lower=0,
        upper=100000000,
        default=6,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Minimum period for oscillations, ie., Baxter and King suggest that the Burns-Mitchell U.S. business cycle has 6 for quarterly data and 1.5 for annual data.",
    )
    high = hyperparams.UniformInt(
        lower=0,
        upper=100000000,
        default=32,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Maximum period for oscillations BK suggest that the U.S. business cycle has 32 for quarterly data and 8 for annual data.",
    )
    K = hyperparams.UniformInt(
        lower=0,
        upper=100000000,
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Lead-lag length of the filter. Baxter and King propose a truncation length of 12 for quarterly data and 3 for annual data.",
    )

    # Control
    columns_using_method= hyperparams.Enumeration(
        values=['name', 'index'],
        default='index',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Choose to use columns by names or indecies. If 'name', \"use_columns\" or \"exclude_columns\" is used. If 'index', \"use_columns_name\" or \"exclude_columns_name\" is used."
    )
    use_columns_name = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](''),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column names to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )
    exclude_columns_name = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](''),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column names to not operate on. Applicable only if \"use_columns_name\" is not provided.",
    )
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )  
    return_result = hyperparams.Enumeration(
        values=['append', 'replace', 'new'],
        default='append',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should parsed columns be appended, should they replace original columns, or should only parsed columns be returned? This hyperparam is ignored if use_semantic_types is set to false.",
    )
    use_semantic_types = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Controls whether semantic_types metadata will be used for filtering columns in input dataframe. Setting this to false makes the code ignore return_result and will produce only the output dataframe"
    )
    add_index_columns = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )
    error_on_no_input = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Throw an exception if no input column is selected/provided. Defaults to true to behave like sklearn. To prevent pipelines from breaking set this to False.",
    )
    
    return_semantic_type = hyperparams.Enumeration[str](
        values=['https://metadata.datadrivendiscovery.org/types/Attribute', 'https://metadata.datadrivendiscovery.org/types/ConstructedAttribute'],
        default='https://metadata.datadrivendiscovery.org/types/Attribute',
        description='Decides what semantic type to attach to generated attributes',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

    
class BKFilterPrimitive(TODSTransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Filter a time series using the Baxter-King bandpass filter.

Parameters
----------
    low: int
        Minimum period for oscillations, ie., Baxter and King suggest that the Burns-Mitchell U.S. business cycle has 6 for quarterly data and 1.5 for annual data.
    
    high: int
        Maximum period for oscillations BK suggest that the U.S. business cycle has 32 for quarterly data and 8 for annual data.
    K: int
        Lead-lag length of the filter. Baxter and King propose a truncation length of 12 for quarterly data and 3 for annual data.  
    
.. dropdown:: Control Parameter

    use_columns: Set
        A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.
    
    exclude_columns: Set
        A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.
    
    return_result: Enumeration
        Should parsed columns be appended, should they replace original columns, or should only parsed columns be returned? This hyperparam is ignored if use_semantic_types is set to false.
    
    use_semantic_types: Bool
        Controls whether semantic_types metadata will be used for filtering columns in input dataframe. Setting this to false makes the code ignore return_result and will produce only the output dataframe.
    
    add_index_columns: Bool
        Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".
    
    error_on_no_input: Bool(
        Throw an exception if no input column is selected/provided. Defaults to true to behave like sklearn. To prevent pipelines from breaking set this to False.
    
    return_semantic_type: Enumeration[str](
        Decides what semantic type to attach to generated attributes'
    """

    metadata = construct_primitive_metadata(module='feature_analysis', name='bk_filter', id='BKFilterPrimitive', primitive_family='feature_construct', hyperparams=['low', 'high', 'K'], description='Baxter-King Filter Primitive')
    
    


    def _produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Process the testing data.
        Args:
            inputs: Container DataFrame.

        Returns:
            Container DataFrame after BKFilter.
        """
        # Get cols to fit.
        self._fitted = False
        self._training_inputs, self._training_indices = self._get_columns_to_fit(inputs, self.hyperparams)
        self._input_column_names = self._training_inputs.columns


        if len(self._training_indices) > 0:
            # self._clf.fit(self._training_inputs)
            self._fitted = True
        else:   # pragma: no cover
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")



        if not self._fitted:    # pragma: no cover
            raise PrimitiveNotFittedError("Primitive not fitted.")
        sk_inputs = inputs
        if self.hyperparams['use_semantic_types']:  # pragma: no cover
            sk_inputs = inputs.iloc[:, self._training_indices]
        output_columns = []
        if len(self._training_indices) > 0:
            sk_output = self._bkfilter(sk_inputs, low=self.hyperparams['low'], high=self.hyperparams['high'], K=self.hyperparams['K'])
            if sparse.issparse(sk_output):  # pragma: no cover
                sk_output = sk_output.toarray()
            outputs = self._wrap_predictions(inputs, sk_output)

            if len(outputs.columns) == len(self._input_column_names):
                outputs.columns = self._input_column_names
            output_columns = [outputs]           
            
        else:   # pragma: no cover
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")
        outputs = base_utils.combine_columns(return_result=self.hyperparams['return_result'],
                                               add_index_columns=self.hyperparams['add_index_columns'],
                                               inputs=inputs, column_indices=self._training_indices,
                                               columns_list=output_columns)
        # outputs.to_csv('BK.csv')
        outputs.fillna(0,inplace=True)
        return CallResult(outputs)
        
    
    @classmethod
    def _get_columns_to_fit(cls, inputs: Inputs, hyperparams: Hyperparams): # pragma: no cover
        """
        Select columns to fit.
        Args:
            inputs: Container DataFrame
            hyperparams: d3m.metadata.hyperparams.Hyperparams

        Returns:
            list
        """
        if not hyperparams['use_semantic_types']:
            return inputs, list(range(len(inputs.columns)))

        inputs_metadata = inputs.metadata

        def can_produce_column(column_index: int) -> bool:
            return cls._can_produce_column(inputs_metadata, column_index, hyperparams)

        use_columns = []
        exclude_columns = []

        # if hyperparams['columns_using_method'] == 'name':
        #     inputs_cols = inputs.columns.values.tolist()
        #     for i in range(len(inputs_cols)):
        #         if inputs_cols[i] in hyperparams['use_columns_name']:
        #             use_columns.append(i)
        #         elif inputs_cols[i] in hyperparams['exclude_columns_name']:
        #             exclude_columns.append(i)      
        # else: 
        use_columns=hyperparams['use_columns']
        exclude_columns=hyperparams['exclude_columns']           
        
        columns_to_produce, columns_not_to_produce = base_utils.get_columns_to_use(inputs_metadata, use_columns=use_columns, exclude_columns=exclude_columns, can_use_column=can_produce_column)
        return inputs.iloc[:, columns_to_produce], columns_to_produce
        # return columns_to_produce

    @classmethod
    def _can_produce_column(cls, inputs_metadata: metadata_base.DataMetadata, column_index: int, hyperparams: Hyperparams) -> bool: # pragma: no cover
        """
        Output whether a column can be processed.
        Args:
            inputs_metadata: d3m.metadata.base.DataMetadata
            column_index: int

        Returns:
            bool
        """
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        accepted_structural_types = (int, float, numpy.integer, numpy.float64)
        accepted_semantic_types = set()
        accepted_semantic_types.add("https://metadata.datadrivendiscovery.org/types/Attribute")
        if not issubclass(column_metadata['structural_type'], accepted_structural_types):
            return False

        semantic_types = set(column_metadata.get('semantic_types', []))

        if len(semantic_types) == 0:
            cls.logger.warning("No semantic types found in column metadata")
            return False
        
        # Making sure all accepted_semantic_types are available in semantic_types
        if len(accepted_semantic_types - semantic_types) == 0:
            return True

        return False
    
    
    @classmethod
    def _update_predictions_metadata(cls, inputs_metadata: metadata_base.DataMetadata, outputs: Optional[Outputs],
                                     target_columns_metadata: List[OrderedDict]) -> metadata_base.DataMetadata:
        """
        Updata metadata for selected columns.
        Args:
            inputs_metadata: metadata_base.DataMetadata
            outputs: Container Dataframe
            target_columns_metadata: list

        Returns:
            d3m.metadata.base.DataMetadata
        """
        outputs_metadata = metadata_base.DataMetadata().generate(value=outputs)

        for column_index, column_metadata in enumerate(target_columns_metadata):
            column_metadata.pop("structural_type", None)
            outputs_metadata = outputs_metadata.update_column(column_index, column_metadata)

        return outputs_metadata

    def _wrap_predictions(self, inputs: Inputs, predictions: ndarray) -> Outputs:
        """
        Wrap predictions into dataframe
        Args:
            inputs: Container Dataframe
            predictions: array-like data (n_samples, n_features)

        Returns:
            Dataframe
        """
        outputs = d3m_dataframe(predictions, generate_metadata=True)
        target_columns_metadata = self._add_target_columns_metadata(outputs.metadata, self.hyperparams)
        outputs.metadata = self._update_predictions_metadata(inputs.metadata, outputs, target_columns_metadata)
        return outputs


    @classmethod
    def _add_target_columns_metadata(cls, outputs_metadata: metadata_base.DataMetadata, hyperparams):
        """
        Add target columns metadata
        Args:
            outputs_metadata: metadata.base.DataMetadata
            hyperparams: d3m.metadata.hyperparams.Hyperparams

        Returns:
            List[OrderedDict]
        """
        outputs_length = outputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']
        target_columns_metadata: List[OrderedDict] = []
        for column_index in range(outputs_length):
            column_name = "output_{}".format(column_index)
            column_metadata = OrderedDict()
            semantic_types = set()
            semantic_types.add(hyperparams["return_semantic_type"])
            column_metadata['semantic_types'] = list(semantic_types)

            column_metadata["name"] = str(column_name)
            target_columns_metadata.append(column_metadata)

        return target_columns_metadata


    def _bkfilter(self, X, low, high, K):
        """
        Perform BKFilter
        Args:
            X: slected rows to be performed
            K, low, high: Parameters of BKFilter

        Returns:
            Dataframe, results of BKFilter
        """
        transformed_X = utils.pandas.DataFrame()
        for col in X.columns:
            cycle = sm.tsa.filters.bkfilter(X[col], low=low, high=high, K=K)
            cycle_df = utils.pandas.DataFrame(cycle)
            transformed_X = utils.pandas.concat([transformed_X,cycle_df], axis=1)

        return transformed_X

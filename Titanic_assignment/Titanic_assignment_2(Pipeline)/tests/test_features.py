from classification_model.config.core import config
from classification_model.processing.features import ExtractLetterTransformer


def test_temporal_variable_transformer(sample_input_data):
    # Given
    variable_name = config.model_config.cabin_var_imputation
    extracter = ExtractLetterTransformer(variables=variable_name)
    
    # Ensure sample input data is correct
    assert sample_input_data["cabin"].iat[1] == "E40"

    # When
    transformed_data = extracter.fit_transform(sample_input_data)
    
    # Then
    assert transformed_data["cabin"].iat[1] == "E"


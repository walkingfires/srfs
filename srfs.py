import pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp, levene
from typeguard import typechecked
import numpy as np
from sklearn.preprocessing import LabelEncoder
import utils


def extract_feature_classes(feature_names):
    feature_classes = set()
    for feature in feature_names:
        _, feature_class, _ = feature.split('_')
        feature_classes.add(feature_class)
    return list(feature_classes)


@typechecked
def srfs(df: pd.DataFrame,
         labels: pd.Series,
         yaml_config: bool = False,
         model_name: str = 'selected') -> pd.DataFrame:
    unique_values = set(labels)

    if unique_values == 2:
        if unique_values != {0, 1}:
            le = LabelEncoder()
            labels = pd.Series(le.fit_transform(labels))
    else:
        raise "Labels does not contain only 2 unique values"

    df = statistic_reduction(df, labels)
    df = full_remove_high_intraclass_correlations(df)
    if yaml_config:
        utils.yaml_maker(df.columns, model_name)
    return df


def full_remove_high_intraclass_correlations(data: pd.DataFrame, threshold_icc=0.7) -> pd.DataFrame:
    features_classes = extract_feature_classes(data.columns)
    for feature_class in features_classes:
        feature_class_columns = data.filter(like=feature_class).columns
        intraclass_corr_matrix = (data.corr(method='spearman').abs().loc[feature_class_columns, feature_class_columns] -
                                  np.eye(len(feature_class_columns)))
        number_of_high_correlations = (intraclass_corr_matrix > threshold_icc).sum(axis=1)
        if len(number_of_high_correlations) > 1 and all(
                value == (len(number_of_high_correlations) - 1) for value in number_of_high_correlations):
            intraclass_ranked_features = intraclass_corr_matrix.rank(method='max').sum(axis=1)
            top_features = intraclass_ranked_features[
                intraclass_ranked_features == max(intraclass_ranked_features.values)].index.tolist()
            data.drop(list(set(feature_class_columns) - set(top_features)), axis=1, inplace=True)
    return data


def statistic_reduction(df: pd.DataFrame, labels: pd.Series, show_plot=False) -> pd.DataFrame:
    selected_features = []
    for column in df.columns:
        sample_with_zero = df[labels == 0][column]
        sample_with_one = df[labels == 1][column]

        u_p_value = mannwhitneyu(sample_with_zero, sample_with_one, alternative='two-sided').pvalue
        ks_p_value = ks_2samp(sample_with_zero, sample_with_one).pvalue
        levene_p_value = levene(sample_with_zero, sample_with_one)[1]

        IQR_sample_with_zero = sample_with_zero.quantile(0.75) - sample_with_zero.quantile(0.25)
        IQR_sample_with_one = sample_with_one.quantile(0.25) - sample_with_one.quantile(0.25)
        scope_sample_with_zero = ((sample_with_zero.quantile(0.25) - 1.7 * IQR_sample_with_zero) <=
                                  sample_with_zero.mean() <=
                                  (sample_with_zero.quantile(0.75) + 1.7 * IQR_sample_with_zero))
        scope_sample_with_one = ((sample_with_one.quantile(0.25) - 1.7 * IQR_sample_with_one) <=
                                 sample_with_one.mean() <=
                                 (sample_with_one.quantile(0.75) + 1.7 * IQR_sample_with_one))

        if ks_p_value < 0.01 and (levene_p_value < 0.05 and u_p_value < 0.05) and (
                scope_sample_with_zero and scope_sample_with_one):

            selected_features.append(column)

            if show_plot:
                utils.showing_plot(sample_with_zero, sample_with_one, column)

    return df[selected_features]


# def print_corr_matrix(data):
#     fig = px.imshow(data.corr(method='spearman').abs(), text_auto=True, width=1000, height=800)
#     fig.show()



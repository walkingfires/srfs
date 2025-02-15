import pandas as pd
import plotly.figure_factory as ff
import yaml


def showing_plot(sample_1, sample_2, label):
    colors = ['orange', 'magenta']
    group_labels = ['Sample 1', 'Sample 2']

    fig = ff.create_distplot([sample_1, sample_2], group_labels, show_hist=False,
                             colors=colors)

    fig.add_vline(x=sample_1.mean(), line_color='orange', annotation_text='Mean Label 0',
                  annotation_position='top left')
    fig.add_vrect(x0=sample_1.quantile(0.25), x1=sample_1.quantile(0.75), line_width=0,
                  fillcolor='orange', opacity=0.2)
    fig.add_vline(x=sample_2.mean(), line_color='magenta', annotation_text='Mean Label 1',
                  annotation_position='top right')
    fig.add_vrect(x0=sample_2.quantile(0.25), x1=sample_2.quantile(0.75), line_width=0,
                  fillcolor='magenta', opacity=0.2)

    fig.update_layout(title=dict(text=f'{label}'))

    fig.show()


def yaml_maker(selected_features: pd.DataFrame,
               model_name: str,
               binCount: int = 32,
               label: int = 1,
               symmetricalGLCM: bool = True,
               correctMask: bool = True) -> None:
    data = {
        'setting': {
            'binCount': binCount,
            'label': label,
            'symmetricalGLCM': symmetricalGLCM,
            'correctMask': correctMask
        },
        'imageType': {
            'Original': {}
        },
        'featureClass': {
        }
    }

    for feature in selected_features:
        feature_type = feature.split('_')[1]
        if feature_type not in data['featureClass'].keys():
            data['featureClass'][feature_type] = []
            data['featureClass'][feature_type].append(feature.split('_')[2])
        else:
            data['featureClass'][feature_type].append(feature.split('_')[2])

    with open(f'{model_name}_extracting_params.yaml', 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

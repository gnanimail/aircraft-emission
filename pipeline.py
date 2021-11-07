import kfp
from kfp import dsl
from kfp import compiler

def preprocess_op():

    return dsl.ContainerOp(
        name='Preprocess Data',
        image='gcr.io/psychic-kite-329307/ae_pipeline_preprocessing',
        arguments=[],
        file_outputs={
            'x_train': '/app/x_train.npy',
            'x_test': '/app/x_test.npy',
            'y_train': '/app/y_train.npy',
            'y_test': '/app/y_test.npy',
        }
    )

def train_op(x_train, y_train):

    return dsl.ContainerOp(
        name='Train Model',
        image='gcr.io/psychic-kite-329307/ae_pipeline_train',
        arguments=[
            '--x_train', x_train,
            '--y_train', y_train
        ],
        file_outputs={
            'model': '/app/model.pkl'
        }
    )

def test_op(x_test, y_test, model):

    return dsl.ContainerOp(
        name='Test Model',
        image='gcr.io/psychic-kite-329307/ae_pipeline_test',
        arguments=[
            '--x_test', x_test,
            '--y_test', y_test,
            '--model', model
        ],
        file_outputs={}
    )

def deploy_model_op(model):

    return dsl.ContainerOp(
        name='Deploy Model',
        image='gcr.io/psychic-kite-329307/ae_pipeline_api',
        arguments=[
            '--model', model
        ]
    )

@dsl.pipeline(
   name='Aircraft Emission',
   description='An example pipeline that trains and logs a regression model.'
)
def aircraft_emission_pipeline():
    _preprocess_op = preprocess_op()
    
    _train_op = train_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['x_train']),
        dsl.InputArgumentPath(_preprocess_op.outputs['y_train'])
    ).after(_preprocess_op)

    _test_op = test_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['x_test']),
        dsl.InputArgumentPath(_preprocess_op.outputs['y_test']),
        dsl.InputArgumentPath(_train_op.outputs['model'])
    ).after(_train_op)

    deploy_model_op(
        dsl.InputArgumentPath(_train_op.outputs['model'])
    ).after(_test_op)

#client = kfp.Client()
#client.create_run_from_pipeline_func(boston_pipeline, arguments={})
compiler.Compiler().compile(aircraft_emission_pipeline, 'aircraft_emission_pipeline.yaml')

import kfp
from kfp import dsl
from kfp import compiler


def preprocess_op():
    return dsl.ContainerOp(
        name='Preprocess Data',
        image='ae/ae_pipeline_preprocessing:latest',
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
        image='ae/ae_pipeline_train:latest',
        arguments=[
            '--x_train', x_train,
            '--y_train', y_train
        ],
        file_outputs={
            'model': '/app/ae_model.pkl'
        }
    )


def api_op(model):
    return dsl.ContainerOp(
        name='Predict API',
        image='ae/ae_pipeline_api:latest',
        arguments=[
            '--model', model
        ],
        file_outputs={}
    )

@dsl.pipeline(
   name='Aircraft Emission Pipeline',
   description='Pipeline that trains and predict the aircraft emission.'
)
def aircraft_emission_pipeline():
    _preprocess_op = preprocess_op()
    
    _train_op = train_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['x_train']),
        dsl.InputArgumentPath(_preprocess_op.outputs['y_train'])
    ).after(_preprocess_op)

    _api_op = api_op(
        dsl.InputArgumentPath(_train_op.outputs['model'])
    ).after(_train_op)


if __name__ == '__main__':
    client = kfp.Client()
    #compiler.Compiler().compile(aircraft_emission_pipeline, 'aircraft_emission_pipeline.tar.gz')
    client.create_run_from_pipeline_func(aircraft_emission_pipeline, arguments={})

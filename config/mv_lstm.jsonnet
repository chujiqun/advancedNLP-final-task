{
    "dataset_reader": {
        "type": "sem",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        }
    },
    "train_data_path": "/home/chujiqun/SNLI/train.txt",
    "validation_data_path": "/home/chujiqun/SNLI/dev.txt",
    "evaluate_on_test": true,
    "model": {
        "type": "mv_lstm",
        "dropout": 0.2,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "pretrained_file": "/home/chujiqun/SNLI/glove.840B.300d.zip",
                    "embedding_dim": 300,
                    "trainable": false
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": 300,
            "hidden_size": 128,
            "num_layers": 1, 
            "bidirectional": true
        },
        "matching_layer": {
            "k": 128,
            "d": 256,
            "activation": "relu"
        },
        "pool_layer": {
            "k": 5
        },
        "output_feedforward": {
            "input_dim": 640,
            "num_layers": 2,
            "hidden_dims": [256, 1],
            "activations": ["relu", "linear"],
            "dropout": [0.2,0.0]
        }
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["Orgquestion", "num_tokens"],
                         ["Relquestion", "num_tokens"]],
        "batch_size": 32
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "validation_metric": "+Auc",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 2,
            "min_lr": 0.00001
        },
        "optimizer": {
            "type": "rmsprop",
            "lr": 0.0001,
        }
    }
}

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
    "model": {
        "type": "match_pyramid",
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
        "matching_layer": {
            "matching_type": "dot"
        },
        "inference_encoder": {
            "input_channel": 1,
            "num_layers": 4,
            "kernel_nums": [8, 16, 32, 64],
            "kernel_sizes": [[3,3],[3,3],[3,3],[3,3]],
            "activations": "relu"
        },
        "pool_layer": {
            "pool_type": "max",
            "output_shape": [2, 2],
        },
        "output_feedforward": {
            "input_dim": 256,
            "num_layers": 2,
            "hidden_dims": [128, 1],
            "activations": ["relu", "linear"],
            "dropout": [0.2, 0.0]
        }
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["Orgquestion", "num_tokens"],
                         ["Relquestion", "num_tokens"]],
        "batch_size": 32
    },
    "trainer": {
        "num_epochs": 50,
        "cuda_device": 0,
        "validation_metric": "-loss",
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 2,
            "min_lr": 0.00001
        },
        "optimizer": {
            "type": "adam",
            "lr": 0.001,
            "weight_decay": 0.001
        }
    }
}

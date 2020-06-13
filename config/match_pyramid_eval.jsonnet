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
    "train_data_path": "/home1/chujiqun/similarity/train_aug.txt",
    "validation_data_path":  "/home1/chujiqun/similarity/dev.txt",
    "model": {
        "type": "match_pyramid",
        "dropout": 0.0,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    "trainable": false
                }
            }
        },
        "matching_layer": {
            "matching_type": "cos"
        },
        "inference_encoder": {
            "input_channel": 1,
            "num_layers": 4,
            "kernel_nums": [8, 16, 32, 64],
            "kernel_sizes": [[1,1],[3,3],[5,5],[5,5]],
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
            "dropout": [0.0, 0.0]
        }
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["Orgquestion", "num_tokens"],
                         ["Relquestion", "num_tokens"]],
        "batch_size": 32
    }
}

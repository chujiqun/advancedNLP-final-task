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
        "dropout": 0.0,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    "trainable": true
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
            "dropout": [0.0,0.0]
        }
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["Orgquestion", "num_tokens"],
                         ["Relquestion", "num_tokens"]],
        "batch_size": 32
    }
}

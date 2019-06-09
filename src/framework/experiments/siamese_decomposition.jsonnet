local data_path = "../data/";

local emb_size = 1024;
{
  "dataset_reader": {
    "type": "reader",
  },
  "train_data_path": data_path + "train",
  "validation_data_path": data_path + "dev",
  "test_data_path": data_path + "test",
  "evaluate_on_test": true,

  "model": {
    "type": "siamese_decomposition",

     "syntax": {
        "input_dim": emb_size,
        "num_layers": 2,
        "hidden_dims": [512, 512],
        "activations": "linear",
        "dropout": 0.0
      },

      "semantic": {
        "input_dim": emb_size,
        "num_layers": 2,
        "hidden_dims": [512, 512],
        "activations": "linear",
        "dropout": 0.0
      },

      "inverse": {
        "input_dim": emb_size,
        "num_layers": 3,
        "hidden_dims": [512, 512, 1024],
        "activations": "linear",
        "dropout": 0.0
      },

      "semantic_predictor": {
        "input_dim": 512,
        "num_layers": 3,
        "hidden_dims": [512, 512, 512],
        "activations": "linear",
        "dropout": 0.0
      },

      "siamese": {
        "input_dim": 512,
        "num_layers": 3,
        "hidden_dims": [512, 256, 2],
        "activations": ["relu", "relu", 'linear'],
        "dropout": 0.0
      },
  },
  "iterator": {
    "type": "basic",
    "batch_size": 64
  },
  "trainer": {
    "num_epochs": 50,
    "grad_norm": 1.0,
    "patience" : 5,
    "cuda_device" : -1,
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 2
    },
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam",
    }
  }
}

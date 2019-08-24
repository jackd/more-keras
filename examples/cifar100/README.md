# Example usage

Quick start

```bash
cd examples/cifar100
python -m more_keras --mk_config=train --config='smoke-test'
python -m more_keras --mk_config=eval --config='smoke-test'
```

Note you can be in a different directory, so long as you use the relative path to the relevant config file (possibly without the extension). Gin includes are, by default, relative to the directory containing the file with the include line.

Longer training

```bash
python -m more_keras --mk_config=train --config='base'
python -m more_keras --mk_config=eval --config='base'
```

Customized

```bash
python -m more_keras --mk_config=train --config='base' --bindings='
@get_cifar_model.conv_filters = (32, 64, 128)
@get_cifar_model.dense_units = 1024
model_dir = "models/big"
'
python -m more_keras --config='models/big/operative-config,$MK_CONFIG_DIR/eval'
```

Hyperparameter search with `ray`
TODO

<!-- ```bash
python -m more_keras --mk_config='ray/ahb' --config='configs/search'
``` -->

# Example usage

Quick start

```bash
cd examples/cifar100
python -m more_keras --mk_config=train --config='configs/smoke-test'
python -m more_keras --mk_config=eval --config='configs/smoke-test'
```

Longer training

```bash
python -m more_keras --mk_config=train --config='configs/base'
python -m more_keras --mk_config=eval --config='configs/base'
```

Customized

```bash
python -m more_keras --mk_config=train --config='configs/base' --bindings='
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

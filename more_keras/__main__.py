"""
Generic entry-point for programs predominantly configured via `gin-config`.

Parses arbitrary config files and bindings. It is assumed that one of these
configures a `main.fn`.

Positional args are interpreted as paths to config files, with `.gin` appended
if missing. The environment variable `MK_CONFIG` is defined before passing as
`more_keras/configs` (though be sure to include quotes if it is not defined
in your environment). See the files contained there for examples that declare
undefined macros (which are expected to be defined in other config files).

Usage:
```python
python -m more_keras '$MK_CONFIG/train.gin' rel/path/to/config \\
    --bindings='f.x = 3' --binding='
        y = @g()
        g.param = 5
    ' \\
    --log_dir=/tmp/my_log  # absl flag
```

General notes:
    - bindings may be repeated or separated by new-lines
    - by default, included files in config files are interpreted first as
        relative to the directory containing the config file, and if missing
        relative to the current working directory. You can disable relative
        includes using `--incl_rel=False`
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
from more_keras import cli
from more_keras import session
from more_keras import utils


def main(argv):
    gin_summary = cli.get_gin_summary(argv)
    gin_summary.enable_path_options()
    gin_summary.parse(finalize=True)
    cli.logging_config()
    logging.info(gin_summary.pretty_format())
    utils.proc()
    session.SessionOptions().configure_session()
    cli.main()


if __name__ == '__main__':
    app.run(main)

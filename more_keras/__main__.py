from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from more_keras import cli
from more_keras import session
from more_keras import utils


def main(argv):
    cli.parse_cli_config()
    utils.proc()
    cli.assert_clargs_parsed(argv)
    session.SessionOptions().configure_session()
    cli.main()


if __name__ == '__main__':
    app.run(main)

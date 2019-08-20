# PR underway: https://github.com/google/gin-config/pull/26
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import copy


class GinState(object):

    def __init__(self, copy_state=False):
        if copy_state:
            self._config = copy.deepcopy(gin.config._CONFIG)
            self._imported_modules = copy.deepcopy(gin.config._IMPORTED_MODULES)
            self._operative_config = copy.deepcopy(gin.config._OPERATIVE_CONFIG)
            self._singletons = copy.deepcopy(gin.config._SINGLETONS)
            self._config_is_locked = gin.config._CONFIG_IS_LOCKED
            self._interactive_mode = gin.config._INTERACTIVE_MODE
        else:
            self._config = {}
            self._imported_modules = set()
            self._operative_config = {}
            self._singletons = {}
            self._config_is_locked = False
            self._interactive_mode = False

    _stack = []

    def _open(self):
        # global _CONFIG
        # global _IMPORTED_MODULES
        # global _OPERATIVE_CONFIG
        # global _SINGLETONS
        # global _CONFIG_IS_LOCKED
        # global _INTERACTIVE_MODE
        gin.config._CONFIG = self._config
        gin.config._IMPORTED_MODULES = self._imported_modules
        gin.config._SINGLETONS = self._singletons
        gin.config._OPERATIVE_CONFIG = self._operative_config
        gin.config._CONFIG_IS_LOCKED = self._config_is_locked
        gin.config._INTERACTIVE_MODE = self._interactive_mode
        return self

    def _leave(self):
        self._config_is_locked = gin.config._CONFIG_IS_LOCKED
        self._interactive_mode = gin.config._INTERACTIVE_MODE

    def __enter__(self):
        GinState._stack[-1]._leave()
        self._open()
        GinState._stack.append(self)
        return self

    def __exit__(self, type, value, traceback):
        top = GinState._stack.pop()
        self._leave()
        assert (top is self)
        GinState._stack[-1]._open()


GinState._stack.append(GinState()._open())  # ensure the stack is never empty

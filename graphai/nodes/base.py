import inspect
from typing import Any, Callable, Dict, Optional

from graphai.callback import Callback
from semantic_router.utils.logger import logger


class NodeMeta(type):
    @staticmethod
    def positional_to_kwargs(cls_type, args) -> Dict[str, Any]:
        init_signature = inspect.signature(cls_type.__init__)
        init_params = {name: arg for name, arg in init_signature.parameters.items() if name != "self"}
        return init_params

    def __call__(cls, *args, **kwargs):
        named_positional_args = NodeMeta.positional_to_kwargs(cls, args)
        kwargs.update(named_positional_args)
        return super().__call__(**kwargs)


class _Node:
    def __init__(
        self,
        is_router: bool = False,
    ):
        self.is_router = is_router

    def _node(
        self,
        func: Callable,
        start: bool = False,
        end: bool = False,
        stream: bool = False,
    ) -> Callable:
        """Decorator validating node structure.
        """
        if not callable(func):
            raise ValueError("Node must be a callable function.")
        
        func_signature = inspect.signature(func)
        parameters = list(func_signature.parameters.values())

        # Automatically add 'callback' parameter if 'stream' is True and 'callback' is not in parameters
        if stream and 'callback' not in func_signature.parameters:
            callback_param = inspect.Parameter(
                'callback',
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Callback
            )
            parameters.append(callback_param)
            # update the function signature
            func_signature = func_signature.replace(parameters=parameters)
            func.__signature__ = func_signature

        class NodeClass:
            _func_signature = func_signature
            is_router = None

            def __init__(self, *args, callback=None, **kwargs):
                if self.stream:
                    self.callback = callback
                bound_args = self._func_signature.bind(*args, **kwargs)
                bound_args.apply_defaults()
                for name, value in bound_args.arguments.items():
                    setattr(self, name, value)

            def execute(self):
                # Bind the current instance attributes to the function signature
                logger.info(f"JB TEMP !!!: {self.__dict__=}")
                bound_args = self._func_signature.bind(**self.__dict__)
                logger.info(f"JB TEMP !!!: {bound_args=}")
                bound_args.apply_defaults()
                # Prepare arguments, including callback if stream is True
                args_dict = bound_args.arguments.copy()  # Copy arguments to modify safely
                if self.stream and 'callback' in self.__dict__:
                    args_dict['callback'] = self.callback
                return func(**args_dict)  # Pass only the necessary arguments

            @classmethod
            def get_signature(cls):
                """Returns the signature of the decorated function as LLM readable
                string.
                """
                signature_components = []
                if NodeClass._func_signature:
                    for param in NodeClass._func_signature.parameters.values():
                        if param.default is param.empty:
                            signature_components.append(f"{param.name}: {param.annotation}")
                        else:
                            signature_components.append(f"{param.name}: {param.annotation} = {param.default}")
                else:
                    return "No signature"
                return "\n".join(signature_components)

            @classmethod
            def invoke(cls, input: Dict[str, Any], callback: Optional[Callback] = None):
                logger.info(f"JB TEMP !!!: {input=}")
                if callback and cls.stream:
                    self.callback = callback
                instance = cls(**input)
                out = instance.execute()
                return out

        NodeClass.__name__ = func.__name__
        NodeClass.name = func.__name__
        NodeClass.__doc__ = func.__doc__
        NodeClass.is_start = start
        NodeClass.is_end = end
        NodeClass.is_router = self.is_router
        NodeClass.stream = stream

        return NodeClass

    def __call__(
        self,
        func: Optional[Callable] = None,
        start: bool = False,
        end: bool = False,
        stream: bool = False,
    ):
        # We must wrap the call to the decorator in a function for it to work
        # correctly with or without parenthesis
        def wrap(func: Callable, start=start, end=end, stream=stream) -> Callable:
            return self._node(func=func, start=start, end=end, stream=stream)
        if func:
            # Decorator is called without parenthesis
            return wrap(func=func, start=start, end=end, stream=stream)
        # Decorator is called with parenthesis
        return wrap


node = _Node()
router = _Node(is_router=True)

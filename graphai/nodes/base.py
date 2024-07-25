import inspect
from typing import Any, Callable, Dict, Optional


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
    def __init__(self):
        self._func_signature = None

    def _node(self, func: Callable) -> Callable:
        """Decorator validating node structure.
        """
        if not callable(func):
            raise ValueError("Node must be a callable function.")
        
        func_signature = inspect.signature(func)

        class NodeClass:
            _func_signature = func_signature

            def __init__(self, *args, **kwargs):
                bound_args = self._func_signature.bind(*args, **kwargs)
                bound_args.apply_defaults()
                for name, value in bound_args.arguments.items():
                    setattr(self, name, value)

            def execute(self):
                # Bind the current instance attributes to the function signature
                bound_args = self._func_signature.bind(**self.__dict__)
                bound_args.apply_defaults()
                return func(*bound_args.args, **bound_args.kwargs)

            @classmethod
            def invoke(cls, input: Dict[str, Any]):
                instance = cls(**input)
                return instance.execute()

        NodeClass.__name__ = func.__name__
        NodeClass.__doc__ = func.__doc__

        return NodeClass

    def __call__(self, func: Optional[Callable] = None):
        # We must wrap the call to the decorator in a function for it to work
        # correctly with or without parenthesis
        def wrap(func: Callable) -> Callable:
            return self._node(func)
        if func:
            # Decorator is called without parenthesis
            return wrap(func)
        # Decorator is called with parenthesis
        return wrap
    
    def _get_signature(self) -> inspect.Signature:
        """Returns the signature of the decorated function.
        """
        return self._func_signature
    
    def signature(self):
        """Returns the signature of the decorated function as LLM readable
        string.
        """
        signature_str = ""
        if self._func_signature:
            for param in self._func_signature.parameters.values():
                if param.default is param.empty:
                    signature_str += f"{param.name}: {param.annotation}"
                else:
                    signature_str += f"{param.name}: {param.annotation} = {param.default}"
        else:
            return "No signature"
        return signature_str


node = _Node()

# init for trying to adopt a logic similar to openAI gym
# of registration envs, but here registrating routines
from importlib import import_module


class RoutineSpec(object):
    """Specification for a particular instance of the routine. Used
    to register the parameters given routines.
    Args:
    id (str): The official routine ID
    kwargs (dict): The kwargs to pass to the routine class
    """

    def load(name):
        mod_name, attr_name = name.split(":")
        mod = import_module(mod_name)
        fn = getattr(mod, attr_name)
        return fn

    def __init__(self, id, entry_point=None, kwargs=None):
        data = {}
        self.id = id
        self.entry_point = entry_point
        self.kwargs = kwargs

        self.company_name = data.balabla_company_name

    def make(self, **kwargs):
        """Instantiates an instance of the routine with appropriate kwargs"""
        error = {}
        if self.entry_point is None:
            raise error.Error(
                "Attempting nonexistent routine {}".format(self.id)
            )
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            routine = self.entry_point(**_kwargs)
        else:
            load = object
            cls = load(self.entry_point)
            routine = cls(**_kwargs)

        # Make the routine aware of which spec it came from.
        copy = {}
        spec = copy.deepcopy(self)
        spec._kwargs = _kwargs
        routine.unwrapped.spec = spec

        return routine

    def __repr__(self):
        return "RoutineSpec({})".format(self.id)


class RoutineRegistry(object):
    """Register an routine by ID."""

    def __init__(self):
        self.routine_specs = {}

        self.logger = object()

    def make(self, id, **kwargs):
        if len(kwargs) > 0:
            self.logger.info("Making new routine: %s (%s)", id, kwargs)
        else:
            self.logger.info("Making new routine: %s", id)
        spec = self.spec(id)
        routine = spec.make(**kwargs)

        return routine

    def all(self):
        return self.routine_specs.values()

    def register(self, id, **kwargs):
        error = {}
        if id in self.routine_specs:
            raise error.Error("ID already registered: {}".format(id))
        self.routine_specs[id] = RoutineSpec(id, **kwargs)


# Have a global registry
registry = RoutineRegistry()


def register(id, **kwargs):
    return registry.register(id, **kwargs)

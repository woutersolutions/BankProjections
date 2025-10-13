from bank_projections.projections.base_registry import BaseRegistry


class IFRS9Stage:
    def __init__(self, is_default: bool):
        self.is_default = is_default


class IFRS9StageRegistry(BaseRegistry[IFRS9Stage]):
    pass


IFRS9StageRegistry.register("1", IFRS9Stage(is_default=False))
IFRS9StageRegistry.register("2", IFRS9Stage(is_default=False))
IFRS9StageRegistry.register("3", IFRS9Stage(is_default=True))
IFRS9StageRegistry.register("Poci", IFRS9Stage(is_default=True))
IFRS9StageRegistry.register("N/a", IFRS9Stage(is_default=False))

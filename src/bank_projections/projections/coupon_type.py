from bank_projections.projections.base_registry import BaseRegistry


class CouponType:
    pass


class CouponTypeRegistry(BaseRegistry[CouponType]):
    pass


CouponTypeRegistry.register("fixed", CouponType())
CouponTypeRegistry.register("floating", CouponType())
CouponTypeRegistry.register("none", CouponType())

from .registration_2 import register
import settings

# from .static import StaticSetup

# init for trying to adopt a logic similar to openAI gym of registration envs,
# but here registrating routines


def _load_default_setup_settings():
    return settings.DEFAULT_CONFIG.copy()


register(
    id="static_setup",
    entry_point="setups:StaticSetup",
    kwargs=_load_default_setup_settings(),
)

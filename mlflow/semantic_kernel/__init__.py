from mlflow.semantic_kernel.autolog import patched_class_call
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

FLAVOR_NAME = "semantic_kernel"


@experimental
@autologging_integration(FLAVOR_NAME)
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from Anthropic to MLflow.
    Only synchronous calls are supported. Asynchnorous APIs and streaming are not recorded.

    Args:
        log_traces: If ``True``, traces are logged for Anthropic models.
            If ``False``, no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the Anthropic autologging. Default to ``False``.
        silent: If ``True``, suppress all event logs and warnings from MLflow during Anthropic
            autologging. If ``False``, show all events and warnings.
    """
    from semantic_kernel import Kernel

    safe_patch(
        FLAVOR_NAME,
        Kernel,
        "invoke",
        patched_class_call,
    )

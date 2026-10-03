from dataclasses import dataclass

from app.pipeline.data_evidence import DataEvidence
from app.pipeline.policy_evidence import PolicyEvidence
from app.routing.router import Route


ORDER_NOT_FOUND_MESSAGE = (
    "I couldn't find that order or any matching orders on your account."
)
POLICY_NOT_FOUND_MESSAGE = (
    "I don't have a Kartly policy that covers that, so I can't answer it reliably."
)
OUT_OF_SCOPE_MESSAGE = "I can only help with your Kartly orders and store policies."


@dataclass(frozen=True, slots=True)
class GroundingResult:
    ok: bool
    message: str | None


def check_grounding(
    route: Route,
    data_evidence: DataEvidence,
    policy_evidence: PolicyEvidence,
) -> GroundingResult:
    """Require the evidence appropriate for the selected route."""
    if route == "data":
        if data_evidence.found:
            return GroundingResult(ok=True, message=None)
        return GroundingResult(ok=False, message=ORDER_NOT_FOUND_MESSAGE)

    if route == "policy":
        if policy_evidence.found:
            return GroundingResult(ok=True, message=None)
        return GroundingResult(ok=False, message=POLICY_NOT_FOUND_MESSAGE)

    if route == "combined":
        if not data_evidence.found:
            return GroundingResult(ok=False, message=ORDER_NOT_FOUND_MESSAGE)
        if not policy_evidence.found:
            return GroundingResult(ok=False, message=POLICY_NOT_FOUND_MESSAGE)
        return GroundingResult(ok=True, message=None)

    return GroundingResult(ok=False, message=OUT_OF_SCOPE_MESSAGE)

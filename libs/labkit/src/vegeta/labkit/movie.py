"""Movie helpers shared by the labs' scenarios."""
from __future__ import annotations


def end_card(images: list, ep, fps: int = 25, hold_s: float = 1.5) -> list:
    """The movie's last frame held for ``hold_s`` with the outcome written on it (red: a failure, green: success),
    so a failed mission's movie ends at the frame where it failed."""
    import cv2

    o = ep.outcome
    ok = bool(o.get("success"))
    text = ("FINISHED" if ok else "FAILED: " + str(o.get("reason", "")).upper()) + f"  t = {o.get('t_end', 0):.1f} s"
    detail = str(o.get("detail", ""))
    last = images[-1].copy()
    colour = (46, 125, 50) if ok else (198, 40, 40)
    cv2.rectangle(last, (0, last.shape[0] - 70), (last.shape[1], last.shape[0]), (20, 20, 20), -1)
    cv2.putText(last, text, (16, last.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)
    cv2.putText(last, detail[:90], (16, last.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return list(images) + [last] * int(round(hold_s * fps))
